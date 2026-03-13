import xgboost as xgb  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import logging
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SCITTSE.TireModel")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    DATASET_PATH: str = "./scittse_tire_dataset.parquet"
    MODEL_OUTPUT:  str = "./models/tire_degradation_model.ubj"
    METRICS_OUTPUT: str = "./models/tire_model_metrics.json"

    # Temporal train/test split — NEVER random split on time-series race data.
    # Train on earlier seasons, test on the most recent held-out seasons.
    # This mirrors real deployment: the model is trained before the season starts.
    TRAIN_YEARS_END: int = 2023   # train on 2018–2023
    TEST_YEARS_START: int = 2024  # test on 2024–2025

    # Features the model trains on.
    # DegradationRate_s_per_lap and DeltaToStintBest_s are EXCLUDED —
    # they are computed from FuelCorrectedLapTime (the target) and would
    # cause data leakage. During real-time inference we don't have future
    # lap times to compute them from anyway.
    FEATURE_COLS: list = field(default_factory=lambda: [
        # Tire state — CompoundOrdinal is sufficient for XGBoost.
        # One-hot columns (Compound_SOFT/MEDIUM/HARD) are redundant:
        # they sum to 1 (linear dependence) and XGBoost handles ordinal
        # splits natively on the ordinal encoding.
        "TyreLife",
        "TyreLifeSquared",
        "CompoundOrdinal",
        "FreshTyre",
        "FreshTyre_x_TyreLife",

        # Race context
        "LapNumber",
        "SessionProgressPct",

        # Circuit identity (target-encoded from training set)
        "CircuitEncoded",

        # Ambient conditions
        "TrackTemp",
        "AirTemp",
        "Humidity",
        "Pressure",
        "WindSpeed",

        # Regulatory era (encoded as int)
        "EraEncoded",
    ])

    TARGET_COL: str = "FuelCorrectedLapTime"

    # XGBoost hyperparameters — tuned for this regression task.
    # These are strong defaults; fine-tune with Optuna in a future pass.
    XGB_PARAMS: dict = field(default_factory=lambda: {
        "objective":        "reg:squarederror",
        "eval_metric":      "rmse",
        "n_estimators":     1000,
        "learning_rate":    0.05,       # low LR + high n_estimators = better generalization
        "max_depth":        6,          # deep enough for compound×temp interactions
        "min_child_weight": 10,         # prevents overfitting on small stints
        "subsample":        0.8,        # row sampling per tree
        "colsample_bytree": 0.8,        # feature sampling per tree
        "gamma":            0.1,        # min loss reduction to split — regularization
        "reg_alpha":        0.1,        # L1 regularization
        "reg_lambda":       1.0,        # L2 regularization
        "early_stopping_rounds": 50,    # stop if val RMSE doesn't improve for 50 rounds
        "tree_method":      "hist",     # fastest CPU training method
        "random_state":     42,
        "n_jobs":           -1,
    })


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_validate(cfg: TrainConfig) -> pd.DataFrame:
    path = Path(cfg.DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run dataset_builder.py first."
        )

    df = pd.read_parquet(path)
    log.info(f"Loaded dataset: {len(df):,} rows × {len(df.columns)} cols")

    # Validate required columns exist
    required = set(cfg.FEATURE_COLS + [cfg.TARGET_COL, "Year", "Compound", "Circuit"])
    # EraEncoded and CircuitEncoded are derived here — not in the parquet
    required.discard("EraEncoded")
    required.discard("CircuitEncoded")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Encode Era string → integer for XGBoost
    era_map = {"PRE_2022": 0, "GROUND_EFFECT": 1}
    if "Era" in df.columns:
        df["EraEncoded"] = df["Era"].map(era_map).fillna(0).astype(int)
    else:
        # Derive from year if Era column absent (older dataset version)
        df["EraEncoded"] = (df["Year"] >= 2022).astype(int)

    # Drop rows missing any training feature or target (excluding derived cols)
    derived_cols = {"CircuitEncoded"}
    # Weather columns get imputed below — don't drop rows for their NaNs
    weather_cols = {"TrackTemp", "AirTemp", "Humidity", "Pressure", "WindSpeed"}
    cols_needed = [c for c in cfg.FEATURE_COLS if c not in derived_cols and c not in weather_cols] + [cfg.TARGET_COL]
    before = len(df)
    df = df.dropna(subset=cols_needed)
    dropped = before - len(df)
    if dropped:
        log.info(f"Dropped {dropped:,} rows with NaN in non-weather features/target.")

    # Impute weather NaNs with per-session medians (Year+Round groups).
    # FastF1 weather data often has gaps for Pressure/WindSpeed.
    # Session medians are stable and preserve within-race variation.
    present_weather = [c for c in weather_cols if c in df.columns]
    if present_weather:
        nan_before = df[present_weather].isna().sum()
        nan_total = nan_before.sum()
        if nan_total > 0:
            if "Round" in df.columns:
                session_medians = df.groupby(["Year", "Round"])[present_weather].transform("median")
                df[present_weather] = df[present_weather].fillna(session_medians)
            else:
                log.warning("'Round' column missing — skipping per-session weather imputation, using global median only.")
            # If entire sessions lack weather, fall back to global median
            global_medians = df[present_weather].median()
            df[present_weather] = df[present_weather].fillna(global_medians)
            log.info(
                f"Imputed {nan_total:,} weather NaNs (per-session median, then global fallback): "
                + ", ".join(f"{c}={nan_before[c]:,}" for c in present_weather if nan_before[c] > 0)
            )

    log.info(f"Clean dataset: {len(df):,} rows | Years: {df['Year'].min()}–{df['Year'].max()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame, cfg: TrainConfig):
    """
    Split by season, not randomly.

    Random splits on race data leak future information into training —
    a model that has seen lap 40 of the 2024 Bahrain race during training
    will trivially predict lap 41. Temporal split forces the model to
    generalize across unseen seasons, which is what deployment actually requires.
    """
    train_df = df[df["Year"] <= cfg.TRAIN_YEARS_END]
    test_df  = df[df["Year"] >= cfg.TEST_YEARS_START]

    if test_df.empty:
        log.warning(
            f"No test data found for years >= {cfg.TEST_YEARS_START}. "
            f"Falling back to last season in dataset as test set."
        )
        last_year = df["Year"].max()
        train_df  = df[df["Year"] < last_year]
        test_df   = df[df["Year"] == last_year]

    log.info(f"Train: {len(train_df):,} laps | Years: {train_df['Year'].min()}–{train_df['Year'].max()}")
    log.info(f"Test:  {len(test_df):,} laps  | Years: {test_df['Year'].min()}–{test_df['Year'].max()}")

    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT TARGET ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_circuits(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    cfg: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Target-encode Circuit using mean FuelCorrectedLapTime from the TRAINING set,
    with Bayesian smoothing toward the global mean for low-sample circuits.

    Why target encoding over label encoding:
      - Label encoding (Circuit → arbitrary int) creates false ordinal
        relationships. XGBoost would learn that "Abu Dhabi" (0) < "Bahrain" (1)
        which is meaningless.
      - One-hot encoding adds 20+ sparse features for 20+ circuits, most of
        which XGBoost won't use efficiently.
      - Target encoding maps each circuit to its training-set mean lap time,
        which is a meaningful numeric signal (fast vs slow tracks) and adds
        only 1 feature.

    Bayesian smoothing: For circuits with few training laps, the raw mean is
    noisy.  We shrink toward the global mean weighted by sample size:
      encoded = (n * circuit_mean + m * global_mean) / (n + m)
    where m is the smoothing strength (higher = more regularization).
    A circuit with 1000 laps barely shrinks; one with 20 laps shrinks heavily.

    Leakage prevention: encoding is computed ONLY from train_df.
    Unseen circuits in test_df get the global training mean as fallback.
    """
    SMOOTHING_M = 100  # strength parameter — circuits with <100 laps shrink noticeably

    global_mean = train_df[cfg.TARGET_COL].mean()
    circuit_stats = train_df.groupby("Circuit")[cfg.TARGET_COL].agg(["mean", "count"])

    # Bayesian shrinkage: (n * local_mean + m * global_mean) / (n + m)
    circuit_stats["smoothed"] = (
        (circuit_stats["count"] * circuit_stats["mean"] + SMOOTHING_M * global_mean)
        / (circuit_stats["count"] + SMOOTHING_M)
    )
    circuit_map = circuit_stats["smoothed"].to_dict()

    # Log circuits where smoothing made a material difference (>0.5s shift)
    raw_map = circuit_stats["mean"].to_dict()
    smoothed_circuits = [
        (c, raw_map[c], circuit_map[c])
        for c in circuit_map
        if abs(raw_map[c] - circuit_map[c]) > 0.5
    ]
    if smoothed_circuits:
        log.info("Circuit smoothing adjustments (>0.5s shift):")
        for c, raw, smoothed in smoothed_circuits:
            log.info(f"  {c:<20} raw={raw:.2f}s → smoothed={smoothed:.2f}s "
                     f"(n={int(circuit_stats.loc[c, 'count'])})")

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["CircuitEncoded"] = train_df["Circuit"].map(circuit_map)
    test_df["CircuitEncoded"] = test_df["Circuit"].map(circuit_map).fillna(global_mean)

    unseen = set(test_df["Circuit"].unique()) - set(circuit_map.keys())
    if unseen:
        log.warning(f"Unseen circuits in test set (using global mean): {unseen}")

    log.info(f"Circuit encoding: {len(circuit_map)} circuits mapped | "
             f"range: {min(circuit_map.values()):.2f}s – {max(circuit_map.values()):.2f}s")

    # Include global_mean in the map so inference can handle new circuits
    circuit_map["__global_mean__"] = float(global_mean)
    circuit_map["__smoothing_m__"] = SMOOTHING_M
    return train_df, test_df, circuit_map


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(X_train, y_train, X_test, y_test, cfg: TrainConfig,
          verbose: int = 100) -> xgb.XGBRegressor:
    """
    Train XGBoost with a proper validation split carved from the training data.

    Early stopping uses a held-out validation slice (last 20% of training laps
    by index, which preserves temporal ordering) — NOT the test set.  Using the
    test set for early stopping leaks information: the model's best iteration
    is chosen to minimize test error, making reported test metrics optimistic.
    """
    log.info("Training XGBoost tire degradation model...")

    # Copy params to avoid mutating the config
    params = {k: v for k, v in cfg.XGB_PARAMS.items() if k != "early_stopping_rounds"}
    early_stopping_rounds = cfg.XGB_PARAMS.get("early_stopping_rounds", 50)

    # Carve a validation set from the tail of training data (temporal ordering).
    # 80/20 split preserves chronological order — the model trains on earlier
    # laps and validates on more recent laps within the training window.
    val_size = int(len(X_train) * 0.2)
    if val_size < 50:
        # Too little data for a meaningful val split — fall back to training
        # set only (no early stopping benefit, but won't crash)
        log.warning(f"Training set too small for val split ({len(X_train)} rows). "
                    f"Using full training set without early stopping.")
        X_fit, y_fit = X_train, y_train
        eval_sets = [(X_train, y_train)]
    else:
        X_fit   = X_train.iloc[:-val_size]
        y_fit   = y_train.iloc[:-val_size]
        X_val   = X_train.iloc[-val_size:]
        y_val   = y_train.iloc[-val_size:]
        eval_sets = [(X_fit, y_fit), (X_val, y_val)]
        log.info(f"Val split: {len(X_fit):,} train / {len(X_val):,} val")

    model = xgb.XGBRegressor(**params)

    model.fit(
        X_fit, y_fit,
        eval_set=eval_sets,
        verbose=verbose,
        early_stopping_rounds=early_stopping_rounds,
    )

    log.info(f"Training complete. Best iteration: {model.best_iteration}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD TEMPORAL CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_cv(df: pd.DataFrame, cfg: TrainConfig) -> list[dict]:
    """
    Walk-forward validation: train on years up to N, test on year N+1.

    A single temporal split gives ONE RMSE estimate — you can't tell if the
    model is stable or if that one number was lucky. Walk-forward produces
    one metric per fold, revealing:
      - Whether model quality degrades on more recent seasons
      - Whether a regulation change (2022) caused a performance cliff
      - Variance of RMSE across folds (high variance = fragile model)

    Each fold re-encodes circuits from its own training set to prevent leakage.
    """
    years = sorted(df["Year"].unique().tolist())
    if len(years) < 3:
        log.warning("Need at least 3 seasons for walk-forward CV. Skipping.")
        return []

    # Start CV from 3rd year so each fold has ≥2 years of training
    cv_test_years: list[int] = [y for y in years[2:]]  # type: ignore
    results: list[dict] = []

    log.info(f"\n{'═' * 60}")
    log.info(f"WALK-FORWARD CV | {len(cv_test_years)} folds")
    log.info(f"{'═' * 60}")

    for test_year in cv_test_years:
        fold_train = df[df["Year"] < test_year]
        fold_test  = df[df["Year"] == test_year]

        if fold_train.empty or fold_test.empty or len(fold_test) < 50:
            continue

        # Re-encode circuits per fold to prevent leakage
        fold_train, fold_test, _ = encode_circuits(fold_train, fold_test, cfg)

        X_tr = fold_train[cfg.FEATURE_COLS]
        y_tr = fold_train[cfg.TARGET_COL]
        X_te = fold_test[cfg.FEATURE_COLS]
        y_te = fold_test[cfg.TARGET_COL]

        params = {k: v for k, v in cfg.XGB_PARAMS.items() if k != "early_stopping_rounds"}
        early_stopping_rounds = cfg.XGB_PARAMS.get("early_stopping_rounds", 50)
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=0,
            early_stopping_rounds=early_stopping_rounds,
        )

        preds = model.predict(X_te)
        fold_rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        fold_mae  = float(mean_absolute_error(y_te, preds))
        fold_r2   = float(r2_score(y_te, preds))

        fold_result = {
            "test_year":   int(test_year),
            "train_years": f"{int(fold_train['Year'].min())}–{int(fold_train['Year'].max())}",
            "train_laps":  int(len(fold_train)),
            "test_laps":   int(len(fold_test)),
            "rmse":        fold_rmse,
            "mae":         fold_mae,
            "r2":          fold_r2,
        }
        results.append(fold_result)
        log.info(
            f"  Fold {test_year} | train: {fold_result['train_years']} ({len(fold_train):,}) "
            f"| test: {len(fold_test):,} "
            f"| RMSE: {fold_rmse:.4f}s | MAE: {fold_mae:.4f}s | R²: {fold_r2:.4f}"
        )

    if results:
        rmses = [r["rmse"] for r in results]
        log.info(f"\n  CV Summary: mean RMSE={np.mean(rmses):.4f}s | "
                 f"std={np.std(rmses):.4f}s | "
                 f"min={np.min(rmses):.4f}s | max={np.max(rmses):.4f}s")
    log.info(f"{'═' * 60}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: xgb.XGBRegressor, X_train, X_test, y_train, y_test,
             test_df: pd.DataFrame, cfg: TrainConfig) -> dict:
    """
    Evaluate overall, per-compound, and against baselines.

    Per-compound evaluation is critical — a single aggregate RMSE hides
    the fact that Soft prediction might be terrible (high degradation variance)
    while Hard is excellent (stable, linear wear). The race engineer needs
    to know which compound predictions to trust.
    """
    preds = model.predict(X_test)

    overall = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae":  float(mean_absolute_error(y_test, preds)),
        "r2":   float(r2_score(y_test, preds)),
    }

    # ── Overfitting guard ─────────────────────────────────────────────────────
    train_preds = model.predict(X_train)
    train_rmse  = float(np.sqrt(mean_squared_error(y_train, train_preds)))
    test_rmse   = overall["rmse"]
    overfit_ratio = test_rmse / train_rmse if train_rmse > 0 else float("inf")
    if overfit_ratio > 2.0:
        log.warning(
            f"OVERFITTING DETECTED: train RMSE={train_rmse:.4f}s vs test RMSE={test_rmse:.4f}s "
            f"(ratio={overfit_ratio:.2f}x). Consider increasing regularization "
            f"(reg_alpha, reg_lambda, min_child_weight) or reducing max_depth."
        )
    else:
        log.info(f"Overfit check: train RMSE={train_rmse:.4f}s | test RMSE={test_rmse:.4f}s "
                 f"| ratio={overfit_ratio:.2f}x ✔")

    # ── Baselines ─────────────────────────────────────────────────────────
    # Without baselines, an R² of 0.85 is uninterpretable — is that good?
    # These simple predictors establish the floor the model must beat.
    train_mean = float(y_train.mean())
    mean_preds = np.full_like(y_test.values, train_mean, dtype=float)
    baselines = {
        "global_mean": {
            "rmse": float(np.sqrt(mean_squared_error(y_test, mean_preds))),
            "mae":  float(mean_absolute_error(y_test, mean_preds)),
            "r2":   float(r2_score(y_test, mean_preds)),
            "description": "Predict training set global mean for every lap",
        },
    }

    # Per-compound mean baseline — use CompoundOrdinal to recover compound labels
    train_compound_means = {}
    ordinal_to_compound = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}
    if "CompoundOrdinal" in X_train.columns:
        for ordinal, compound in ordinal_to_compound.items():
            mask: pd.Series = X_train["CompoundOrdinal"] == ordinal
            if mask.any():  # type: ignore
                train_compound_means[compound] = float(y_train[mask].mean())
    if train_compound_means:
        compound_mean_preds = test_df["Compound"].map(train_compound_means).fillna(train_mean).values
        baselines["compound_mean"] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, compound_mean_preds))),
            "mae":  float(mean_absolute_error(y_test, compound_mean_preds)),
            "r2":   float(r2_score(y_test, compound_mean_preds)),
            "description": "Predict training set per-compound mean for each lap",
        }

    log.info("─" * 50)
    log.info("BASELINE COMPARISONS")
    for name, bl in baselines.items():
        log.info(f"  {name:<16} | RMSE: {bl['rmse']:.4f}s | MAE: {bl['mae']:.4f}s | R²: {bl['r2']:.4f}")
    rmse_lift = float(baselines["global_mean"]["rmse"]) - float(overall["rmse"])
    log.info(f"  Model RMSE lift over global mean: {rmse_lift:+.4f}s")
    log.info("OVERALL TEST SET METRICS")
    log.info(f"  RMSE : {overall['rmse']:.4f} s")
    log.info(f"  MAE  : {overall['mae']:.4f} s")
    log.info(f"  R²   : {overall['r2']:.4f}")

    # Per-compound breakdown
    per_compound: dict[str, dict[str, float]] = {}
    log.info("\nPER-COMPOUND METRICS")
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        mask: pd.Series = test_df["Compound"] == compound
        if mask.sum() < 10:  # type: ignore
            continue
        c_preds  = preds[mask.values]
        c_actual = y_test[mask]
        per_compound[compound] = {
            "rmse":  float(np.sqrt(mean_squared_error(c_actual, c_preds))),
            "mae":   float(mean_absolute_error(c_actual, c_preds)),
            "r2":    float(r2_score(c_actual, c_preds)),
            "n_laps": float(mask.sum()),  # type: ignore
        }
        log.info(
            f"  {compound:<8} | RMSE: {per_compound[compound]['rmse']:.4f}s "
            f"| MAE: {per_compound[compound]['mae']:.4f}s "
            f"| R²: {per_compound[compound]['r2']:.4f} "
            f"| n={per_compound[compound]['n_laps']:,}"
        )

    # Per-era breakdown — validates model generalizes across regulation changes
    per_era: dict[str, dict[str, float]] = {}
    log.info("\nPER-ERA METRICS")
    if "Era" in test_df.columns:
        for era in test_df["Era"].unique():
            mask: pd.Series = test_df["Era"] == era
            if mask.sum() < 10:  # type: ignore
                continue
            e_preds  = preds[mask.values]
            e_actual = y_test[mask]
            per_era[era] = {
                "rmse":  float(np.sqrt(mean_squared_error(e_actual, e_preds))),
                "mae":   float(mean_absolute_error(e_actual, e_preds)),
                "r2":    float(r2_score(e_actual, e_preds)),
                "n_laps": float(mask.sum()),  # type: ignore
            }
            log.info(
                f"  {era:<16} | RMSE: {per_era[era]['rmse']:.4f}s "
                f"| R²: {per_era[era]['r2']:.4f} "
                f"| n={per_era[era]['n_laps']:,}"
            )

    # Per-circuit breakdown — reveals which tracks the model struggles on
    per_circuit: dict[str, dict[str, float]] = {}
    if "Circuit" in test_df.columns:
        log.info("\nPER-CIRCUIT METRICS (top 5 worst RMSE)")
        for circuit in test_df["Circuit"].unique():
            mask: pd.Series = test_df["Circuit"] == circuit
            if mask.sum() < 10:  # type: ignore
                continue
            ci_preds  = preds[mask.values]
            ci_actual = y_test[mask]
            per_circuit[circuit] = {
                "rmse":  float(np.sqrt(mean_squared_error(ci_actual, ci_preds))),
                "mae":   float(mean_absolute_error(ci_actual, ci_preds)),
                "r2":    float(r2_score(ci_actual, ci_preds)),
                "n_laps": float(mask.sum()),  # type: ignore
            }
        # Log worst circuits first — these need attention
        worst: list[tuple[str, dict[str, float]]] = sorted(per_circuit.items(), key=lambda x: x[1]["rmse"], reverse=True)
        import itertools
        for circuit, m in itertools.islice(worst, 5):
            log.info(
                f"  {circuit:<20} | RMSE: {m['rmse']:.4f}s "
                f"| MAE: {m['mae']:.4f}s "
                f"| R²: {m['r2']:.4f} "
                f"| n={m['n_laps']:,}"
            )

    # Feature importance — top 10
    log.info("\nFEATURE IMPORTANCE (top 10 by gain)")
    feature_importance: dict[str, float] = model.get_booster().get_score(importance_type="gain")
    importance_list: list[tuple[str, float]] = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    import itertools
    for feat, score in itertools.islice(importance_list, 10):
        log.info(f"  {feat:<30} {score:.2f}")
    log.info("─" * 50)

    return {
        "overall":      overall,
        "baselines":    baselines,
        "per_compound": per_compound,
        "per_era":      per_era,
        "per_circuit":  per_circuit,
        "best_iteration": model.best_iteration,
        "feature_importance": dict(importance_list),
        "train_years_end":   cfg.TRAIN_YEARS_END,
        "test_years_start":  cfg.TEST_YEARS_START,
        "feature_cols":      cfg.FEATURE_COLS,
        "target_col":        cfg.TARGET_COL,
        "dataset_stats": {
            "train_laps": int(len(X_train)),
            "test_laps":  int(len(y_test)),
            "train_rmse": float(train_rmse),
            "overfit_ratio": float(round(float(overfit_ratio), 3)),  # type: ignore
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save(model: xgb.XGBRegressor, metrics: dict, cfg: TrainConfig,
         circuit_map: dict | None = None,
         feature_dtypes: dict | None = None):
    model_path   = Path(cfg.MODEL_OUTPUT)
    metrics_path = Path(cfg.METRICS_OUTPUT)

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # .ubj = Universal Binary JSON — XGBoost's native binary format.
    # Faster to load than .json, fully reproducible, version-stable.
    model.save_model(str(model_path))
    log.info(f"Model saved  → {model_path.resolve()}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved → {metrics_path.resolve()}")

    # Save circuit encoding map for inference — the inference service needs
    # this to encode new circuit names to the same numeric values.
    if circuit_map:
        circuit_path = model_path.parent / "circuit_encoding.json"
        with open(circuit_path, "w") as f:
            json.dump(circuit_map, f, indent=2)
        log.info(f"Circuit map  → {circuit_path.resolve()}")

    # Save model config for inference — self-documenting artifact that tells
    # the inference service which features to provide and in what order.
    model_config = {
        "feature_cols": cfg.FEATURE_COLS,
        "target_col":   cfg.TARGET_COL,
        "feature_dtypes": feature_dtypes or {col: "float64" for col in cfg.FEATURE_COLS},
        "model_format":  "ubj",
        "xgb_version":   xgb.__version__,
    }
    config_path = model_path.parent / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    log.info(f"Model config → {config_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCITTSE Tire Degradation Model Trainer")
    parser.add_argument("--dataset",         type=str, default="./scittse_tire_dataset.parquet")
    parser.add_argument("--model-output",    type=str, default="./models/tire_degradation_model.ubj")
    parser.add_argument("--metrics-output",  type=str, default="./models/tire_model_metrics.json")
    parser.add_argument("--train-years-end", type=int, default=2023)
    parser.add_argument("--test-years-start",type=int, default=2024)
    parser.add_argument("--n-estimators",    type=int, default=1000)
    parser.add_argument("--learning-rate",   type=float, default=0.05)
    parser.add_argument("--max-depth",       type=int, default=6)
    parser.add_argument("--walk-forward-cv",  action="store_true",
                        help="Run walk-forward temporal CV before final training")
    parser.add_argument("--verbose",          type=int, default=100,
                        help="XGBoost training log frequency (0=silent, 100=every 100 rounds)")
    args = parser.parse_args()

    cfg = TrainConfig(
        DATASET_PATH=args.dataset,
        MODEL_OUTPUT=args.model_output,
        METRICS_OUTPUT=args.metrics_output,
        TRAIN_YEARS_END=args.train_years_end,
        TEST_YEARS_START=args.test_years_start,
    )
    cfg.XGB_PARAMS["n_estimators"]  = args.n_estimators
    cfg.XGB_PARAMS["learning_rate"] = args.learning_rate
    cfg.XGB_PARAMS["max_depth"]     = args.max_depth

    df                                    = load_and_validate(cfg)

    # Optional walk-forward CV — run before final training to assess stability
    cv_results = []
    if args.walk_forward_cv:
        cv_results = walk_forward_cv(df, cfg)

    train_df, test_df                     = temporal_split(df, cfg)
    train_df, test_df, circuit_map        = encode_circuits(train_df, test_df, cfg)
    X_train = train_df[cfg.FEATURE_COLS]
    y_train = train_df[cfg.TARGET_COL]
    X_test  = test_df[cfg.FEATURE_COLS]
    y_test  = test_df[cfg.TARGET_COL]
    model                                 = train(X_train, y_train, X_test, y_test, cfg,
                                                   verbose=args.verbose)
    metrics                               = evaluate(model, X_train, X_test, y_train, y_test, test_df, cfg)
    if cv_results:
        metrics["walk_forward_cv"] = cv_results
    real_dtypes = {col: str(X_train[col].dtype) for col in cfg.FEATURE_COLS}
    save(model, metrics, cfg, circuit_map, feature_dtypes=real_dtypes)