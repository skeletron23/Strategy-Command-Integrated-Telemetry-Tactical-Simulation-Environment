import fastf1
import pandas as pd
import numpy as np
import logging
import time
import threading
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SCITTSE.DatasetBuilder")

# ---------------------------------------------------------------------------
# Named constants — no magic numbers in pipeline logic
# ---------------------------------------------------------------------------
MIN_VALID_LAPTIME_S = 60.0    # No modern F1 lap is under 1 minute
MAX_VALID_LAPTIME_S = 200.0   # No F1 lap exceeds ~3:20
SC_FUEL_BURN_FACTOR = 0.6     # SC laps burn ~60% of normal fuel
WEATHER_MERGE_TOLERANCE = pd.Timedelta("3min")
SC_MERGE_NAN_WARN_THRESHOLD = 0.05  # Warn if >5% of SC merge rows are NaN-filled

# FastF1's local cache (Parquet/pickle) is NOT thread-safe for concurrent
# writes.  This lock serializes session.load() calls across worker threads
# to prevent cache corruption while still allowing parallel DataFrame work.
_cache_lock = threading.Lock()


def get_regulatory_era(year: int) -> str:
    """Classify season into regulatory era for fuel/aero model selection."""
    if year >= 2022:
        return "GROUND_EFFECT"
    return "PRE_2022"


@dataclass
class BuilderConfig:
    START_YEAR: int = 2019
    END_YEAR:   int = 2025  # 2019-2025 covers PRE_2022 + GROUND_EFFECT eras

    VALID_COMPOUNDS: list = field(
        default_factory=lambda: ["SOFT", "MEDIUM", "HARD"]
    )

    MIN_STINT_LAPS:     int   = 4
    MAX_STINT_LAPS:     int   = 45   # No F1 tire survives >45 racing laps
    QUICKLAP_THRESHOLD: float = 1.07 # lower bound: 107% of session fastest
    UPPER_OUTLIER_S:    float = 5.0  # upper bound: session median + 5s

    START_FUEL_KG:             float = 110.0
    FUEL_TIME_EFFECT_S_PER_KG: float = 0.033  # Barcelona Constant — midpoint of [0.030, 0.035] Pirelli range

    CACHE_DIR:   str = "./fastf1_cache"
    OUTPUT_PATH: str = "./scittse_tire_dataset.parquet"

    SESSION_TYPES: list = field(
        default_factory=lambda: ["R"]  # Add "Sprint" to include sprint races
    )

    HARVEST_WORKERS: int = 1  # Sequential by default; >1 uses cache lock

    # Burn rate is a property of circuit length, not race lap count.
    # Values derived from FIA post-race energy reports.
    CIRCUIT_FUEL_BURN: dict = field(default_factory=lambda: {
        "Bahrain":           2.28,
        "Saudi Arabia":      2.22,
        "Australia":         2.35,
        "Azerbaijan":        2.10,
        "Miami":             2.25,
        "Monaco":            1.55,
        "Spain":             2.30,
        "Canada":            2.05,
        "Great Britain":     2.40,
        "Austria":           1.92,
        "France":            2.20,
        "Hungary":           2.15,
        "Belgium":           2.75,
        "Netherlands":       2.25,
        "Italy":             2.65,
        "Singapore":         2.00,
        "Japan":             2.40,
        "United States":     2.30,
        "Mexico":            2.15,
        "Brazil":            2.28,
        "Las Vegas":         2.10,
        "Abu Dhabi":         2.20,
        "Qatar":             2.32,
        "China":             2.28,
        "Portugal":          2.30,
        "Emilia Romagna":    2.32,
        "Turkey":            2.35,
        "Russia":            2.25,
        "Styria":            1.92,
        "70th Anniversary":  2.40,
    })
    DEFAULT_BURN: float = 2.30


class TireDegradationBuilder:
    """
    Builds a clean, physics-corrected lap dataset for a single race session.

    Each lap row is corrected for fuel mass and enriched with weather,
    stint segmentation, and degradation rate features.
    """

    def __init__(self, year: int, round_number: int, cfg: BuilderConfig,
                 session_type: str = "R"):
        self.year          = year
        self.round_number  = round_number
        self.cfg           = cfg
        self.session_type  = session_type
        self.session       = None
        self.circuit_key   = None
        self.total_laps    = None
        self.era           = get_regulatory_era(year)

    @property
    def start_fuel_kg(self) -> float:
        return self.cfg.START_FUEL_KG

    @property
    def fuel_time_effect(self) -> float:
        return self.cfg.FUEL_TIME_EFFECT_S_PER_KG

    def load_session(self) -> bool:
        """Load a race or sprint session via FastF1."""
        try:
            self.session = fastf1.get_session(
                self.year, self.round_number, self.session_type,
            )
            # Serialize cache I/O across threads — FastF1's pickle/parquet
            # cache is not safe for concurrent writes.
            with _cache_lock:
                self.session.load(
                    laps=True,
                    telemetry=False,
                    weather=True,
                    messages=False,
                    livedata=None,
                )
            event_name       = self.session.event["EventName"]
            self.circuit_key = event_name.replace(" Grand Prix", "").strip()
            self.total_laps  = int(self.session.total_laps) if hasattr(self.session, 'total_laps') else None
            burn = self.cfg.CIRCUIT_FUEL_BURN.get(self.circuit_key, self.cfg.DEFAULT_BURN)
            tag = "Sprint" if self.session_type == "Sprint" else "Race"
            log.info(
                f"  [{self.year} R{self.round_number:02d}] {event_name} ({tag}) "
                f"| Era: {self.era} | Burn: {burn:.2f} kg/lap"
            )
            return True
        except Exception as e:
            log.error(
                f"  [{self.year} R{self.round_number:02d}] Load failed: {e}",
                exc_info=True,
            )
            return False

    def clean_laps(self, laps: pd.DataFrame) -> pd.DataFrame:
        """
        Remove dirty laps: pit in/out, invalid compounds, outlier times.

        The two-sided outlier gate:
          lower = fastest lap × 1.07 keeps legitimate racing laps
          upper = session median + 5s catches residual SC/VSC noise
        """
        clean = laps.pick_track_status('1')

        clean = clean[clean['PitOutTime'].isnull()]
        clean = clean[clean['PitInTime'].isnull()]

        clean = clean[clean['Compound'].isin(self.cfg.VALID_COMPOUNDS)]
        clean = clean.dropna(subset=['LapTime', 'TyreLife', 'Compound']).copy()
        clean['LapTimeSeconds'] = clean['LapTime'].dt.total_seconds()

        session_fastest_s = clean['LapTimeSeconds'].min()
        session_median_s  = clean['LapTimeSeconds'].median()
        lower_bound = session_fastest_s * self.cfg.QUICKLAP_THRESHOLD
        upper_bound = session_median_s  + self.cfg.UPPER_OUTLIER_S

        # P0 FIX: >= lower_bound (keep laps FASTER than 107% cutoff)
        #         <= upper_bound (discard laps SLOWER than median+5s)
        clean = clean[
            (clean['LapTimeSeconds'] >= session_fastest_s) &
            (clean['LapTimeSeconds'] <= lower_bound)       &
            (clean['LapTimeSeconds'] <= upper_bound)       &
            (clean['LapTimeSeconds'] > MIN_VALID_LAPTIME_S)  &
            (clean['LapTimeSeconds'] < MAX_VALID_LAPTIME_S)
        ]
        return clean

    def _count_sc_laps_before(self, laps: pd.DataFrame) -> pd.Series:
        """
        For each driver+lap, count how many of that driver's previous laps
        had non-'1' track status (SC, VSC, yellow).  Used to correct
        cumulative fuel burn — SC laps burn ~40% less fuel.
        """
        all_laps = self.session.laps.sort_values(['Driver', 'LapNumber']).copy()
        all_laps['IsSCLap'] = 0

        try:
            sc_laps = self.session.laps.copy()
            sc_laps['TrackStatusStr'] = sc_laps['TrackStatus'].astype(str)
            sc_laps['IsSCLap'] = (~sc_laps['TrackStatusStr'].isin(['1', ''])).astype(int)
        except Exception:
            # If TrackStatus column is missing/unusable, fall back to zero
            all_laps['IsSCLap'] = 0
            sc_laps = all_laps

        # Cumulative SC laps per driver up to (but not including) current lap
        sc_cumsum = (
            sc_laps.sort_values(['Driver', 'LapNumber'])
            .groupby('Driver')['IsSCLap']
            .cumsum()
            - sc_laps['IsSCLap']  # exclude current lap
        )
        sc_lookup = pd.DataFrame({
            'Driver': sc_laps['Driver'].values,
            'LapNumber': sc_laps['LapNumber'].values,
            'SCLapsBefore': sc_cumsum.values,
        })

        merged = laps.merge(
            sc_lookup, on=['Driver', 'LapNumber'], how='left',
        )

        # Monitor NaN fill rate — high rate signals lap-numbering
        # inconsistencies (e.g. DNF drivers) causing silent fallback
        # to the uncorrected fuel model.
        nan_count = merged['SCLapsBefore'].isna().sum()
        total     = len(merged)
        if total > 0:
            fill_rate = nan_count / total
            if fill_rate > SC_MERGE_NAN_WARN_THRESHOLD:
                log.warning(
                    f"    SC merge: {nan_count}/{total} rows "
                    f"({fill_rate:.1%}) had no match — filled with 0. "
                    f"Possible lap-numbering inconsistency for "
                    f"{self.year} R{self.round_number:02d} {self.circuit_key}"
                )

        return merged['SCLapsBefore'].fillna(0).astype(int)

    def apply_fuel_correction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove fuel mass effect from lap times.

        Physics: Every 10 kg of fuel adds ~0.33s/lap at Barcelona.
        We normalize all laps to "zero fuel" equivalent times.

        SC-aware: Safety car laps burn ~60% of normal fuel, so post-SC
        laps carry more fuel than a naive (N-1)*burn model predicts.
        """
        burn_rate = self.cfg.CIRCUIT_FUEL_BURN.get(self.circuit_key, self.cfg.DEFAULT_BURN)

        df = df.copy()
        df['SCLapsBefore'] = self._count_sc_laps_before(df)

        # Normal laps before current = (LapNumber - 1) - SCLapsBefore
        normal_laps_before = (df['LapNumber'] - 1) - df['SCLapsBefore']
        sc_laps_before     = df['SCLapsBefore']

        df['EstimatedFuelLoad_kg'] = (
            self.start_fuel_kg
            - normal_laps_before * burn_rate
            - sc_laps_before     * burn_rate * SC_FUEL_BURN_FACTOR
        ).clip(lower=0.0)

        df['WeightPenalty_s']      = df['EstimatedFuelLoad_kg'] * self.fuel_time_effect
        df['FuelCorrectedLapTime'] = df['LapTimeSeconds'] - df['WeightPenalty_s']
        return df

    def merge_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-nearest join with 3-min tolerance to avoid stale weather matches."""
        weather = self.session.weather_data
        if weather is None or weather.empty:
            log.warning(f"    No weather data for {self.circuit_key}")
            for col in ['TrackTemp', 'AirTemp', 'Humidity', 'Pressure', 'WindSpeed']:
                df[col] = np.nan
            return df

        weather_cols = weather[['Time', 'TrackTemp', 'AirTemp', 'Humidity', 'Pressure', 'WindSpeed']]
        return pd.merge_asof(
            df.sort_values('Time'),
            weather_cols.sort_values('Time'),
            on='Time',
            direction='nearest',
            tolerance=WEATHER_MERGE_TOLERANCE,
        )

    def assign_stint_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Segment laps into tire stints by detecting TyreLife resets."""
        df = df.sort_values(['Driver', 'LapNumber']).copy()

        def _assign(group):
            tyre_life        = group['TyreLife'].fillna(0).astype(int)
            breaks           = (tyre_life.diff() != 1) & (tyre_life.diff().notna())
            group            = group.copy()
            group['StintID'] = breaks.cumsum()
            return group

        return df.groupby('Driver', group_keys=False).apply(_assign)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for the XGBoost tire degradation model.

        Features:
          - TyreLifeSquared: encodes quadratic degradation cliff
          - DegradationRate: EWMA(span=5) of lap-to-lap delta
          - DeltaToStintBest: gap to best lap in current stint
          - SessionProgressPct: track rubber evolution proxy
          - FreshTyre_x_TyreLife: interaction between tyre newness and age
          - Compound one-hot + ordinal encoding
          - Regulatory era flag
        """
        df = df.copy()

        # Quadratic tire life term
        df['TyreLifeSquared'] = df['TyreLife'] ** 2

        # EWMA degradation rate — smoother than polyfit, catches cliff in 3-4 laps
        def _ewma_deg_rate(group: pd.Series) -> pd.Series:
            return group.diff().ewm(span=5, adjust=False).mean()

        df['DegradationRate_s_per_lap'] = (
            df.groupby(['Driver', 'StintID'])['FuelCorrectedLapTime']
            .transform(_ewma_deg_rate)
        )

        stint_best = df.groupby(['Driver', 'StintID'])['FuelCorrectedLapTime'].transform('min')
        df['DeltaToStintBest_s'] = df['FuelCorrectedLapTime'] - stint_best

        # Track evolution proxy — rubber buildup reduces deg ~0.05-0.15s over a race
        if self.total_laps and self.total_laps > 0:
            df['SessionProgressPct'] = df['LapNumber'] / self.total_laps
        else:
            max_lap = df['LapNumber'].max()
            df['SessionProgressPct'] = df['LapNumber'] / max_lap if max_lap > 0 else 0.0

        # Compound encoding
        df['CompoundOrdinal'] = df['Compound'].map({"SOFT": 0, "MEDIUM": 1, "HARD": 2})
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            df[f'Compound_{compound}'] = (df['Compound'] == compound).astype(int)

        # FreshTyre — cast nullable boolean for XGBoost compatibility
        if 'FreshTyre' in df.columns:
            df['FreshTyre'] = df['FreshTyre'].fillna(False).astype(int)
        else:
            df['FreshTyre'] = 0

        # Interaction: fresh tyres on lap 10 ≠ used tyres on lap 10
        df['FreshTyre_x_TyreLife'] = df['FreshTyre'] * df['TyreLife']

        # Regulatory era — useful for model to learn era-specific behavior
        df['Era'] = self.era

        return df

    def build_dataset(self) -> pd.DataFrame:
        """Full pipeline for a single session: load → clean → correct → feature."""
        if not self.load_session():
            return pd.DataFrame()

        clean = self.clean_laps(self.session.laps)
        if clean.empty:
            return pd.DataFrame()

        clean = self.assign_stint_ids(clean)

        # Two-sided stint length filter: too short = not useful, too long = data error
        stint_lengths = clean.groupby(['Driver', 'StintID'])['LapNumber'].transform('count')
        clean = clean[
            (stint_lengths >= self.cfg.MIN_STINT_LAPS) &
            (stint_lengths <= self.cfg.MAX_STINT_LAPS)
        ]
        if clean.empty:
            return pd.DataFrame()

        clean = self.apply_fuel_correction(clean)
        clean = self.merge_weather(clean)
        clean = self.engineer_features(clean)

        clean['Year']        = self.year
        clean['Round']       = self.round_number
        clean['Circuit']     = self.circuit_key
        clean['SessionType'] = self.session_type

        output_cols = [
            'Year', 'Round', 'Circuit', 'Driver', 'SessionType', 'Era',
            'FuelCorrectedLapTime',
            'TyreLife', 'TyreLifeSquared', 'Compound',
            'CompoundOrdinal', 'Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD',
            'FreshTyre', 'FreshTyre_x_TyreLife',
            'LapNumber', 'StintID', 'SessionProgressPct',
            'TrackTemp', 'AirTemp', 'Humidity', 'Pressure', 'WindSpeed',
            'DegradationRate_s_per_lap', 'DeltaToStintBest_s',
            'LapTimeSeconds', 'EstimatedFuelLoad_kg', 'WeightPenalty_s',
            'SCLapsBefore',
        ]
        output_cols = [c for c in output_cols if c in clean.columns]
        final = clean[output_cols].dropna(subset=['FuelCorrectedLapTime', 'TyreLife', 'Compound'])

        log.info(f"    -> {len(final):,} clean stint laps")
        return final


def _build_one_session(year: int, round_num: int, session_type: str,
                       cfg: BuilderConfig) -> pd.DataFrame:
    """Worker function for parallel harvest — builds one session."""
    builder = TireDegradationBuilder(year, round_num, cfg, session_type=session_type)
    return builder.build_dataset()


def run_full_harvest(cfg: BuilderConfig) -> pd.DataFrame:
    """
    Harvest all race (and optionally sprint) sessions across configured years.

    Default: sequential (workers=1).  Set --workers >1 to parallelize;
    session.load() calls are serialized behind _cache_lock to prevent
    FastF1 cache corruption, while DataFrame transforms run concurrently.
    """
    cache_path = Path(cfg.CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))

    all_frames = []
    processed, skipped = 0, 0

    # Collect all (year, round, session_type) jobs first
    jobs: list[tuple[int, int, str]] = []
    for year in range(cfg.START_YEAR, cfg.END_YEAR + 1):
        log.info(f"\nSeason {year}")
        try:
            schedule    = fastf1.get_event_schedule(year, include_testing=False)
            race_events = schedule[schedule["RoundNumber"] > 0]
        except Exception as e:
            log.error(f"Failed to fetch {year} schedule: {e}", exc_info=True)
            continue

        for _, event in race_events.iterrows():
            round_num = int(event["RoundNumber"])
            for stype in cfg.SESSION_TYPES:
                jobs.append((year, round_num, stype))

    effective_workers = cfg.HARVEST_WORKERS
    if effective_workers > 1:
        log.info(
            f"\nSubmitting {len(jobs)} session jobs with {effective_workers} workers "
            f"(cache writes serialized via lock)"
        )
    else:
        log.info(f"\nProcessing {len(jobs)} sessions sequentially")

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_job = {
            executor.submit(_build_one_session, y, r, s, cfg): (y, r, s)
            for y, r, s in jobs
        }
        for future in as_completed(future_to_job):
            y, r, s = future_to_job[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    all_frames.append(df)
                    processed += 1
                else:
                    skipped += 1
            except Exception as e:
                log.error(f"  [{y} R{r:02d} {s}] Worker exception: {e}", exc_info=True)
                skipped += 1

    log.info(f"\nHarvest done | Processed: {processed} | Skipped: {skipped}")

    if not all_frames:
        log.error("No data collected.")
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


def post_process(df: pd.DataFrame) -> pd.DataFrame:
    """Final quality gate: plausibility filter, dedup, and summary stats."""
    if df.empty:
        return df

    before = len(df)
    df = df[
        (df['FuelCorrectedLapTime'] > MIN_VALID_LAPTIME_S) &
        (df['FuelCorrectedLapTime'] < MAX_VALID_LAPTIME_S)
    ]
    log.info(f"Dropped {before - len(df):,} implausible rows.")

    # Deduplicate in case of overlapping harvests or re-runs
    dup_before = len(df)
    df = df.drop_duplicates(
        subset=['Year', 'Round', 'Driver', 'LapNumber', 'SessionType'],
        keep='last',
    )
    if len(df) < dup_before:
        log.info(f"Removed {dup_before - len(df):,} duplicate rows.")

    log.info(
        f"Rows: {len(df):,} | Drivers: {df['Driver'].nunique()} | "
        f"Circuits: {df['Circuit'].nunique()} | Years: {df['Year'].min()}-{df['Year'].max()}"
    )
    for c, n in df['Compound'].value_counts().items():
        log.info(f"  {c:<8}: {n:,} laps ({n/len(df)*100:.1f}%)")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SCITTSE Tire Degradation Dataset Builder v2")
    parser.add_argument("--start-year",     type=int,   default=2019)
    parser.add_argument("--end-year",       type=int,   default=2025)
    parser.add_argument("--output",         type=str,   default="./scittse_tire_dataset.parquet")
    parser.add_argument("--cache-dir",      type=str,   default="./fastf1_cache")
    parser.add_argument("--min-stint-laps", type=int,   default=4)
    parser.add_argument("--max-stint-laps", type=int,   default=45)
    parser.add_argument("--fuel-effect",    type=float, default=0.033)
    parser.add_argument("--workers",        type=int,   default=1)
    parser.add_argument(
        "--session-types", type=str, nargs="+", default=["R"],
        help='Session types to harvest, e.g. "R" "Sprint"',
    )
    args = parser.parse_args()

    cfg = BuilderConfig(
        START_YEAR=args.start_year,
        END_YEAR=args.end_year,
        OUTPUT_PATH=args.output,
        CACHE_DIR=args.cache_dir,
        MIN_STINT_LAPS=args.min_stint_laps,
        MAX_STINT_LAPS=args.max_stint_laps,
        FUEL_TIME_EFFECT_S_PER_KG=args.fuel_effect,
        SESSION_TYPES=args.session_types,
        HARVEST_WORKERS=args.workers,
    )

    raw_df   = run_full_harvest(cfg)
    final_df = post_process(raw_df)

    if not final_df.empty:
        out = Path(cfg.OUTPUT_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(out, index=False, compression="snappy")
        log.info(f"Saved -> {out.resolve()} | {len(final_df):,} rows x {len(final_df.columns)} cols")
    else:
        log.error("Build failed.")
        exit(1)
