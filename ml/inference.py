import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(levelname)-8s | %(message)s",
	datefmt="%H:%M:%S",
)
log = logging.getLogger("SCITTSE.Inference")


COMPOUND_TO_ORDINAL = {
	"SOFT": 0,
	"MEDIUM": 1,
	"HARD": 2,
}


@dataclass
class InferenceSettings:
	model_path: Path = Path(os.getenv("SCITTSE_MODEL_PATH", "./models/tire_degradation_model.ubj"))
	circuit_encoding_path: Path = Path(os.getenv("SCITTSE_CIRCUIT_PATH", "./models/circuit_encoding.json"))
	model_config_path: Path = Path(os.getenv("SCITTSE_MODEL_CONFIG_PATH", "./models/model_config.json"))
	host: str = os.getenv("SCITTSE_INFERENCE_HOST", "0.0.0.0")
	port: int = int(os.getenv("SCITTSE_INFERENCE_PORT", "8001"))


class TireDegradationRequest(BaseModel):
	tyre_life: int = Field(..., ge=0)
	compound: str = Field(..., description="SOFT, MEDIUM, or HARD")
	lap_number: int = Field(..., ge=1)
	stint_laps_ahead: int = Field(5, ge=1, le=80)

	circuit: str
	fresh_tyre: int = Field(0, ge=0, le=1)

	track_temp: float
	air_temp: float
	humidity: float
	pressure: float
	wind_speed: float

	session_progress_pct: float | None = Field(default=None, ge=0.0, le=1.0)
	total_race_laps: int | None = Field(default=None, ge=1)
	era_encoded: int | None = Field(default=None, ge=0, le=1)
	year: int | None = Field(default=None, ge=2018)

	@field_validator("compound")
	@classmethod
	def validate_compound(cls, value: str) -> str:
		normalized = value.strip().upper()
		if normalized not in COMPOUND_TO_ORDINAL:
			raise ValueError("compound must be SOFT, MEDIUM, or HARD")
		return normalized


class TireDegradationInferenceService:
	def __init__(self, settings: InferenceSettings):
		self.settings = settings
		self.model: xgb.XGBRegressor | None = None
		self.circuit_map: dict[str, float] = {}
		self.model_config: dict[str, Any] = {}
		self.feature_cols: list[str] = []

	def load(self) -> None:
		model_path = self.settings.model_path
		circuit_path = self.settings.circuit_encoding_path
		config_path = self.settings.model_config_path

		for required in (model_path, circuit_path, config_path):
			if not required.exists():
				raise FileNotFoundError(f"Required artifact missing: {required.resolve()}")

		model = xgb.XGBRegressor()
		model.load_model(str(model_path))
		self.model = model

		with open(circuit_path, "r", encoding="utf-8") as f:
			self.circuit_map = json.load(f)

		with open(config_path, "r", encoding="utf-8") as f:
			self.model_config = json.load(f)

		self.feature_cols = list(self.model_config.get("feature_cols", []))
		if not self.feature_cols:
			raise ValueError("model_config.json is missing feature_cols")

		log.info("Inference artifacts loaded successfully.")
		log.info("Model path: %s", model_path.resolve())
		log.info("Circuit map path: %s", circuit_path.resolve())
		log.info("Model config path: %s", config_path.resolve())

	def _infer_era_encoded(self, request: TireDegradationRequest) -> int:
		if request.era_encoded is not None:
			return int(request.era_encoded)
		if request.year is not None:
			return int(request.year >= 2022)
		return 1

	def _infer_session_progress(self, lap_number: int, request: TireDegradationRequest) -> float:
		if request.total_race_laps:
			return min(1.0, lap_number / request.total_race_laps)
		if request.session_progress_pct is not None:
			return float(request.session_progress_pct)
		return min(1.0, lap_number / 60.0)

	def _feature_row(self, request: TireDegradationRequest, lap_offset: int) -> dict[str, float]:
		projected_tyre_life = request.tyre_life + lap_offset
		projected_lap_number = request.lap_number + lap_offset

		compound_ordinal = COMPOUND_TO_ORDINAL[request.compound]
		fresh_tyre = int(request.fresh_tyre)
		session_progress_pct = self._infer_session_progress(projected_lap_number, request)
		era_encoded = self._infer_era_encoded(request)

		circuit_encoded = self.circuit_map.get(
			request.circuit,
			self.circuit_map.get("__global_mean__", 0.0),
		)

		computed = {
			"TyreLife": float(projected_tyre_life),
			"TyreLifeSquared": float(projected_tyre_life ** 2),
			"CompoundOrdinal": float(compound_ordinal),
			"FreshTyre": float(fresh_tyre),
			"FreshTyre_x_TyreLife": float(fresh_tyre * projected_tyre_life),
			"LapNumber": float(projected_lap_number),
			"SessionProgressPct": float(session_progress_pct),
			"CircuitEncoded": float(circuit_encoded),
			"TrackTemp": float(request.track_temp),
			"AirTemp": float(request.air_temp),
			"Humidity": float(request.humidity),
			"Pressure": float(request.pressure),
			"WindSpeed": float(request.wind_speed),
			"EraEncoded": float(era_encoded),
		}

		return {feature: float(computed.get(feature, 0.0)) for feature in self.feature_cols}

	def predict_stint_curve(self, request: TireDegradationRequest) -> dict[str, Any]:
		if self.model is None:
			raise RuntimeError("Model not loaded. Ensure startup completed successfully.")

		rows = [
			self._feature_row(request=request, lap_offset=offset)
			for offset in range(1, request.stint_laps_ahead + 1)
		]
		feature_frame = pd.DataFrame(rows, columns=self.feature_cols)
		predictions = self.model.predict(feature_frame)

		points = []
		for offset, predicted_lap_time in enumerate(predictions, start=1):
			points.append(
				{
					"lap_offset": offset,
					"projected_lap_number": request.lap_number + offset,
					"projected_tyre_life": request.tyre_life + offset,
					"predicted_lap_time_s": round(float(predicted_lap_time), 3),
				}
			)

		first_lap = points[0]["predicted_lap_time_s"]
		final_lap = points[-1]["predicted_lap_time_s"]
		curve_delta = round(float(final_lap - first_lap), 3)

		return {
			"request": request.model_dump(),
			"projection": points,
			"summary": {
				"first_projected_lap_time_s": first_lap,
				"last_projected_lap_time_s": final_lap,
				"stint_time_delta_s": curve_delta,
			},
		}


settings = InferenceSettings()
service = TireDegradationInferenceService(settings=settings)

app = FastAPI(title="SCITTSE Tire Degradation Inference")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
	service.load()


@app.get("/health")
def health() -> dict[str, str]:
	return {"status": "ok"}


@app.get("/model/info")
def model_info() -> dict[str, Any]:
	return {
		"model_path": str(settings.model_path),
		"circuit_encoding_path": str(settings.circuit_encoding_path),
		"model_config_path": str(settings.model_config_path),
		"feature_count": len(service.feature_cols),
		"features": service.feature_cols,
	}


@app.post("/predict/tire-degradation")
def predict_tire_degradation(payload: TireDegradationRequest) -> dict[str, Any]:
	return service.predict_stint_curve(payload)


@app.websocket("/ws/predict/tire-degradation")
async def predict_tire_degradation_ws(websocket: WebSocket) -> None:
	await websocket.accept()
	try:
		while True:
			request_json = await websocket.receive_json()
			payload = TireDegradationRequest(**request_json)
			result = service.predict_stint_curve(payload)
			await websocket.send_json(result)
	except WebSocketDisconnect:
		log.info("Inference websocket client disconnected.")
	except Exception as exc:
		await websocket.send_json({"error": str(exc)})
		await websocket.close(code=1003)


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(
		"inference:app",
		host=settings.host,
		port=settings.port,
		reload=False,
	)
