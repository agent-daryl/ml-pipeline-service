from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.serving.schemas import (
    PredictionRequest, PredictionResponse, HealthResponse, HousingFeatures,
)
from src.serving.metrics import (
    REQUEST_COUNTER, PREDICTION_LATENCY, PREDICTION_VALUE,
    DATA_DRIFT_SCORE, get_prometheus_metrics, set_prometheus_content_type,
)
from src.monitoring.validator import DataValidator

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "housing_model.joblib"
STATS_PATH = BASE_DIR / "data" / "dataset_stats.json"

app = FastAPI(title="ML Pipeline Service", version="1.0.0")
start_time = time.time()

model_artifact = None
validator = None


@app.on_event("startup")
def load_model():
    global model_artifact, validator

    if not MODEL_PATH.exists():
        logger.error("Model file not found at %s. Run scripts/train.py first.", MODEL_PATH)
        model_artifact = None
        return

    model_artifact = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)

    if STATS_PATH.exists():
        with open(STATS_PATH) as f:
            stats = json.load(f)
        validator = DataValidator(stats)
        logger.info("Data validator initialized with baseline stats")


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy" if model_artifact else "degraded_no_model",
        model_loaded=model_artifact is not None,
        model_version="1.0.0",
        uptime_seconds=round(time.time() - start_time, 2),
    )


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    if model_artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        features = np.array([[
            request.features.MedInc,
            request.features.HouseAge,
            request.features.AveRooms,
            request.features.AveBedrms,
            request.features.Population,
            request.features.AveOccup,
            request.features.Latitude,
            request.features.Longitude,
        ]], dtype=np.float64)

        scaler = model_artifact["scaler"]
        model = model_artifact["model"]
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        latency = time.time() - start

        REQUEST_COUNTER.labels(endpoint="/predict", status="success").inc()
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_VALUE.observe(prediction)

        confidence = "high" if 0.5 < prediction < 5.0 else "medium" if 0.1 < prediction < 8.0 else "low"

        return PredictionResponse(
            predicted_value=round(float(prediction), 4),
            confidence=confidence,
            features_used=model_artifact["feature_names"],
        )

    except Exception as e:
        REQUEST_COUNTER.labels(endpoint="/predict", status="error").inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
def validate(request: PredictionRequest) -> dict:
    if validator is None:
        raise HTTPException(status_code=503, detail="Validator not initialized — no baseline stats")

    feature_dict = request.features.model_dump()
    report = validator.validate_single(feature_dict)
    drift = report.get("drift_score", 0.0)
    DATA_DRIFT_SCORE.observe(drift)
    REQUEST_COUNTER.labels(endpoint="/validate", status="success").inc()

    return {
        "valid": report["is_valid"],
        "drift_score": round(drift, 4),
        "warnings": report["warnings"],
        "feature_checks": {k: {"in_range": v["in_range"], "z_score": round(v["z_score"], 2)} for k, v in report["feature_checks"].items()},
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(content=get_prometheus_metrics().decode(), media_type=set_prometheus_content_type())
