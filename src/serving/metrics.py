from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

METRICS_DIR = Path(__file__).resolve().parent.parent.parent / "data"

REQUEST_COUNTER = Counter(
    "ml_pipeline_requests_total",
    "Total number of prediction requests",
    ["endpoint", "status"],
)

PREDICTION_LATENCY = Histogram(
    "ml_pipeline_prediction_latency_seconds",
    "Latency of prediction requests in seconds",
)

PREDICTION_VALUE = Histogram(
    "ml_pipeline_prediction_value",
    "Distribution of prediction output values (median house value in $100k)",
    buckets=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0],
)

DATA_DRIFT_SCORE = Histogram(
    "ml_pipeline_data_drift_score",
    "Data drift score from baseline statistics (higher = more drift)",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0],
)


def get_prometheus_metrics() -> bytes:
    return generate_latest()


def set_prometheus_content_type():
    return CONTENT_TYPE_LATEST
