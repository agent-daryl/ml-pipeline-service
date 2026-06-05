from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.ingestion.data_loader import FEATURE_COLUMNS, DatasetStats

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ModelTrainer:
    def __init__(self, model_dir: Path = MODELS_DIR):
        self.model_dir = model_dir
        self.scaler = StandardScaler()
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

    def train(self, X_train, y_train, X_test, y_test):
        logger.info("Fitting feature scaler on training data...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info("Training GradientBoostingRegressor (200 estimators)...")
        start = time.time()
        self.model.fit(X_train_scaled, y_train)
        train_duration = time.time() - start
        logger.info("Training completed in %.2f seconds", train_duration)

        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        metrics = {
            "train_mse": float(mean_squared_error(y_train, y_train_pred)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
            "train_r2": float(r2_score(y_train, y_train_pred)),
            "test_mse": float(mean_squared_error(y_test, y_test_pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
            "test_r2": float(r2_score(y_test, y_test_pred)),
            "training_duration_sec": round(train_duration, 2),
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "feature_names": FEATURE_COLUMNS,
        }

        logger.info("Test R2: %.4f | Test RMSE: %.4f | Test MAE: %.4f",
                     metrics["test_r2"], metrics["test_rmse"], metrics["test_mae"])

        return metrics

    def save(self, metrics: dict[str, Any], stats: DatasetStats) -> Path:
        artifact = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": FEATURE_COLUMNS,
        }

        model_path = self.model_dir / "housing_model.joblib"
        joblib.dump(artifact, model_path)
        logger.info("Model saved to %s", model_path)

        metadata = {
            "metrics": metrics,
            "stats": stats.to_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        meta_path = self.model_dir / "model_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Metadata saved to %s", meta_path)

        return model_path
