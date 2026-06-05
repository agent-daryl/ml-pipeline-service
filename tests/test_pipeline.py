#!/usr/bin/env python3
"""Unit tests for the ML pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.ingestion.data_loader import (
    load_california_housing, compute_stats, prepare_features, FEATURE_COLUMNS,
)
from src.monitoring.validator import DataValidator


class TestDataIngestion:
    def test_load_dataset(self):
        train_df, test_df = load_california_housing()
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert len(train_df) > len(test_df)
        for col in FEATURE_COLUMNS:
            assert col in train_df.columns
        assert "MedHouseVal" in train_df.columns

    def test_no_missing_values(self):
        train_df, test_df = load_california_housing()
        assert train_df.isnull().sum().sum() == 0
        assert test_df.isnull().sum().sum() == 0

    def test_compute_stats(self):
        train_df, test_df = load_california_housing()
        stats = compute_stats(train_df, test_df)
        assert stats.train_samples == len(train_df)
        assert stats.test_samples == len(test_df)
        for col in FEATURE_COLUMNS:
            assert col in stats.feature_means
            assert stats.feature_mins[col] <= stats.feature_means[col] <= stats.feature_maxs[col]

    def test_prepare_features_shape(self):
        train_df, test_df = load_california_housing()
        X_train, y_train, X_test, y_test = prepare_features(train_df, test_df)
        assert X_train.shape == (len(train_df), len(FEATURE_COLUMNS))
        assert y_train.shape == (len(train_df),)
        assert X_train.dtype == np.float64


class TestDataValidator:
    @pytest.fixture
    def validator(self):
        train_df, test_df = load_california_housing()
        stats = compute_stats(train_df, test_df)
        return DataValidator(stats.to_dict())

    def test_valid_input(self, validator):
        features = {
            "MedInc": 3.5, "HouseAge": 25, "AveRooms": 5, "AveBedrms": 1,
            "Population": 1200, "AveOccup": 3, "Latitude": 37.5, "Longitude": -122.0,
        }
        report = validator.validate_single(features)
        assert report["is_valid"] is True
        assert len(report["warnings"]) == 0

    def test_out_of_range_detection(self, validator):
        features = {
            "MedInc": 99.0, "HouseAge": 25, "AveRooms": 5, "AveBedrms": 1,
            "Population": 1200, "AveOccup": 3, "Latitude": 37.5, "Longitude": -122.0,
        }
        report = validator.validate_single(features)
        assert report["is_valid"] is False
        assert any("MedInc" in w for w in report["warnings"])

    def test_batch_validation(self, validator):
        good = {
            "MedInc": 3.5, "HouseAge": 25, "AveRooms": 5, "AveBedrms": 1,
            "Population": 1200, "AveOccup": 3, "Latitude": 37.5, "Longitude": -122.0,
        }
        report = validator.validate_batch([good, good])
        assert report["is_valid"] is True
        assert report["samples_valid"] == 2


class TestModelTraining:
    def test_training_produces_model(self):
        from src.training.trainer import ModelTrainer
        import tempfile
        import joblib

        train_df, test_df = load_california_housing()
        X_train, y_train, X_test, y_test = prepare_features(train_df, test_df)

        trainer = ModelTrainer(model_dir=Path(tempfile.mkdtemp()))
        metrics = trainer.train(X_train, y_train, X_test, y_test)

        assert "test_r2" in metrics
        assert metrics["test_r2"] > 0.7
        assert metrics["test_rmse"] < 0.5

        stats_obj, _ = compute_stats(train_df, test_df), None
        train_df2, test_df2 = load_california_housing()
        stats = compute_stats(train_df2, test_df2)
        trainer.save(metrics, stats)

        model_path = trainer.model_dir / "housing_model.joblib"
        assert model_path.exists()
        artifact = joblib.load(model_path)
        assert "model" in artifact
        assert "scaler" in artifact

    def test_prediction_consistency(self):
        from src.training.trainer import ModelTrainer
        import tempfile

        train_df, test_df = load_california_housing()
        X_train, y_train, X_test, y_test = prepare_features(train_df, test_df)

        trainer = ModelTrainer(model_dir=Path(tempfile.mkdtemp()))
        trainer.train(X_train, y_train, X_test, y_test)

        sample = X_test[:1]
        sample_scaled = trainer.scaler.transform(sample)
        pred1 = trainer.model.predict(sample_scaled)
        pred2 = trainer.model.predict(sample_scaled)
        assert np.allclose(pred1, pred2)
