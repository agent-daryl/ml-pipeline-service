#!/usr/bin/env python3
"""Train the housing price prediction model."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.data_loader import load_california_housing, compute_stats, prepare_features
from src.training.trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

logger.info("Loading California Housing dataset...")
train_df, test_df = load_california_housing()
logger.info("Dataset loaded: %d train, %d test samples", len(train_df), len(test_df))

logger.info("Computing dataset statistics...")
stats = compute_stats(train_df, test_df)

logger.info("Preparing features...")
X_train, y_train, X_test, y_test = prepare_features(train_df, test_df)

logger.info("Initializing trainer...")
trainer = ModelTrainer()

logger.info("Starting training...")
metrics = trainer.train(X_train, y_train, X_test, y_test)

logger.info("Saving model artifacts...")
model_path = trainer.save(metrics, stats)

logger.info("Training complete!")
logger.info("  Model: %s", model_path)
logger.info("  Test R2: %.4f", metrics["test_r2"])
logger.info("  Test RMSE: %.4f", metrics["test_rmse"])
logger.info("  Test MAE: %.4f", metrics["test_mae"])
logger.info("  Duration: %.2fs", metrics["training_duration_sec"])
