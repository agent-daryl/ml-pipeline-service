from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
TARGET_COLUMN = "MedHouseVal"


@dataclass
class DatasetStats:
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    feature_mins: dict[str, float]
    feature_maxs: dict[str, float]
    target_mean: float
    target_std: float
    target_min: float
    target_max: float
    train_samples: int
    test_samples: int

    def to_dict(self) -> dict:
        return {
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "feature_mins": self.feature_mins,
            "feature_maxs": self.feature_maxs,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
        }


def load_california_housing(test_size: float = 0.2, random_state: int = 42):
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()

    missing_mask = df.isnull().any(axis=1)
    n_missing = missing_mask.sum()
    if n_missing > 0:
        logger.warning("Dropping %d rows with missing values", n_missing)
        df = df.dropna()

    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(df) * (1.0 - test_size))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    return train_df, test_df


def compute_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> DatasetStats:
    feat_means = train_df[FEATURE_COLUMNS].mean().to_dict()
    feat_stds = train_df[FEATURE_COLUMNS].std().to_dict()
    feat_mins = train_df[FEATURE_COLUMNS].min().to_dict()
    feat_maxs = train_df[FEATURE_COLUMNS].max().to_dict()

    stats = DatasetStats(
        feature_means={k: float(v) for k, v in feat_means.items()},
        feature_stds={k: float(v) for k, v in feat_stds.items()},
        feature_mins={k: float(v) for k, v in feat_mins.items()},
        feature_maxs={k: float(v) for k, v in feat_maxs.items()},
        target_mean=float(train_df[TARGET_COLUMN].mean()),
        target_std=float(train_df[TARGET_COLUMN].std()),
        target_min=float(train_df[TARGET_COLUMN].min()),
        target_max=float(train_df[TARGET_COLUMN].max()),
        train_samples=len(train_df),
        test_samples=len(test_df),
    )

    stats_path = DATA_DIR / "dataset_stats.json"
    import json
    with open(stats_path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)
    logger.info("Saved dataset stats to %s", stats_path)

    return stats


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df[FEATURE_COLUMNS].values.astype(np.float64)
    y_train = train_df[TARGET_COLUMN].values.astype(np.float64)
    X_test = test_df[FEATURE_COLUMNS].values.astype(np.float64)
    y_test = test_df[TARGET_COLUMN].values.astype(np.float64)

    logger.info(
        "Features ready — train: %d samples, test: %d samples, %d features",
        len(X_train), len(X_test), X_train.shape[1],
    )
    return X_train, y_train, X_test, y_test
