from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.ingestion.data_loader import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class HousingFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group ($10k)", ge=0, le=15)
    HouseAge: float = Field(..., description="Median house age in block group", ge=1, le=100)
    AveRooms: float = Field(..., description="Average number of rooms", ge=0.5, le=50)
    AveBedrms: float = Field(..., description="Average number of bedrooms", ge=0.2, le=10)
    Population: float = Field(..., description="Block group population", ge=1, le=50000)
    AveOccup: float = Field(..., description="Average household occupancy", ge=0.5, le=50)
    Latitude: float = Field(..., description="Latitude", ge=32.5, le=42.0)
    Longitude: float = Field(..., description="Longitude", ge=-124.5, le=-114.0)


class PredictionRequest(BaseModel):
    features: HousingFeatures


class PredictionResponse(BaseModel):
    predicted_value: float
    confidence: str
    features_used: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
