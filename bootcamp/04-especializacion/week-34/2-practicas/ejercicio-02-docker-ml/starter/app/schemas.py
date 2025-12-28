"""
Schemas Pydantic para la API.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class IrisFeatures(BaseModel):
    """Features de entrada para predicción."""

    sepal_length: float = Field(..., gt=0, lt=10)
    sepal_width: float = Field(..., gt=0, lt=10)
    petal_length: float = Field(..., gt=0, lt=10)
    petal_width: float = Field(..., gt=0, lt=10)


class PredictionResponse(BaseModel):
    """Respuesta de predicción."""

    prediction: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Respuesta de health check."""

    status: str = "ok"
    model_loaded: bool = True
    version: str = "1.0.0"
