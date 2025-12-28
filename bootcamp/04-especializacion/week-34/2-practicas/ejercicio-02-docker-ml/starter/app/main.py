"""
API FastAPI para ML - Lista para Docker.
"""

import os
from datetime import datetime

from app.model import model
from app.schemas import HealthResponse, IrisFeatures, PredictionResponse
from fastapi import FastAPI, HTTPException, status

# Leer configuración de variables de entorno
APP_NAME = os.getenv("APP_NAME", "ML API")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="API de clasificación de Iris containerizada con Docker",
)


@app.get("/")
def root():
    """Endpoint raíz."""
    return {"name": APP_NAME, "version": APP_VERSION, "status": "running"}


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check para Docker y load balancers."""
    return HealthResponse(
        status="ok", model_loaded=model.is_loaded, version=model.version
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    """Realizar predicción."""
    try:
        feature_list = [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]

        prediction, confidence, probabilities = model.predict(feature_list)

        return PredictionResponse(
            prediction=prediction,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            model_version=model.version,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
