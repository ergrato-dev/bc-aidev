"""
API FastAPI con Métricas de Prometheus.
"""

import os
import time
from datetime import datetime

from app.model import model
from app.schemas import HealthResponse, IrisFeatures, PredictionResponse
from fastapi import FastAPI, HTTPException, status

# ============================================
# PASO 1: Importar Métricas
# ============================================
# Descomenta el import de métricas

# from app.metrics import (
#     get_metrics,
#     record_request,
#     record_prediction,
#     MODEL_LOADED,
#     ACTIVE_REQUESTS
# )


APP_NAME = os.getenv("APP_NAME", "ML API")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

app = FastAPI(
    title=APP_NAME, version=APP_VERSION, description="API ML con métricas Prometheus"
)


# ============================================
# PASO 2: Evento de Startup
# ============================================
# Descomenta para registrar que el modelo está cargado

# @app.on_event("startup")
# async def startup_event():
#     """Registrar métricas al iniciar."""
#     MODEL_LOADED.labels(model_version=model.version).set(1)


# ============================================
# PASO 3: Endpoint de Métricas
# ============================================
# Descomenta el endpoint

# @app.get("/metrics")
# async def metrics():
#     """Endpoint de métricas para Prometheus."""
#     return get_metrics()


@app.get("/")
def root():
    """Endpoint raíz."""
    return {"name": APP_NAME, "version": APP_VERSION}


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check."""
    return HealthResponse(
        status="ok", model_loaded=model.is_loaded, version=model.version
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    """
    Realizar predicción con métricas.
    """
    # ============================================
    # PASO 4: Instrumentar Predicción
    # ============================================
    # Descomenta para agregar métricas

    # ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        feature_list = [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]

        prediction, confidence, probabilities = model.predict(feature_list)

        inference_time = time.time() - start_time

        # ============================================
        # PASO 5: Registrar Métricas de Predicción
        # ============================================
        # Descomenta para registrar

        # record_prediction(
        #     model_version=model.version,
        #     predicted_class=prediction,
        #     confidence=confidence,
        #     inference_time=inference_time
        # )

        # record_request(
        #     method="POST",
        #     endpoint="/predict",
        #     status_code=200,
        #     duration=inference_time
        # )

        return PredictionResponse(
            prediction=prediction,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            model_version=model.version,
        )

    except Exception as e:
        # record_request("POST", "/predict", 500, time.time() - start_time)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    # finally:
    #     ACTIVE_REQUESTS.dec()
