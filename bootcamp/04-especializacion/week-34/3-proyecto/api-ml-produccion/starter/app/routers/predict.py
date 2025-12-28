"""
Router de Predicciones.

Endpoints para realizar predicciones con el modelo ML.
"""

import logging
import time

from app.models.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    IrisFeatures,
    PredictionResponse,
)
from fastapi import APIRouter, HTTPException, status

# TODO: Importar servicio del modelo y métricas
# from app.services.ml_model import ml_model
# from app.monitoring.metrics import record_prediction, ACTIVE_REQUESTS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["predictions"])


# ============================================
# Predicción Individual
# ============================================


@router.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """
    Realizar predicción de especie de Iris.

    TODO: Implementar predicción real con el modelo
    """
    # TODO: Incrementar contador de requests activos
    # ACTIVE_REQUESTS.inc()

    start_time = time.time()

    try:
        # TODO: Convertir features a lista
        # feature_list = [
        #     features.sepal_length,
        #     features.sepal_width,
        #     features.petal_length,
        #     features.petal_width
        # ]

        # TODO: Realizar predicción
        # prediction, confidence, probabilities = ml_model.predict(feature_list)

        # Predicción simulada (remover cuando se implemente el modelo)
        prediction = "setosa"
        confidence = 0.95
        probabilities = {"setosa": 0.95, "versicolor": 0.03, "virginica": 0.02}

        inference_time = time.time() - start_time

        # TODO: Registrar métricas
        # record_prediction(
        #     model_version=ml_model.version,
        #     predicted_class=prediction,
        #     confidence=confidence,
        #     inference_time=inference_time
        # )

        logger.info(f"Predicción: {prediction} (confianza: {confidence:.2f})")

        return PredictionResponse(
            prediction=prediction,
            confidence=round(confidence, 4),
            # probabilities=probabilities,  # TODO: Descomentar
            # model_version=ml_model.version  # TODO: Descomentar
        )

    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando predicción: {str(e)}",
        )

    # finally:
    #     ACTIVE_REQUESTS.dec()


# ============================================
# Predicción en Lote
# ============================================


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Realizar predicciones en lote.

    TODO: Implementar predicción en lote
    """
    start_time = time.time()

    predictions = []

    for instance in request.instances:
        # TODO: Usar el modelo real
        # feature_list = [...]
        # pred, conf, probs = ml_model.predict(feature_list)

        # Predicción simulada
        predictions.append(PredictionResponse(prediction="setosa", confidence=0.95))

    processing_time = (time.time() - start_time) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        total_instances=len(predictions),
        processing_time_ms=round(processing_time, 2),
    )
