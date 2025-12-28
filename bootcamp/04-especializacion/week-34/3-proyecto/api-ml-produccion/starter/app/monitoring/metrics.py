"""
Métricas de Prometheus.

Define y expone métricas para monitoreo de la API y el modelo.
"""

from fastapi import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ============================================
# Métricas de Infraestructura
# ============================================

# TODO: Definir métricas de requests

# REQUEST_COUNT = Counter(
#     'ml_api_requests_total',
#     'Total HTTP requests',
#     ['method', 'endpoint', 'status_code']
# )

# REQUEST_LATENCY = Histogram(
#     'ml_api_request_duration_seconds',
#     'Request duration in seconds',
#     ['endpoint'],
#     buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
# )

# ACTIVE_REQUESTS = Gauge(
#     'ml_api_active_requests',
#     'Currently processing requests'
# )


# ============================================
# Métricas de Modelo ML
# ============================================

# TODO: Definir métricas del modelo

# PREDICTIONS_BY_CLASS = Counter(
#     'ml_model_predictions_total',
#     'Predictions by class',
#     ['model_version', 'predicted_class']
# )

# PREDICTION_CONFIDENCE = Histogram(
#     'ml_model_prediction_confidence',
#     'Prediction confidence distribution',
#     ['model_version'],
#     buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
# )

# INFERENCE_TIME = Histogram(
#     'ml_model_inference_seconds',
#     'Model inference time',
#     ['model_version'],
#     buckets=[0.001, 0.005, 0.01, 0.05, 0.1]
# )

# MODEL_LOADED = Gauge(
#     'ml_model_loaded',
#     'Model loaded status (1=loaded, 0=not loaded)',
#     ['model_version']
# )


# ============================================
# Funciones Helper
# ============================================


def get_metrics():
    """Generar métricas en formato Prometheus."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def record_request(method: str, endpoint: str, status_code: int, duration: float):
    """
    Registrar métricas de un request.

    TODO: Implementar registro de métricas
    """
    # REQUEST_COUNT.labels(
    #     method=method,
    #     endpoint=endpoint,
    #     status_code=str(status_code)
    # ).inc()
    #
    # REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    pass


def record_prediction(
    model_version: str, predicted_class: str, confidence: float, inference_time: float
):
    """
    Registrar métricas de una predicción.

    TODO: Implementar registro de métricas
    """
    # PREDICTIONS_BY_CLASS.labels(
    #     model_version=model_version,
    #     predicted_class=predicted_class
    # ).inc()
    #
    # PREDICTION_CONFIDENCE.labels(model_version=model_version).observe(confidence)
    # INFERENCE_TIME.labels(model_version=model_version).observe(inference_time)
    pass
