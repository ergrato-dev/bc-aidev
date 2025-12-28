"""
Métricas de Prometheus para la API ML.

Este módulo define las métricas que expondremos para monitoreo.
Prometheus las recolectará periódicamente del endpoint /metrics.
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
# PASO 1: Métricas de Infraestructura
# ============================================
# Descomenta las métricas de requests

# Total de requests (Counter - solo incrementa)
# REQUEST_COUNT = Counter(
#     'ml_api_requests_total',
#     'Total number of HTTP requests',
#     ['method', 'endpoint', 'status_code']
# )

# Latencia de requests (Histogram - distribución)
# REQUEST_LATENCY = Histogram(
#     'ml_api_request_duration_seconds',
#     'Request duration in seconds',
#     ['endpoint'],
#     buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
# )

# Requests activos (Gauge - sube y baja)
# ACTIVE_REQUESTS = Gauge(
#     'ml_api_active_requests',
#     'Number of requests currently being processed'
# )


# ============================================
# PASO 2: Métricas de Modelo ML
# ============================================
# Descomenta las métricas específicas de ML

# Predicciones por clase
# PREDICTIONS_BY_CLASS = Counter(
#     'ml_model_predictions_total',
#     'Total predictions by class',
#     ['model_version', 'predicted_class']
# )

# Confianza de predicciones
# PREDICTION_CONFIDENCE = Histogram(
#     'ml_model_prediction_confidence',
#     'Prediction confidence distribution',
#     ['model_version'],
#     buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
# )

# Tiempo de inferencia
# INFERENCE_TIME = Histogram(
#     'ml_model_inference_seconds',
#     'Model inference time',
#     ['model_version'],
#     buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
# )

# Estado del modelo
# MODEL_LOADED = Gauge(
#     'ml_model_loaded',
#     'Whether model is loaded (1) or not (0)',
#     ['model_version']
# )


# ============================================
# PASO 3: Función para Obtener Métricas
# ============================================
# Descomenta la función

# def get_metrics():
#     """
#     Generar métricas en formato Prometheus.
#
#     Returns:
#         Response con métricas en formato text/plain
#     """
#     return Response(
#         content=generate_latest(),
#         media_type=CONTENT_TYPE_LATEST
#     )


# ============================================
# PASO 4: Helpers para Registrar Métricas
# ============================================
# Descomenta los helpers

# def record_request(method: str, endpoint: str, status_code: int, duration: float):
#     """Registrar métricas de un request."""
#     REQUEST_COUNT.labels(
#         method=method,
#         endpoint=endpoint,
#         status_code=str(status_code)
#     ).inc()
#
#     REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)


# def record_prediction(model_version: str, predicted_class: str, confidence: float, inference_time: float):
#     """Registrar métricas de una predicción."""
#     PREDICTIONS_BY_CLASS.labels(
#         model_version=model_version,
#         predicted_class=predicted_class
#     ).inc()
#
#     PREDICTION_CONFIDENCE.labels(model_version=model_version).observe(confidence)
#     INFERENCE_TIME.labels(model_version=model_version).observe(inference_time)
