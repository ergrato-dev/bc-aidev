"""
API FastAPI para servir modelo de clasificación Iris.

Este es el archivo principal de la aplicación.
Ejecutar con: uvicorn main:app --reload --port 8000
"""

from datetime import datetime

from fastapi import FastAPI, HTTPException, status

# Importar nuestros módulos
from model import model

# from schemas import IrisFeatures, PredictionResponse, HealthResponse


# ============================================
# PASO 1: Crear instancia de FastAPI
# ============================================
# Descomenta y personaliza la configuración

# app = FastAPI(
#     title="Iris Classifier API",
#     description="""
#     API para clasificación de flores Iris.
#
#     ## Endpoints
#     - `/health` - Verificar estado del servicio
#     - `/predict` - Realizar predicciones
#
#     ## Modelo
#     Clasificador entrenado con el dataset Iris.
#     """,
#     version="1.0.0"
# )


# ============================================
# PASO 2: Endpoint raíz
# ============================================
# Descomenta el endpoint básico

# @app.get("/")
# def root():
#     """Endpoint raíz - información básica de la API."""
#     return {
#         "name": "Iris Classifier API",
#         "version": "1.0.0",
#         "status": "running",
#         "docs": "/docs"
#     }


# ============================================
# PASO 3: Health Check
# ============================================
# Descomenta e implementa el health check

# @app.get("/health", response_model=HealthResponse)
# def health_check():
#     """
#     Health check endpoint.
#
#     Usado por load balancers y sistemas de monitoreo
#     para verificar que el servicio está activo.
#     """
#     return HealthResponse(
#         status="ok",
#         model_loaded=model.is_loaded,
#         version=model.version
#     )


# ============================================
# PASO 4: Endpoint de Predicción
# ============================================
# Descomenta e implementa la predicción

# @app.post("/predict", response_model=PredictionResponse)
# def predict(features: IrisFeatures):
#     """
#     Realizar predicción de especie de Iris.
#
#     Recibe las características de la flor y retorna
#     la especie predicha con su confianza.
#
#     - **sepal_length**: Longitud del sépalo (cm)
#     - **sepal_width**: Ancho del sépalo (cm)
#     - **petal_length**: Longitud del pétalo (cm)
#     - **petal_width**: Ancho del pétalo (cm)
#     """
#     try:
#         # Convertir features a lista
#         feature_list = [
#             features.sepal_length,
#             features.sepal_width,
#             features.petal_length,
#             features.petal_width
#         ]
#
#         # Realizar predicción
#         prediction, confidence, probabilities = model.predict(feature_list)
#
#         # Retornar respuesta
#         return PredictionResponse(
#             prediction=prediction,
#             confidence=round(confidence, 4),
#             probabilities=probabilities,
#             model_version=model.version,
#             timestamp=datetime.utcnow()
#         )
#
#     except Exception as e:
#         # Manejar errores
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error en predicción: {str(e)}"
#         )


# ============================================
# PASO 5: Ejecutar (opcional para desarrollo)
# ============================================
# Descomenta para poder ejecutar con: python main.py

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
