"""
Schemas Pydantic para validación de datos.

Los modelos Pydantic nos permiten:
- Validar automáticamente los datos de entrada
- Serializar/deserializar JSON
- Generar documentación OpenAPI automáticamente
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# ============================================
# PASO 1: Definir IrisFeatures
# ============================================
# Descomenta y completa el modelo de entrada

# class IrisFeatures(BaseModel):
#     """
#     Features de entrada para predicción de Iris.
#
#     Cada campo tiene validaciones:
#     - gt=0: mayor que 0
#     - lt=10: menor que 10
#     - description: para documentación
#     - examples: para Swagger UI
#     """
#
#     sepal_length: float = Field(
#         ...,  # ... significa requerido
#         gt=0,
#         lt=10,
#         description="Longitud del sépalo en cm",
#         examples=[5.1]
#     )
#
#     # TODO: Agregar sepal_width con las mismas validaciones
#     # sepal_width: float = Field(...)
#
#     # TODO: Agregar petal_length
#     # petal_length: float = Field(...)
#
#     # TODO: Agregar petal_width
#     # petal_width: float = Field(...)
#
#     # Validador personalizado (opcional)
#     @field_validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
#     @classmethod
#     def check_positive(cls, v: float) -> float:
#         """Asegurar que los valores sean positivos."""
#         if v <= 0:
#             raise ValueError('El valor debe ser positivo')
#         return round(v, 2)


# ============================================
# PASO 2: Definir PredictionResponse
# ============================================
# Descomenta y completa el modelo de respuesta

# class PredictionResponse(BaseModel):
#     """
#     Respuesta de predicción.
#
#     Incluye la predicción, confianza y metadatos.
#     """
#
#     prediction: str = Field(
#         ...,
#         description="Especie predicha (setosa, versicolor, virginica)"
#     )
#
#     confidence: float = Field(
#         ...,
#         ge=0,
#         le=1,
#         description="Confianza de la predicción (0-1)"
#     )
#
#     # TODO: Agregar campo probabilities (dict[str, float])
#     # probabilities: dict[str, float] = Field(...)
#
#     # TODO: Agregar campo model_version (str con default "1.0.0")
#     # model_version: str = Field(...)
#
#     # TODO: Agregar campo timestamp (datetime)
#     # timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================
# PASO 3: Definir HealthResponse
# ============================================
# Descomenta el modelo de health check

# class HealthResponse(BaseModel):
#     """Respuesta del health check."""
#
#     status: str = Field(default="ok")
#     model_loaded: bool = Field(default=True)
#     version: str = Field(default="1.0.0")


# ============================================
# PASO 4: Definir ErrorResponse (opcional)
# ============================================
# Modelo para respuestas de error estandarizadas

# class ErrorResponse(BaseModel):
#     """Respuesta de error."""
#
#     error: str
#     message: str
#     detail: Optional[str] = None
