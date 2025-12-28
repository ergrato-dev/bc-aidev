"""
Servicio del Modelo ML.

Encapsula la lógica de carga y predicción del modelo.
"""

import logging
import random
from typing import Tuple

# TODO: Descomentar para usar modelos reales
# import joblib
# from pathlib import Path

logger = logging.getLogger(__name__)


class MLModel:
    """
    Wrapper para el modelo de Machine Learning.

    TODO: Implementar carga de modelo real
    """

    def __init__(self, model_path: str = "ml_models/model.pkl"):
        self.model = None
        self.model_path = model_path
        self.version = "1.0.0"
        self.classes = ["setosa", "versicolor", "virginica"]

    def load(self) -> None:
        """
        Cargar modelo desde disco.

        TODO: Implementar carga real con joblib
        """
        logger.info(f"Cargando modelo desde {self.model_path}")

        # TODO: Cargar modelo real
        # path = Path(self.model_path)
        # if not path.exists():
        #     raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        # self.model = joblib.load(path)

        # Modelo simulado
        self.model = "simulated_model"
        logger.info("Modelo cargado (simulado)")

    def predict(self, features: list[float]) -> Tuple[str, float, dict]:
        """
        Realizar predicción.

        Args:
            features: [sepal_length, sepal_width, petal_length, petal_width]

        Returns:
            (predicción, confianza, probabilidades)

        TODO: Implementar predicción real
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado")

        # TODO: Usar modelo real
        # X = np.array(features).reshape(1, -1)
        # prediction_idx = self.model.predict(X)[0]
        # probabilities = self.model.predict_proba(X)[0]

        # Predicción simulada basada en reglas
        sepal_length, sepal_width, petal_length, petal_width = features

        if petal_length < 2.5:
            prediction = "setosa"
            base_confidence = 0.92
        elif petal_length < 5.0:
            prediction = "versicolor"
            base_confidence = 0.85
        else:
            prediction = "virginica"
            base_confidence = 0.88

        confidence = min(0.99, base_confidence + random.uniform(-0.05, 0.05))

        remaining = 1.0 - confidence
        probabilities = {}
        for cls in self.classes:
            if cls == prediction:
                probabilities[cls] = round(confidence, 4)
            else:
                probabilities[cls] = round(remaining / 2, 4)

        return prediction, confidence, probabilities

    def is_healthy(self) -> bool:
        """Verificar si el modelo está saludable."""
        return self.model is not None


# Singleton del modelo
ml_model = MLModel()
