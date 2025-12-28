"""
Modelo ML Simulado.
"""

import random
from typing import Tuple


class SimpleIrisModel:
    """Modelo simulado de clasificación de Iris."""

    def __init__(self):
        self.version = "1.0.0"
        self.classes = ["setosa", "versicolor", "virginica"]
        self.is_loaded = True

    def predict(self, features: list[float]) -> Tuple[str, float, dict]:
        """Realizar predicción simulada."""
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


model = SimpleIrisModel()
