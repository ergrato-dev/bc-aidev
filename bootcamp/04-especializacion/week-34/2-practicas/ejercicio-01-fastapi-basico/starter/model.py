"""
Modelo ML Simulado para el ejercicio.

En producción, aquí cargaríamos un modelo real con joblib o pickle.
Para este ejercicio, simulamos las predicciones con reglas simples.
"""

import random
from typing import Tuple


class SimpleIrisModel:
    """
    Modelo simulado de clasificación de Iris.

    Simula predicciones basándose en reglas simples
    derivadas de las características de las flores Iris.
    """

    def __init__(self):
        self.version = "1.0.0"
        self.classes = ["setosa", "versicolor", "virginica"]
        self.is_loaded = True

    def predict(self, features: list[float]) -> Tuple[str, float, dict]:
        """
        Realizar predicción simulada.

        Args:
            features: Lista [sepal_length, sepal_width, petal_length, petal_width]

        Returns:
            Tuple con (clase_predicha, confianza, probabilidades)
        """
        sepal_length, sepal_width, petal_length, petal_width = features

        # Reglas simples basadas en características reales del dataset Iris
        # Setosa: pétalos pequeños (< 2.5 cm)
        # Versicolor: pétalos medianos (2.5-5 cm)
        # Virginica: pétalos grandes (> 5 cm)

        if petal_length < 2.5:
            prediction = "setosa"
            base_confidence = 0.92
        elif petal_length < 5.0:
            prediction = "versicolor"
            base_confidence = 0.85
        else:
            prediction = "virginica"
            base_confidence = 0.88

        # Agregar algo de variación realista
        confidence = min(0.99, base_confidence + random.uniform(-0.05, 0.05))

        # Generar probabilidades
        remaining = 1.0 - confidence
        probabilities = {}
        for cls in self.classes:
            if cls == prediction:
                probabilities[cls] = round(confidence, 4)
            else:
                # Distribuir el resto entre las otras clases
                probabilities[cls] = round(remaining / 2, 4)

        return prediction, confidence, probabilities


# Singleton del modelo
model = SimpleIrisModel()
