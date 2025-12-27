"""
Proyecto: Predictor de Temperatura con LSTM
============================================
Objetivo: MAE < 2°C en test set

Implementa las funciones marcadas con TODO.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Reproducibilidad
torch.manual_seed(42)
np.random.seed(42)


# ============================================
# FUNCIÓN: generate_temperature_data
# Generar datos sintéticos de temperatura
# ============================================


def generate_temperature_data(days: int) -> np.ndarray:
    """
    Genera datos sintéticos de temperatura.

    Patrón: base + estacional (seno anual) + ruido

    Args:
        days: Número de días a generar

    Returns:
        Array con temperaturas diarias (en °C)
    """
    # TODO: Implementar generación de datos
    # Hint: temp = base + amplitude * sin(2*pi*t/365) + noise
    pass


# ============================================
# FUNCIÓN: create_sequences
# Crear ventanas deslizantes para entrenamiento
# ============================================


def create_sequences(data: np.ndarray, seq_len: int) -> tuple:
    """
    Crea secuencias X, y para entrenamiento supervisado.

    Args:
        data: Serie temporal normalizada
        seq_len: Longitud de la ventana de entrada

    Returns:
        X: (n_samples, seq_len)
        y: (n_samples,)
    """
    # TODO: Implementar ventanas deslizantes
    pass


# ============================================
# CLASE: TemperatureLSTM
# Modelo LSTM para predicción
# ============================================


class TemperatureLSTM(nn.Module):
    """
    LSTM para predicción de temperatura.

    Arquitectura:
    - LSTM con múltiples capas
    - Dropout entre capas
    - Capa fully connected para output
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        # TODO: Implementar arquitectura
        pass

    def forward(self, x):
        # TODO: Implementar forward pass
        pass


# ============================================
# FUNCIÓN: train_model
# Entrenar el modelo
# ============================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.001,
) -> list:
    """
    Entrena el modelo LSTM.

    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        epochs: Número de épocas
        lr: Learning rate

    Returns:
        Lista con historial de pérdidas
    """
    # TODO: Implementar loop de entrenamiento
    pass


# ============================================
# FUNCIÓN: evaluate_model
# Evaluar el modelo
# ============================================


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, data_min: float, data_max: float
) -> float:
    """
    Evalúa el modelo y retorna MAE en escala original.

    Args:
        model: Modelo entrenado
        test_loader: DataLoader de test
        data_min: Mínimo para desnormalizar
        data_max: Máximo para desnormalizar

    Returns:
        MAE en grados Celsius
    """
    # TODO: Implementar evaluación
    pass


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=== Predictor de Temperatura con LSTM ===\n")

    # Configuración
    DAYS = 365 * 3  # 3 años
    SEQ_LEN = 30  # 30 días de ventana
    BATCH_SIZE = 32
    EPOCHS = 100

    # 1. Generar datos
    print("1. Generando datos...")
    # TODO: Llamar generate_temperature_data

    # 2. Normalizar
    print("2. Normalizando...")
    # TODO: Normalizar datos (guardar min/max)

    # 3. Crear secuencias
    print("3. Creando secuencias...")
    # TODO: Llamar create_sequences

    # 4. Split train/val/test
    print("4. Dividiendo datos...")
    # TODO: 70% train, 15% val, 15% test

    # 5. Crear DataLoaders
    print("5. Creando DataLoaders...")
    # TODO: Crear DataLoaders

    # 6. Crear modelo
    print("6. Creando modelo...")
    # TODO: Instanciar TemperatureLSTM

    # 7. Entrenar
    print("7. Entrenando...")
    # TODO: Llamar train_model

    # 8. Evaluar
    print("8. Evaluando...")
    # TODO: Llamar evaluate_model

    # 9. Resultado
    print("\n=== Resultado ===")
    # TODO: Mostrar MAE y verificar < 2°C
