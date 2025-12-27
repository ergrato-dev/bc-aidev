"""
Proyecto: Clasificador CIFAR-10
===============================

Construir una CNN para clasificar imágenes CIFAR-10.

Objetivo: ≥70% accuracy en test set.

Instrucciones:
1. Implementa cada sección marcada con TODO
2. Entrena el modelo
3. Genera visualizaciones
4. Responde las preguntas de análisis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================
# CONFIGURACIÓN
# ============================================

# Hiperparámetros
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
WEIGHT_DECAY = 1e-4

# Clases CIFAR-10
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ============================================
# 1. DEFINICIÓN DEL MODELO
# ============================================


class CIFAR10CNN(nn.Module):
    """
    CNN para clasificación CIFAR-10.

    Arquitectura requerida:
    - Al menos 3 bloques convolucionales
    - BatchNorm después de cada Conv
    - MaxPooling para reducir dimensiones
    - Dropout para regularización
    - Clasificador con capa oculta
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # TODO: Implementar extractor de características
        # Sugerencia: 3 bloques de Conv -> BatchNorm -> ReLU -> MaxPool
        self.features = None  # TODO: nn.Sequential(...)

        # TODO: Implementar clasificador
        # Sugerencia: Flatten -> Linear -> ReLU -> Dropout -> Linear
        self.classifier = None  # TODO: nn.Sequential(...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implementar forward pass
        pass


# ============================================
# 2. DATA AUGMENTATION Y LOADERS
# ============================================


def get_data_loaders(batch_size: int = 128):
    """
    Prepara los DataLoaders con data augmentation.

    Returns:
        train_loader, test_loader
    """
    # TODO: Definir transformaciones de entrenamiento
    # Incluir: RandomHorizontalFlip, RandomCrop, ToTensor, Normalize
    train_transform = None  # TODO

    # TODO: Definir transformaciones de test
    # Solo: ToTensor, Normalize
    test_transform = None  # TODO

    # Estadísticas CIFAR-10 para normalización
    # mean = [0.4914, 0.4822, 0.4465]
    # std = [0.2470, 0.2435, 0.2616]

    # TODO: Crear datasets
    train_dataset = None  # TODO
    test_dataset = None  # TODO

    # TODO: Crear DataLoaders
    train_loader = None  # TODO
    test_loader = None  # TODO

    return train_loader, test_loader


# ============================================
# 3. FUNCIONES DE ENTRENAMIENTO
# ============================================


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Entrena una época.

    Returns:
        (loss promedio, accuracy)
    """
    # TODO: Implementar loop de entrenamiento
    # 1. model.train()
    # 2. Para cada batch:
    #    - Mover datos a device
    #    - Forward pass
    #    - Calcular loss
    #    - Backward pass
    #    - Actualizar pesos
    # 3. Retornar loss y accuracy promedios
    pass


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    """
    Evalúa el modelo.

    Returns:
        (loss promedio, accuracy)
    """
    # TODO: Implementar evaluación
    # 1. model.eval()
    # 2. torch.no_grad()
    # 3. Calcular loss y accuracy
    pass


# ============================================
# 4. LOOP DE ENTRENAMIENTO
# ============================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 15,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
) -> dict:
    """
    Entrena el modelo completo.

    Returns:
        Diccionario con historial de entrenamiento
    """
    # TODO: Configurar
    # - criterion (CrossEntropyLoss)
    # - optimizer (Adam con weight_decay)
    # - scheduler (StepLR o ReduceLROnPlateau)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    best_acc = 0.0

    # TODO: Loop de entrenamiento
    # Para cada época:
    # 1. Entrenar
    # 2. Evaluar
    # 3. Guardar mejor modelo
    # 4. Actualizar scheduler
    # 5. Imprimir métricas

    return history


# ============================================
# 5. VISUALIZACIONES
# ============================================


def plot_training_curves(history: dict, save_path: str = "training_curves.png"):
    """Grafica curvas de entrenamiento."""
    # TODO: Crear figura con 2 subplots
    # - Subplot 1: Loss (train y test)
    # - Subplot 2: Accuracy (train y test)
    # - Guardar en save_path
    pass


def plot_confusion_matrix(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_path: str = "confusion_matrix.png",
):
    """Genera matriz de confusión."""
    # TODO: Implementar
    # 1. Obtener todas las predicciones
    # 2. Crear matriz de confusión
    # 3. Visualizar con heatmap
    # 4. Guardar en save_path
    pass


def plot_sample_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_samples: int = 10,
    save_path: str = "sample_predictions.png",
):
    """Muestra ejemplos de predicciones."""
    # TODO: Implementar
    # 1. Obtener algunas imágenes de test
    # 2. Predecir clases
    # 3. Mostrar imagen + predicción + label real
    # 4. Colorear verde si correcto, rojo si incorrecto
    pass


# ============================================
# 6. MAIN
# ============================================


def main():
    """Función principal."""
    print("=" * 60)
    print("CLASIFICADOR CIFAR-10")
    print("=" * 60)

    # TODO: 1. Preparar datos
    print("\n1. Cargando datos...")
    # train_loader, test_loader = get_data_loaders(BATCH_SIZE)

    # TODO: 2. Crear modelo
    print("\n2. Creando modelo...")
    # model = CIFAR10CNN(num_classes=10).to(device)
    # print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    # TODO: 3. Entrenar
    print("\n3. Entrenando...")
    # history = train_model(model, train_loader, test_loader, NUM_EPOCHS)

    # TODO: 4. Evaluar
    print("\n4. Evaluación final...")
    # test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    # print(f"Test Accuracy: {test_acc:.2f}%")

    # TODO: 5. Visualizaciones
    print("\n5. Generando visualizaciones...")
    # plot_training_curves(history)
    # plot_confusion_matrix(model, test_loader, device)
    # plot_sample_predictions(model, test_loader, device)

    # TODO: 6. Guardar modelo
    print("\n6. Guardando modelo...")
    # torch.save(model.state_dict(), 'model_cifar10.pth')

    print("\n" + "=" * 60)
    print("¡PROYECTO COMPLETADO!")
    print("=" * 60)


if __name__ == "__main__":
    main()


# ============================================
# PREGUNTAS DE ANÁLISIS
# ============================================
"""
Responde las siguientes preguntas:

1. ¿Por qué CIFAR-10 es más difícil que MNIST?
   Respuesta: 
   
2. ¿Qué efecto tiene el data augmentation en el accuracy?
   Respuesta: 
   
3. ¿Qué clases confunde más el modelo? ¿Por qué crees que ocurre?
   Respuesta: 
"""
