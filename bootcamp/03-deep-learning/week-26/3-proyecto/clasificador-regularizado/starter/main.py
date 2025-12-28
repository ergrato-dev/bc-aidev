"""
Proyecto: Clasificador Regularizado
===================================
Combina Dropout, BatchNorm y Data Augmentation para maximizar generalización.

Objetivos:
- Test accuracy > 85%
- Gap train-test < 5%
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================
# TODO 1: Definir Data Augmentation Pipeline
# ============================================
# Crear transform con:
# - RandomHorizontalFlip
# - RandomRotation (10-15 grados)
# - RandomCrop con padding
# - ColorJitter
# - ToTensor + Normalize

train_transform = None  # TODO: Implementar

test_transform = None  # TODO: Solo ToTensor + Normalize


# ============================================
# TODO 2: Cargar Datasets
# ============================================
# train_dataset = ...
# test_dataset = ...
# train_loader = ...
# test_loader = ...


# ============================================
# TODO 3: Definir Modelo Regularizado
# ============================================
class RegularizedCNN(nn.Module):
    """
    CNN con todas las técnicas de regularización.

    Arquitectura:
    - 3 bloques conv: Conv → BatchNorm → ReLU → MaxPool
    - 2 capas FC con Dropout
    """

    def __init__(self):
        super().__init__()
        # TODO: Implementar capas
        # Conv blocks con BatchNorm2d
        # FC layers con BatchNorm1d y Dropout
        pass

    def forward(self, x):
        # TODO: Implementar forward pass
        pass


# ============================================
# TODO 4: Funciones de Entrenamiento
# ============================================
def train_epoch(model, loader, criterion, optimizer):
    """Entrena una época."""
    # TODO: Implementar
    # No olvidar model.train()
    pass


def evaluate(model, loader, criterion):
    """Evalúa el modelo."""
    # TODO: Implementar
    # No olvidar model.eval()
    pass


# ============================================
# TODO 5: Early Stopping
# ============================================
class EarlyStopping:
    """Para entrenamiento si no mejora."""

    def __init__(self, patience=7, min_delta=0.001):
        # TODO: Implementar
        pass

    def __call__(self, val_loss):
        # TODO: Retornar True si debe parar
        pass


# ============================================
# TODO 6: Loop de Entrenamiento
# ============================================
def train_model(model, train_loader, test_loader, epochs=50):
    """Entrena el modelo completo."""
    # TODO: Implementar con:
    # - Optimizer Adam
    # - CrossEntropyLoss
    # - Early Stopping
    # - Guardar métricas
    pass


# ============================================
# TODO 7: Visualizar y Analizar
# ============================================
def plot_results(train_accs, test_accs, train_losses, test_losses):
    """Grafica resultados del entrenamiento."""
    # TODO: Implementar gráficas de:
    # - Accuracy train vs test
    # - Loss train vs test
    # - Gap por época
    pass


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("Proyecto: Clasificador Regularizado")
    print("=" * 50)

    # TODO: Ejecutar entrenamiento y mostrar resultados

    print("\nImplementa los TODOs para completar el proyecto!")
