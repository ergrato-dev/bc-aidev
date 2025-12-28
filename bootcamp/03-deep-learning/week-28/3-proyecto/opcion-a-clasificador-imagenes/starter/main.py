"""
Clasificador de Imágenes con Transfer Learning
==============================================

Proyecto final de Deep Learning - Opción A: Computer Vision

Objetivo: Clasificar imágenes de CIFAR-10 con accuracy > 85%
usando transfer learning con ResNet.

Ejecutar:
    python main.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# ============================================
# CONFIGURACIÓN
# ============================================

CONFIG = {
    "data": {
        "data_dir": "./data",
        "batch_size": 64,
        "num_workers": 2,
        "val_split": 0.1,
        "test_split": 0.1,
    },
    "model": {
        "architecture": "resnet18",
        "num_classes": 10,
        "pretrained": True,
        "dropout": 0.5,
    },
    "training": {
        "epochs": 20,
        "lr": 0.001,
        "weight_decay": 0.01,
        "patience": 5,  # Early stopping
    },
}

# Nombres de las clases CIFAR-10
CLASS_NAMES = [
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


# ============================================
# TODO 1: TRANSFORMACIONES DE DATOS
# ============================================


def get_train_transforms():
    """
    Crea transformaciones para entrenamiento con data augmentation.

    Returns:
        transforms.Compose con las transformaciones

    Hints:
        - RandomCrop con padding para variabilidad
        - RandomHorizontalFlip para simetría
        - Normalizar con media/std de CIFAR-10
        - Media: [0.4914, 0.4822, 0.4465]
        - Std: [0.2470, 0.2435, 0.2616]
    """
    # TODO: Implementar transformaciones de entrenamiento
    # return transforms.Compose([...])
    pass


def get_val_transforms():
    """
    Crea transformaciones para validación/test (sin augmentation).

    Returns:
        transforms.Compose con las transformaciones
    """
    # TODO: Implementar transformaciones de validación
    # return transforms.Compose([...])
    pass


# ============================================
# TODO 2: DATALOADERS
# ============================================


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoaders para train, validation y test.

    Args:
        config: Diccionario de configuración

    Returns:
        Tuple de (train_loader, val_loader, test_loader)

    Hints:
        - Usar datasets.CIFAR10 con download=True
        - Dividir train en train/val usando random_split
        - Test set es separado (train=False)
    """
    # TODO: Implementar carga de datos
    # 1. Cargar dataset de entrenamiento completo
    # 2. Dividir en train/val
    # 3. Cargar test set
    # 4. Crear DataLoaders
    pass


# ============================================
# TODO 3: MODELO CON TRANSFER LEARNING
# ============================================


def create_model(config: dict) -> nn.Module:
    """
    Crea modelo ResNet adaptado para CIFAR-10.

    Args:
        config: Configuración del modelo

    Returns:
        Modelo PyTorch

    Hints:
        - Cargar ResNet18 preentrenado
        - Modificar conv1 para imágenes 32x32 (kernel=3, stride=1, padding=1)
        - Remover maxpool (usar nn.Identity())
        - Reemplazar fc con nueva capa para 10 clases
    """
    # TODO: Implementar creación del modelo
    # 1. Cargar modelo preentrenado
    # 2. Adaptar para CIFAR-10
    # 3. Modificar clasificador
    pass


# ============================================
# TODO 4: LOOP DE ENTRENAMIENTO
# ============================================


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
) -> tuple[float, float]:
    """
    Entrena el modelo por una época.

    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        criterion: Función de pérdida
        optimizer: Optimizador
        device: Dispositivo (cuda/cpu)
        scheduler: Learning rate scheduler (opcional)

    Returns:
        Tuple de (loss promedio, accuracy)
    """
    # TODO: Implementar entrenamiento de una época
    # 1. model.train()
    # 2. Iterar sobre batches
    # 3. Forward, loss, backward, step
    # 4. Calcular métricas
    pass


def validate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    """
    Evalúa el modelo en el set de validación.

    Args:
        model: Modelo a evaluar
        val_loader: DataLoader de validación
        criterion: Función de pérdida
        device: Dispositivo

    Returns:
        Tuple de (loss promedio, accuracy)
    """
    # TODO: Implementar validación
    # 1. model.eval()
    # 2. torch.no_grad()
    # 3. Calcular métricas
    pass


# ============================================
# TODO 5: ENTRENAMIENTO COMPLETO
# ============================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> dict:
    """
    Entrena el modelo completo con early stopping.

    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        config: Configuración de entrenamiento
        device: Dispositivo

    Returns:
        Diccionario con historial de entrenamiento
    """
    # TODO: Implementar entrenamiento completo
    # 1. Configurar criterion, optimizer, scheduler
    # 2. Loop de épocas
    # 3. Early stopping
    # 4. Guardar mejor modelo
    # 5. Retornar historial
    pass


# ============================================
# TODO 6: EVALUACIÓN FINAL
# ============================================


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> dict:
    """
    Evalúa el modelo en el test set y genera métricas.

    Args:
        model: Modelo entrenado
        test_loader: DataLoader de test
        device: Dispositivo

    Returns:
        Diccionario con métricas

    Hints:
        - Calcular accuracy, precision, recall, F1
        - Generar matriz de confusión
        - Usar sklearn.metrics
    """
    # TODO: Implementar evaluación completa
    # 1. Obtener predicciones
    # 2. Calcular métricas
    # 3. Generar matriz de confusión
    pass


def plot_training_history(history: dict, save_path: str = "training_history.png"):
    """
    Genera gráficas del historial de entrenamiento.

    Args:
        history: Diccionario con métricas por época
        save_path: Ruta para guardar la imagen
    """
    # TODO: Implementar visualización
    # 1. Gráfica de loss (train vs val)
    # 2. Gráfica de accuracy (train vs val)
    pass


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: list,
    save_path: str = "confusion_matrix.png",
):
    """
    Genera y guarda la matriz de confusión.

    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones
        class_names: Nombres de las clases
        save_path: Ruta para guardar
    """
    # TODO: Implementar matriz de confusión
    # Usar sklearn.metrics.confusion_matrix
    # Visualizar con seaborn o matplotlib
    pass


# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================


def main():
    """Función principal del proyecto."""
    print("=" * 60)
    print("CLASIFICADOR DE IMÁGENES - CIFAR-10")
    print("Transfer Learning con ResNet")
    print("=" * 60)

    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")

    # TODO: Descomentar cuando las funciones estén implementadas

    # # 1. Cargar datos
    # print("\n--- Cargando datos ---")
    # train_loader, val_loader, test_loader = get_dataloaders(CONFIG['data'])
    #
    # # 2. Crear modelo
    # print("\n--- Creando modelo ---")
    # model = create_model(CONFIG['model']).to(device)
    #
    # # 3. Entrenar
    # print("\n--- Entrenando ---")
    # history = train_model(model, train_loader, val_loader, CONFIG['training'], device)
    #
    # # 4. Evaluar
    # print("\n--- Evaluando en test set ---")
    # model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
    # metrics = evaluate_model(model, test_loader, device)
    #
    # # 5. Visualizar
    # print("\n--- Generando visualizaciones ---")
    # plot_training_history(history)
    #
    # # 6. Resumen
    # print("\n" + "=" * 60)
    # print("RESULTADOS FINALES")
    # print("=" * 60)
    # print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
    # print(f"Test F1-Score: {metrics['f1']:.4f}")
    # print("Modelo guardado en: best_model.pth")
    # print("Gráficas guardadas en: training_history.png, confusion_matrix.png")

    print("\nPara ejecutar el proyecto:")
    print("1. Implementa todos los TODOs")
    print("2. Descomenta el código en main()")
    print("3. Ejecuta: python main.py")


if __name__ == "__main__":
    main()
