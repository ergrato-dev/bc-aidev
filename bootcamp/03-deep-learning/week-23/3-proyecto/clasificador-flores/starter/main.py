# ============================================
# PROYECTO: Clasificador de Flores con Transfer Learning
# ============================================
# Objetivo: Clasificar 102 tipos de flores con accuracy >= 85%
# Dataset: Flowers-102
# ============================================

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import Flowers102
from tqdm import tqdm

# ============================================
# CONFIGURACIÓN
# ============================================

CONFIG = {
    "batch_size": 32,
    "num_classes": 102,
    "image_size": 224,
    "epochs_feature_extraction": 10,
    "epochs_fine_tuning": 15,
    "lr_feature_extraction": 1e-3,
    "lr_fine_tuning": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir": "./data",
}

print(f"Dispositivo: {CONFIG['device']}")


# ============================================
# PARTE 1: CARGA DE DATOS
# ============================================


def get_transforms():
    """
    Define las transformaciones para train y val/test.

    Returns:
        train_transform: Transformaciones con augmentation
        val_transform: Transformaciones sin augmentation
    """
    # TODO: Definir transformaciones de entrenamiento
    # - RandomResizedCrop(224)
    # - RandomHorizontalFlip
    # - RandomRotation(15)
    # - ColorJitter
    # - ToTensor
    # - Normalize con medias de ImageNet
    train_transform = None

    # TODO: Definir transformaciones de validación/test
    # - Resize(256)
    # - CenterCrop(224)
    # - ToTensor
    # - Normalize con medias de ImageNet
    val_transform = None

    return train_transform, val_transform


def load_data(config):
    """
    Carga el dataset Flowers-102.

    Args:
        config: Diccionario de configuración

    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform, val_transform = get_transforms()

    # TODO: Cargar datasets
    # train_dataset = Flowers102(...)
    # val_dataset = Flowers102(...)
    # test_dataset = Flowers102(...)

    # TODO: Crear DataLoaders
    # train_loader = DataLoader(...)
    # val_loader = DataLoader(...)
    # test_loader = DataLoader(...)

    # return train_loader, val_loader, test_loader
    pass


# ============================================
# PARTE 2: MODELO
# ============================================


def create_model(num_classes, pretrained=True):
    """
    Crea modelo ResNet con nuevo clasificador.

    Args:
        num_classes: Número de clases (102)
        pretrained: Si cargar pesos de ImageNet

    Returns:
        model: Modelo modificado
    """
    # TODO: Cargar ResNet preentrenado
    # model = models.resnet18(weights=...)

    # TODO: Obtener features de entrada del clasificador
    # in_features = model.fc.in_features

    # TODO: Reemplazar clasificador
    # model.fc = nn.Linear(in_features, num_classes)

    # return model
    pass


def freeze_backbone(model):
    """
    Congela todas las capas excepto el clasificador.

    Args:
        model: Modelo ResNet
    """
    # TODO: Congelar todas las capas
    # for param in model.parameters():
    #     param.requires_grad = False

    # TODO: Descongelar clasificador
    # for param in model.fc.parameters():
    #     param.requires_grad = True
    pass


def unfreeze_layers(model, num_layers=2):
    """
    Descongela las últimas N capas del modelo.

    Args:
        model: Modelo ResNet
        num_layers: Número de capas a descongelar
    """
    # TODO: Implementar descongelamiento gradual
    # Las capas de ResNet son: layer1, layer2, layer3, layer4
    pass


# ============================================
# PARTE 3: ENTRENAMIENTO
# ============================================


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Entrena una época.

    Args:
        model: Modelo
        loader: DataLoader de entrenamiento
        criterion: Función de pérdida
        optimizer: Optimizador
        device: Dispositivo (cuda/cpu)

    Returns:
        avg_loss: Pérdida promedio
        accuracy: Precisión
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # TODO: Implementar loop de entrenamiento
    # for images, labels in tqdm(loader, desc='Training'):
    #     images, labels = images.to(device), labels.to(device)
    #
    #     optimizer.zero_grad()
    #     outputs = model(images)
    #     loss = criterion(outputs, labels)
    #     loss.backward()
    #     optimizer.step()
    #
    #     total_loss += loss.item()
    #     _, predicted = outputs.max(1)
    #     total += labels.size(0)
    #     correct += predicted.eq(labels).sum().item()

    # avg_loss = total_loss / len(loader)
    # accuracy = 100. * correct / total
    # return avg_loss, accuracy
    pass


def validate(model, loader, criterion, device):
    """
    Valida el modelo.

    Args:
        model: Modelo
        loader: DataLoader de validación
        criterion: Función de pérdida
        device: Dispositivo

    Returns:
        avg_loss: Pérdida promedio
        accuracy: Precisión
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # TODO: Implementar validación
    # with torch.no_grad():
    #     for images, labels in tqdm(loader, desc='Validating'):
    #         ...
    pass


def train_model(model, train_loader, val_loader, config):
    """
    Entrena el modelo completo con 2 fases.

    Args:
        model: Modelo
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        config: Configuración

    Returns:
        history: Diccionario con métricas de entrenamiento
    """
    device = config["device"]
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ==========================================
    # FASE 1: Feature Extraction
    # ==========================================
    print("\n" + "=" * 50)
    print("FASE 1: Feature Extraction")
    print("=" * 50)

    # TODO: Congelar backbone
    # freeze_backbone(model)

    # TODO: Crear optimizer solo para parámetros entrenables
    # optimizer = torch.optim.Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=config['lr_feature_extraction']
    # )

    # TODO: Entrenar por epochs_feature_extraction epochs
    # for epoch in range(config['epochs_feature_extraction']):
    #     train_loss, train_acc = train_epoch(...)
    #     val_loss, val_acc = validate(...)
    #     ...

    # ==========================================
    # FASE 2: Fine-tuning
    # ==========================================
    print("\n" + "=" * 50)
    print("FASE 2: Fine-tuning")
    print("=" * 50)

    # TODO: Descongelar últimas capas
    # unfreeze_layers(model, num_layers=2)

    # TODO: Crear nuevo optimizer con LR más bajo
    # optimizer = torch.optim.Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=config['lr_fine_tuning']
    # )

    # TODO: Scheduler (opcional)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(...)

    # TODO: Entrenar por epochs_fine_tuning epochs
    # for epoch in range(config['epochs_fine_tuning']):
    #     ...

    return history


# ============================================
# PARTE 4: EVALUACIÓN
# ============================================


def evaluate_model(model, test_loader, device):
    """
    Evalúa el modelo en el conjunto de test.

    Args:
        model: Modelo entrenado
        test_loader: DataLoader de test
        device: Dispositivo

    Returns:
        accuracy: Precisión top-1
        top5_accuracy: Precisión top-5
    """
    model.eval()
    correct = 0
    correct_top5 = 0
    total = 0

    # TODO: Implementar evaluación con top-1 y top-5 accuracy
    # with torch.no_grad():
    #     for images, labels in tqdm(test_loader, desc='Testing'):
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = model(images)
    #
    #         # Top-1
    #         _, predicted = outputs.max(1)
    #         correct += predicted.eq(labels).sum().item()
    #
    #         # Top-5
    #         _, top5_pred = outputs.topk(5, 1, True, True)
    #         correct_top5 += top5_pred.eq(labels.view(-1, 1)).sum().item()
    #
    #         total += labels.size(0)

    # accuracy = 100. * correct / total
    # top5_accuracy = 100. * correct_top5 / total
    # return accuracy, top5_accuracy
    pass


def plot_training_history(history):
    """
    Visualiza el historial de entrenamiento.

    Args:
        history: Diccionario con métricas
    """
    # TODO: Crear gráficos de loss y accuracy
    pass


# ============================================
# MAIN
# ============================================


def main():
    print("=" * 60)
    print("Clasificador de Flores - Flowers-102")
    print("=" * 60)

    # 1. Cargar datos
    print("\n[1/4] Cargando datos...")
    train_loader, val_loader, test_loader = load_data(CONFIG)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 2. Crear modelo
    print("\n[2/4] Creando modelo...")
    model = create_model(CONFIG["num_classes"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros totales: {total_params:,}")

    # 3. Entrenar
    print("\n[3/4] Entrenando modelo...")
    history = train_model(model, train_loader, val_loader, CONFIG)

    # 4. Evaluar
    print("\n[4/4] Evaluando modelo...")
    accuracy, top5_accuracy = evaluate_model(model, test_loader, CONFIG["device"])

    print("\n" + "=" * 60)
    print("RESULTADOS FINALES")
    print("=" * 60)
    print(f"Test Accuracy (Top-1): {accuracy:.2f}%")
    print(f"Test Accuracy (Top-5): {top5_accuracy:.2f}%")

    if accuracy >= 85:
        print("\n✅ ¡Objetivo alcanzado! (>= 85%)")
    else:
        print(f"\n⚠️  Accuracy por debajo del objetivo ({accuracy:.2f}% < 85%)")

    # Guardar modelo
    torch.save(model.state_dict(), "flowers_model.pth")
    print("\nModelo guardado en: flowers_model.pth")


if __name__ == "__main__":
    main()
