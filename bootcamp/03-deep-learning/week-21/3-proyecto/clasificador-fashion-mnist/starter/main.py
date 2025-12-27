"""
Proyecto: Clasificador Fashion-MNIST
Bootcamp IA: Zero to Hero | Semana 21

Objetivo: Alcanzar ≥88% accuracy en el test set de Fashion-MNIST

Completa las secciones marcadas con TODO.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================
# CONFIGURACIÓN
# ============================================

# Detectar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Hiperparámetros
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15

# Nombres de las clases
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# ============================================
# 1. CARGAR DATOS
# ============================================
print("\n" + "=" * 50)
print("1. CARGANDO DATOS")
print("=" * 50)

# TODO: Definir transformaciones
# - Convertir a tensor
# - Normalizar a media=0.5, std=0.5
transform = None  # TODO: transforms.Compose([...])

# TODO: Cargar dataset de entrenamiento
# train_dataset = datasets.FashionMNIST(
#     root='./data',
#     train=True,
#     download=True,
#     transform=transform
# )

# TODO: Cargar dataset de test
# test_dataset = datasets.FashionMNIST(...)

# TODO: Crear DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Verificar datos
# print(f"Train samples: {len(train_dataset)}")
# print(f"Test samples: {len(test_dataset)}")
# print(f"Image shape: {train_dataset[0][0].shape}")


# ============================================
# 2. DEFINIR MODELO
# ============================================
print("\n" + "=" * 50)
print("2. DEFINIENDO MODELO")
print("=" * 50)


class FashionClassifier(nn.Module):
    """
    Red neuronal para clasificar Fashion-MNIST.

    Arquitectura sugerida:
    - Input: 784 (28x28 aplanado)
    - Hidden 1: 512 unidades + ReLU + Dropout
    - Hidden 2: 256 unidades + ReLU + Dropout
    - Output: 10 clases
    """

    def __init__(self):
        super().__init__()
        # TODO: Definir capas
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(784, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 10)
        # self.dropout = nn.Dropout(0.2)
        pass

    def forward(self, x):
        # TODO: Implementar forward pass
        # x = self.flatten(x)
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.fc3(x)
        # return x
        pass


# TODO: Crear instancia del modelo y mover a device
# model = FashionClassifier().to(device)
# print(model)

# TODO: Contar parámetros
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total parámetros: {total_params:,}")


# ============================================
# 3. CONFIGURAR ENTRENAMIENTO
# ============================================
print("\n" + "=" * 50)
print("3. CONFIGURANDO ENTRENAMIENTO")
print("=" * 50)

# TODO: Definir función de pérdida
# criterion = nn.CrossEntropyLoss()

# TODO: Definir optimizador
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# print(f"Loss: {criterion}")
# print(f"Optimizer: Adam, lr={LEARNING_RATE}")


# ============================================
# 4. FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN
# ============================================


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Entrena el modelo por una época.

    Returns:
        tuple: (loss promedio, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # TODO: Implementar loop de entrenamiento
    # for data, target in loader:
    #     data, target = data.to(device), target.to(device)
    #
    #     optimizer.zero_grad()
    #     output = model(data)
    #     loss = criterion(output, target)
    #     loss.backward()
    #     optimizer.step()
    #
    #     running_loss += loss.item()
    #     _, predicted = output.max(1)
    #     total += target.size(0)
    #     correct += predicted.eq(target).sum().item()

    # avg_loss = running_loss / len(loader)
    # accuracy = 100. * correct / total
    # return avg_loss, accuracy

    return 0.0, 0.0  # TODO: Reemplazar


def evaluate(model, loader, criterion, device):
    """
    Evalúa el modelo en un dataset.

    Returns:
        tuple: (loss promedio, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # TODO: Implementar evaluación
    # with torch.no_grad():
    #     for data, target in loader:
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         loss = criterion(output, target)
    #
    #         running_loss += loss.item()
    #         _, predicted = output.max(1)
    #         total += target.size(0)
    #         correct += predicted.eq(target).sum().item()

    # avg_loss = running_loss / len(loader)
    # accuracy = 100. * correct / total
    # return avg_loss, accuracy

    return 0.0, 0.0  # TODO: Reemplazar


# ============================================
# 5. ENTRENAR MODELO
# ============================================
print("\n" + "=" * 50)
print("5. ENTRENANDO MODELO")
print("=" * 50)

# Historial de métricas
history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

# TODO: Loop de entrenamiento
# for epoch in range(EPOCHS):
#     train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
#     test_loss, test_acc = evaluate(model, test_loader, criterion, device)
#
#     history['train_loss'].append(train_loss)
#     history['train_acc'].append(train_acc)
#     history['test_loss'].append(test_loss)
#     history['test_acc'].append(test_acc)
#
#     print(f"Epoch {epoch+1}/{EPOCHS}")
#     print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
#     print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")


# ============================================
# 6. VISUALIZACIÓN
# ============================================
print("\n" + "=" * 50)
print("6. VISUALIZACIÓN")
print("=" * 50)


def plot_history(history):
    """Grafica el historial de entrenamiento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["test_loss"], label="Test")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss por Época")
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["test_acc"], label="Test")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy por Época")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("✓ Gráfica guardada en 'training_history.png'")


def show_predictions(model, loader, device, num_samples=10):
    """Muestra ejemplos de predicciones."""
    model.eval()

    # Obtener un batch
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)

    with torch.no_grad():
        output = model(data)
        _, predicted = output.max(1)

    # Visualizar
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            img = data[i].cpu().squeeze()
            true_label = CLASS_NAMES[target[i].item()]
            pred_label = CLASS_NAMES[predicted[i].item()]

            ax.imshow(img, cmap="gray")
            color = "green" if target[i] == predicted[i] else "red"
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    plt.show()
    print("✓ Predicciones guardadas en 'predictions.png'")


# TODO: Descomentar para visualizar
# plot_history(history)
# show_predictions(model, test_loader, device)


# ============================================
# 7. GUARDAR MODELO
# ============================================
print("\n" + "=" * 50)
print("7. GUARDANDO MODELO")
print("=" * 50)

# TODO: Guardar modelo
# torch.save(model.state_dict(), 'fashion_classifier.pth')
# print("✓ Modelo guardado en 'fashion_classifier.pth'")


# ============================================
# VERIFICACIÓN FINAL
# ============================================
print("\n" + "=" * 50)
print("VERIFICACIÓN FINAL")
print("=" * 50)

# TODO: Descomentar cuando completes el proyecto
# final_acc = history['test_acc'][-1]
# print(f"Accuracy final en test: {final_acc:.2f}%")
#
# if final_acc >= 88:
#     print("✅ ¡Objetivo alcanzado! Accuracy ≥ 88%")
# else:
#     print(f"❌ Accuracy {final_acc:.2f}% < 88%. Intenta:")
#     print("   - Más epochs")
#     print("   - Ajustar learning rate")
#     print("   - Modificar arquitectura")
