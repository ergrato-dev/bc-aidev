"""
Ejercicio 03: LeNet-5 en MNIST
==============================

Implementar y entrenar LeNet-5 en MNIST.

Instrucciones:
- Lee cada paso en el README.md
- Descomenta las secciones correspondientes
- Ejecuta y entrena el modelo
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================
# PASO 1: Cargar Dataset MNIST
# ============================================
print("--- Paso 1: Cargar Dataset MNIST ---")

# LeNet-5 original espera imágenes 32×32
# MNIST original es 28×28, redimensionamos

# Descomenta las siguientes líneas:
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
#
# train_dataset = datasets.MNIST(
#     root='./data',
#     train=True,
#     download=True,
#     transform=transform
# )
# test_dataset = datasets.MNIST(
#     root='./data',
#     train=False,
#     transform=transform
# )
#
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
#
# print(f"Training samples: {len(train_dataset)}")
# print(f"Test samples: {len(test_dataset)}")
# print(f"Image shape: {train_dataset[0][0].shape}")

print()

# ============================================
# PASO 2: Arquitectura LeNet-5 Original
# ============================================
print("--- Paso 2: Arquitectura LeNet-5 Original ---")

# LeNet-5 (LeCun et al., 1998)
# Usa tanh como activación y average pooling

# Descomenta las siguientes líneas:
# class LeNet5(nn.Module):
#     """
#     LeNet-5 original.
#
#     Arquitectura:
#     Input: 32×32×1
#     C1: Conv 5×5, 6 filtros -> 28×28×6
#     S2: AvgPool 2×2 -> 14×14×6
#     C3: Conv 5×5, 16 filtros -> 10×10×16
#     S4: AvgPool 2×2 -> 5×5×16
#     C5: Conv 5×5, 120 filtros -> 1×1×120
#     F6: FC 120 -> 84
#     Output: FC 84 -> 10
#     """
#
#     def __init__(self, num_classes=10):
#         super().__init__()
#
#         # Capas convolucionales
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
#         self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
#
#         # Pooling
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
#
#         # Fully connected
#         self.fc1 = nn.Linear(120, 84)
#         self.fc2 = nn.Linear(84, num_classes)
#
#         # Activación (original usa tanh)
#         self.activation = nn.Tanh()
#
#     def forward(self, x):
#         # C1 -> S2
#         x = self.pool(self.activation(self.conv1(x)))
#         # C3 -> S4
#         x = self.pool(self.activation(self.conv2(x)))
#         # C5
#         x = self.activation(self.conv3(x))
#         # Flatten
#         x = x.view(x.size(0), -1)
#         # F6
#         x = self.activation(self.fc1(x))
#         # Output
#         x = self.fc2(x)
#         return x
#
# # Verificar arquitectura
# model_original = LeNet5()
# x = torch.randn(1, 1, 32, 32)
# output = model_original(x)
# print(f"Input: {x.shape}")
# print(f"Output: {output.shape}")
# print(f"Parámetros: {sum(p.numel() for p in model_original.parameters()):,}")

print()

# ============================================
# PASO 3: LeNet-5 Moderna
# ============================================
print("--- Paso 3: LeNet-5 Moderna ---")

# Versión moderna con ReLU, BatchNorm, MaxPool

# Descomenta las siguientes líneas:
# class LeNet5Modern(nn.Module):
#     """LeNet-5 con mejoras modernas."""
#
#     def __init__(self, num_classes=10):
#         super().__init__()
#
#         self.features = nn.Sequential(
#             # C1: 32×32×1 -> 32×32×6 -> 16×16×6
#             nn.Conv2d(1, 6, kernel_size=5, padding=2),
#             nn.BatchNorm2d(6),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             # C3: 16×16×6 -> 12×12×16 -> 6×6×16
#             nn.Conv2d(6, 16, kernel_size=5),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             # C5: 6×6×16 -> 2×2×120
#             nn.Conv2d(16, 120, kernel_size=5),
#             nn.BatchNorm2d(120),
#             nn.ReLU(),
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(120 * 2 * 2, 84),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(84, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x
#
# # Verificar arquitectura moderna
# model_modern = LeNet5Modern()
# x = torch.randn(1, 1, 32, 32)
# output = model_modern(x)
# print(f"Input: {x.shape}")
# print(f"Output: {output.shape}")
# print(f"Parámetros: {sum(p.numel() for p in model_modern.parameters()):,}")

print()

# ============================================
# PASO 4: Funciones de Entrenamiento
# ============================================
print("--- Paso 4: Funciones de Entrenamiento ---")

# Funciones auxiliares para train y evaluate

# Descomenta las siguientes líneas:
# def train_epoch(model, loader, criterion, optimizer, device):
#     """Entrena una época."""
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#
#     for images, labels in loader:
#         images, labels = images.to(device), labels.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()
#
#     return total_loss / len(loader), 100. * correct / total
#
#
# def evaluate(model, loader, criterion, device):
#     """Evalúa el modelo."""
#     model.eval()
#     total_loss = 0
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#
#             total_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
#
#     return total_loss / len(loader), 100. * correct / total
#
# print("Funciones de entrenamiento definidas.")

print()

# ============================================
# PASO 5: Loop de Entrenamiento
# ============================================
print("--- Paso 5: Loop de Entrenamiento ---")

# Entrenar el modelo

# Descomenta las siguientes líneas:
# # Configuración
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Usando dispositivo: {device}")
#
# # Modelo (usar versión moderna para mejor accuracy)
# model = LeNet5Modern(num_classes=10).to(device)
# print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
#
# # Loss y optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
#
# # Entrenamiento
# num_epochs = 10
# best_acc = 0
# history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
#
# print("\nIniciando entrenamiento...")
# for epoch in range(num_epochs):
#     train_loss, train_acc = train_epoch(
#         model, train_loader, criterion, optimizer, device
#     )
#     test_loss, test_acc = evaluate(
#         model, test_loader, criterion, device
#     )
#     scheduler.step()
#
#     # Guardar historial
#     history['train_loss'].append(train_loss)
#     history['train_acc'].append(train_acc)
#     history['test_loss'].append(test_loss)
#     history['test_acc'].append(test_acc)
#
#     # Guardar mejor modelo
#     if test_acc > best_acc:
#         best_acc = test_acc
#         torch.save(model.state_dict(), 'lenet5_best.pth')
#
#     print(f"Epoch {epoch+1}/{num_epochs}")
#     print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
#     print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
#
# print(f"\n✓ Mejor Test Accuracy: {best_acc:.2f}%")

print()

# ============================================
# PASO 6: Visualizar Filtros Aprendidos
# ============================================
print("--- Paso 6: Visualizar Filtros Aprendidos ---")

# Visualizar qué aprendió la primera capa

# Descomenta las siguientes líneas:
# import matplotlib.pyplot as plt
#
# def visualize_filters(model):
#     """Visualiza filtros de la primera capa convolucional."""
#     # Obtener primera capa conv
#     if hasattr(model, 'conv1'):
#         weights = model.conv1.weight.data.cpu()
#     else:
#         weights = model.features[0].weight.data.cpu()
#
#     n_filters = weights.shape[0]
#
#     fig, axes = plt.subplots(1, n_filters, figsize=(12, 2))
#
#     for i, ax in enumerate(axes):
#         w = weights[i, 0]
#         ax.imshow(w, cmap='gray')
#         ax.axis('off')
#         ax.set_title(f'F{i+1}')
#
#     plt.suptitle('Filtros Capa 1 (5×5)')
#     plt.tight_layout()
#     plt.savefig('lenet5_filters.png', dpi=150)
#     plt.close()
#     print("Filtros guardados en 'lenet5_filters.png'")
#
# # Cargar mejor modelo y visualizar
# model.load_state_dict(torch.load('lenet5_best.pth'))
# visualize_filters(model)

print()

# ============================================
# PASO 7: Visualizar Feature Maps
# ============================================
print("--- Paso 7: Visualizar Feature Maps ---")

# Ver cómo la red "ve" una imagen

# Descomenta las siguientes líneas:
# def visualize_feature_maps(model, image, device):
#     """Visualiza activaciones de la primera capa."""
#     model.eval()
#
#     with torch.no_grad():
#         x = image.unsqueeze(0).to(device)
#         if hasattr(model, 'conv1'):
#             features = torch.relu(model.conv1(x))
#         else:
#             features = model.features[:3](x)
#
#     features = features.cpu().squeeze(0)
#     n_maps = features.shape[0]
#
#     fig, axes = plt.subplots(1, n_maps + 1, figsize=(14, 2))
#
#     # Imagen original
#     axes[0].imshow(image.squeeze(), cmap='gray')
#     axes[0].set_title('Original')
#     axes[0].axis('off')
#
#     # Feature maps
#     for i in range(n_maps):
#         axes[i+1].imshow(features[i], cmap='viridis')
#         axes[i+1].axis('off')
#         axes[i+1].set_title(f'FM{i+1}')
#
#     plt.suptitle('Feature Maps Primera Capa')
#     plt.tight_layout()
#     plt.savefig('lenet5_feature_maps.png', dpi=150)
#     plt.close()
#     print("Feature maps guardados en 'lenet5_feature_maps.png'")
#
# # Visualizar con una imagen de test
# test_image, label = test_dataset[0]
# print(f"Imagen de prueba: dígito {label}")
# visualize_feature_maps(model, test_image, device)

print()

# ============================================
# PASO 8: Curvas de Entrenamiento
# ============================================
print("--- Paso 8: Curvas de Entrenamiento ---")

# Graficar loss y accuracy durante entrenamiento

# Descomenta las siguientes líneas:
# def plot_training_curves(history):
#     """Grafica curvas de entrenamiento."""
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#
#     # Loss
#     axes[0].plot(history['train_loss'], label='Train')
#     axes[0].plot(history['test_loss'], label='Test')
#     axes[0].set_xlabel('Epoch')
#     axes[0].set_ylabel('Loss')
#     axes[0].set_title('Loss por Época')
#     axes[0].legend()
#     axes[0].grid(True, alpha=0.3)
#
#     # Accuracy
#     axes[1].plot(history['train_acc'], label='Train')
#     axes[1].plot(history['test_acc'], label='Test')
#     axes[1].set_xlabel('Epoch')
#     axes[1].set_ylabel('Accuracy (%)')
#     axes[1].set_title('Accuracy por Época')
#     axes[1].legend()
#     axes[1].grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig('lenet5_training_curves.png', dpi=150)
#     plt.close()
#     print("Curvas guardadas en 'lenet5_training_curves.png'")
#
# plot_training_curves(history)

print()

# ============================================
# RESUMEN
# ============================================
print("=" * 50)
print("RESUMEN")
print("=" * 50)
print(
    """
LeNet-5 (1998) - Pionera en CNNs:

Arquitectura Original:
  Input 32×32 -> C1(6) -> S2 -> C3(16) -> S4 -> C5(120) -> F6(84) -> Output(10)

Características:
  - ~61,000 parámetros
  - Activación: tanh
  - Pooling: Average

Versión Moderna:
  - Activación: ReLU
  - Pooling: MaxPool
  - Regularización: BatchNorm + Dropout

Resultados esperados en MNIST:
  - LeNet-5 original: ~98-99% accuracy
  - LeNet-5 moderna: ~99%+ accuracy

Key Insights:
  - Los filtros aprenden detectores de bordes/texturas
  - Feature maps muestran qué "ve" la red
  - Más capas = características más abstractas
"""
)
