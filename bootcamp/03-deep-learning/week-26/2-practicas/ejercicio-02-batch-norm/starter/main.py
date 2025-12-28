"""
Ejercicio 02: Batch Normalization
=================================
Compara convergencia con y sin BatchNorm.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================
# PASO 1: Preparar CIFAR-10
# ============================================
print("--- Paso 1: Preparar CIFAR-10 ---")

# Descomenta las siguientes líneas:
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
# test_dataset = datasets.CIFAR10('data', train=False, transform=transform)
#
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=256)
#
# print(f'Train: {len(train_dataset)}, Test: {len(test_dataset)}')
# print(f'Clases: {train_dataset.classes}')

print()

# ============================================
# PASO 2: CNN SIN BatchNorm
# ============================================
print("--- Paso 2: CNN SIN BatchNorm ---")

# Descomenta las siguientes líneas:
# class CNNNoBatchNorm(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 8 * 8, 256)
#         self.fc2 = nn.Linear(256, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
#         x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
#         x = x.view(-1, 64 * 8 * 8)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)
#
# model_no_bn = CNNNoBatchNorm()
# print(f'CNN sin BatchNorm creada')

print()

# ============================================
# PASO 3: CNN CON BatchNorm
# ============================================
print("--- Paso 3: CNN CON BatchNorm ---")

# Descomenta las siguientes líneas:
# class CNNWithBatchNorm(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)  # BatchNorm para conv
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 8 * 8, 256)
#         self.bn3 = nn.BatchNorm1d(256)  # BatchNorm para FC
#         self.fc2 = nn.Linear(256, 10)
#
#     def forward(self, x):
#         # Patrón: Conv -> BN -> ReLU -> Pool
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = x.view(-1, 64 * 8 * 8)
#         x = F.relu(self.bn3(self.fc1(x)))
#         return self.fc2(x)
#
# model_with_bn = CNNWithBatchNorm()
# print(f'CNN con BatchNorm creada')

print()

# ============================================
# PASO 4: Funciones de Entrenamiento
# ============================================
print("--- Paso 4: Funciones de Entrenamiento ---")

# Descomenta las siguientes líneas:
# def train_epoch(model, loader, criterion, optimizer):
#     model.train()  # BatchNorm: usa estadísticas del batch
#     total_loss, correct, total = 0, 0, 0
#
#     for x, y in loader:
#         optimizer.zero_grad()
#         output = model(x)
#         loss = criterion(output, y)
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#         correct += (output.argmax(1) == y).sum().item()
#         total += y.size(0)
#
#     return total_loss / len(loader), correct / total
#
# def evaluate(model, loader, criterion):
#     model.eval()  # BatchNorm: usa running_mean/var
#     total_loss, correct, total = 0, 0, 0
#
#     with torch.no_grad():
#         for x, y in loader:
#             output = model(x)
#             total_loss += criterion(output, y).item()
#             correct += (output.argmax(1) == y).sum().item()
#             total += y.size(0)
#
#     return total_loss / len(loader), correct / total
#
# print('Funciones definidas')

print()

# ============================================
# PASO 5: Entrenar y Comparar
# ============================================
print("--- Paso 5: Entrenar y Comparar ---")

# Descomenta las siguientes líneas:
# criterion = nn.CrossEntropyLoss()
# epochs = 15
#
# # Sin BatchNorm: lr conservador
# optimizer_no_bn = torch.optim.Adam(model_no_bn.parameters(), lr=0.001)
# # Con BatchNorm: lr más agresivo
# optimizer_with_bn = torch.optim.Adam(model_with_bn.parameters(), lr=0.005)
#
# no_bn_losses, no_bn_accs = [], []
# with_bn_losses, with_bn_accs = [], []
#
# print('\nEntrenando SIN BatchNorm (lr=0.001)...')
# for epoch in range(epochs):
#     train_loss, train_acc = train_epoch(model_no_bn, train_loader, criterion, optimizer_no_bn)
#     test_loss, test_acc = evaluate(model_no_bn, test_loader, criterion)
#     no_bn_losses.append(train_loss)
#     no_bn_accs.append(test_acc)
#     if (epoch + 1) % 5 == 0:
#         print(f'Epoch {epoch+1}: Loss={train_loss:.4f}, Test Acc={test_acc:.4f}')
#
# print('\nEntrenando CON BatchNorm (lr=0.005)...')
# for epoch in range(epochs):
#     train_loss, train_acc = train_epoch(model_with_bn, train_loader, criterion, optimizer_with_bn)
#     test_loss, test_acc = evaluate(model_with_bn, test_loader, criterion)
#     with_bn_losses.append(train_loss)
#     with_bn_accs.append(test_acc)
#     if (epoch + 1) % 5 == 0:
#         print(f'Epoch {epoch+1}: Loss={train_loss:.4f}, Test Acc={test_acc:.4f}')

print()

# ============================================
# PASO 6: Visualizar Resultados
# ============================================
print("--- Paso 6: Visualizar Resultados ---")

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# # Loss comparison
# axes[0].plot(no_bn_losses, 'r-', label='Sin BatchNorm', linewidth=2)
# axes[0].plot(with_bn_losses, 'g-', label='Con BatchNorm', linewidth=2)
# axes[0].set_xlabel('Epoch')
# axes[0].set_ylabel('Training Loss')
# axes[0].set_title('Convergencia: BatchNorm acelera entrenamiento')
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)
#
# # Accuracy comparison
# axes[1].plot(no_bn_accs, 'r-', label='Sin BatchNorm', linewidth=2)
# axes[1].plot(with_bn_accs, 'g-', label='Con BatchNorm', linewidth=2)
# axes[1].set_xlabel('Epoch')
# axes[1].set_ylabel('Test Accuracy')
# axes[1].set_title('Test Accuracy por Época')
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig('batchnorm_comparison.png', dpi=150)
# plt.show()
#
# print('\n' + '='*50)
# print('RESUMEN')
# print('='*50)
# print(f'Sin BatchNorm - Acc final: {no_bn_accs[-1]:.4f}')
# print(f'Con BatchNorm - Acc final: {with_bn_accs[-1]:.4f}')
# print(f'Mejora: {with_bn_accs[-1] - no_bn_accs[-1]:.4f}')

print()
print("Ejercicio completado!")
