"""
Ejercicio 03: Data Augmentation
===============================
Implementa pipelines de augmentation y mide su impacto.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================
# PASO 1: Transform Básico (Sin Augmentation)
# ============================================
print("--- Paso 1: Transform Básico ---")

# Descomenta las siguientes líneas:
# basic_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# print('Transform básico: solo ToTensor + Normalize')

print()

# ============================================
# PASO 2: Transform con Augmentation
# ============================================
print("--- Paso 2: Transform con Augmentation ---")

# Descomenta las siguientes líneas:
# augment_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(degrees=15),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ColorJitter(
#         brightness=0.2,
#         contrast=0.2,
#         saturation=0.1
#     ),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# print('Transform con augmentation:')
# print('  - RandomHorizontalFlip(0.5)')
# print('  - RandomRotation(15°)')
# print('  - RandomCrop(32, padding=4)')
# print('  - ColorJitter(brightness, contrast, saturation)')

print()

# ============================================
# PASO 3: Visualizar Augmentations
# ============================================
print("--- Paso 3: Visualizar Augmentations ---")

# Descomenta las siguientes líneas:
# # Cargar una imagen de ejemplo
# temp_dataset = datasets.CIFAR10('data', train=True, download=True)
# original_img, label = temp_dataset[0]
#
# # Transform solo para visualización (sin normalizar)
# vis_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(degrees=15),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
# ])
#
# fig, axes = plt.subplots(2, 5, figsize=(12, 5))
# fig.suptitle(f'Data Augmentation - Clase: {temp_dataset.classes[label]}')
#
# # Primera fila: original
# for i in range(5):
#     axes[0, i].imshow(original_img)
#     axes[0, i].set_title('Original' if i == 0 else '')
#     axes[0, i].axis('off')
#
# # Segunda fila: augmented
# for i in range(5):
#     aug_img = vis_transform(original_img)
#     axes[1, i].imshow(aug_img)
#     axes[1, i].set_title(f'Aug {i+1}')
#     axes[1, i].axis('off')
#
# plt.tight_layout()
# plt.savefig('augmentation_examples.png', dpi=150)
# plt.show()
# print('Visualización guardada en augmentation_examples.png')

print()

# ============================================
# PASO 4: Crear Datasets y Modelo
# ============================================
print("--- Paso 4: Crear Datasets y Modelo ---")

# Descomenta las siguientes líneas:
# # Dataset SIN augmentation
# train_no_aug = datasets.CIFAR10('data', train=True, transform=basic_transform)
# # Dataset CON augmentation
# train_with_aug = datasets.CIFAR10('data', train=True, transform=augment_transform)
# # Test SIEMPRE sin augmentation
# test_dataset = datasets.CIFAR10('data', train=False, transform=basic_transform)
#
# loader_no_aug = DataLoader(train_no_aug, batch_size=128, shuffle=True)
# loader_with_aug = DataLoader(train_with_aug, batch_size=128, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=256)
#
# # Modelo simple
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 8 * 8, 256)
#         self.fc2 = nn.Linear(256, 10)
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = x.view(-1, 64 * 8 * 8)
#         x = self.dropout(F.relu(self.fc1(x)))
#         return self.fc2(x)
#
# print('Datasets y modelo creados')

print()

# ============================================
# PASO 5: Entrenar y Comparar
# ============================================
print("--- Paso 5: Entrenar y Comparar ---")

# Descomenta las siguientes líneas:
# def train_model(model, train_loader, test_loader, epochs=20):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#     train_accs, test_accs = [], []
#
#     for epoch in range(epochs):
#         model.train()
#         correct, total = 0, 0
#         for x, y in train_loader:
#             optimizer.zero_grad()
#             output = model(x)
#             loss = criterion(output, y)
#             loss.backward()
#             optimizer.step()
#             correct += (output.argmax(1) == y).sum().item()
#             total += y.size(0)
#         train_accs.append(correct / total)
#
#         model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for x, y in test_loader:
#                 output = model(x)
#                 correct += (output.argmax(1) == y).sum().item()
#                 total += y.size(0)
#         test_accs.append(correct / total)
#
#         if (epoch + 1) % 5 == 0:
#             print(f'Epoch {epoch+1}: Train={train_accs[-1]:.4f}, Test={test_accs[-1]:.4f}')
#
#     return train_accs, test_accs
#
# # Entrenar SIN augmentation
# print('\nEntrenando SIN Data Augmentation...')
# model_no_aug = SimpleCNN()
# no_aug_train, no_aug_test = train_model(model_no_aug, loader_no_aug, test_loader)
#
# # Entrenar CON augmentation
# print('\nEntrenando CON Data Augmentation...')
# model_with_aug = SimpleCNN()
# aug_train, aug_test = train_model(model_with_aug, loader_with_aug, test_loader)

print()

# ============================================
# PASO 6: Analizar Resultados
# ============================================
print("--- Paso 6: Analizar Resultados ---")

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# # Accuracy curves
# axes[0].plot(no_aug_train, 'r-', label='Sin Aug - Train', linewidth=2)
# axes[0].plot(no_aug_test, 'r--', label='Sin Aug - Test', linewidth=2)
# axes[0].plot(aug_train, 'g-', label='Con Aug - Train', linewidth=2)
# axes[0].plot(aug_test, 'g--', label='Con Aug - Test', linewidth=2)
# axes[0].set_xlabel('Epoch')
# axes[0].set_ylabel('Accuracy')
# axes[0].set_title('Efecto de Data Augmentation')
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)
#
# # Gap comparison
# no_aug_gaps = [t - v for t, v in zip(no_aug_train, no_aug_test)]
# aug_gaps = [t - v for t, v in zip(aug_train, aug_test)]
#
# axes[1].plot(no_aug_gaps, 'r-', label='Sin Augmentation', linewidth=2)
# axes[1].plot(aug_gaps, 'g-', label='Con Augmentation', linewidth=2)
# axes[1].axhline(y=0.1, color='orange', linestyle='--', label='Umbral 10%')
# axes[1].set_xlabel('Epoch')
# axes[1].set_ylabel('Gap (Train - Test)')
# axes[1].set_title('Reducción de Overfitting')
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig('augmentation_comparison.png', dpi=150)
# plt.show()
#
# print('\n' + '='*50)
# print('RESUMEN')
# print('='*50)
# print(f'Sin Augmentation - Test final: {no_aug_test[-1]:.4f}, Gap: {no_aug_gaps[-1]:.4f}')
# print(f'Con Augmentation - Test final: {aug_test[-1]:.4f}, Gap: {aug_gaps[-1]:.4f}')
# print(f'Mejora en Test: {aug_test[-1] - no_aug_test[-1]:.4f}')
# print(f'Reducción de Gap: {no_aug_gaps[-1] - aug_gaps[-1]:.4f}')

print()
print("Ejercicio completado!")
