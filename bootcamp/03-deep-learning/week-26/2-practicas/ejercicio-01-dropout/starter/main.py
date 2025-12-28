"""
Ejercicio 01: Implementar Dropout
=================================
Compara modelos con y sin Dropout para ver su efecto en overfitting.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================
# PASO 1: Cargar MNIST
# ============================================
print("--- Paso 1: Cargar MNIST ---")

# Descomenta las siguientes líneas:
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
#
# train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST('data', train=False, transform=transform)
#
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1000)
#
# print(f'Train samples: {len(train_dataset)}')
# print(f'Test samples: {len(test_dataset)}')

print()

# ============================================
# PASO 2: Modelo SIN Dropout
# ============================================
print("--- Paso 2: Modelo SIN Dropout ---")

# Descomenta las siguientes líneas:
# model_no_dropout = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(784, 512),
#     nn.ReLU(),
#     nn.Linear(512, 256),
#     nn.ReLU(),
#     nn.Linear(256, 10)
# )
#
# total_params = sum(p.numel() for p in model_no_dropout.parameters())
# print(f'Modelo sin Dropout - Parámetros: {total_params:,}')

print()

# ============================================
# PASO 3: Modelo CON Dropout
# ============================================
print("--- Paso 3: Modelo CON Dropout ---")

# Descomenta las siguientes líneas:
# model_with_dropout = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(784, 512),
#     nn.ReLU(),
#     nn.Dropout(0.5),      # 50% de neuronas apagadas
#     nn.Linear(512, 256),
#     nn.ReLU(),
#     nn.Dropout(0.3),      # 30% de neuronas apagadas
#     nn.Linear(256, 10)
# )
#
# print('Modelo con Dropout creado')
# print('Dropout layers: p=0.5 después de capa 1, p=0.3 después de capa 2')

print()

# ============================================
# PASO 4: Función de Entrenamiento
# ============================================
print("--- Paso 4: Función de Entrenamiento ---")

# Descomenta las siguientes líneas:
# def train_epoch(model, train_loader, criterion, optimizer):
#     """Entrena una época y retorna loss y accuracy."""
#     model.train()  # Activa Dropout
#     total_loss, correct, total = 0, 0, 0
#
#     for x, y in train_loader:
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
#     return total_loss / len(train_loader), correct / total
#
# print('Función train_epoch definida')

print()

# ============================================
# PASO 5: Función de Evaluación
# ============================================
print("--- Paso 5: Función de Evaluación ---")

# Descomenta las siguientes líneas:
# def evaluate(model, test_loader, criterion):
#     """Evalúa el modelo y retorna loss y accuracy."""
#     model.eval()  # Desactiva Dropout
#     total_loss, correct, total = 0, 0, 0
#
#     with torch.no_grad():
#         for x, y in test_loader:
#             output = model(x)
#             total_loss += criterion(output, y).item()
#             correct += (output.argmax(1) == y).sum().item()
#             total += y.size(0)
#
#     return total_loss / len(test_loader), correct / total
#
# print('Función evaluate definida')
# print('IMPORTANTE: model.eval() desactiva Dropout en inferencia')

print()

# ============================================
# PASO 6: Entrenar Ambos Modelos
# ============================================
print("--- Paso 6: Entrenar Ambos Modelos ---")

# Descomenta las siguientes líneas:
# criterion = nn.CrossEntropyLoss()
# epochs = 20
#
# # Listas para guardar métricas
# no_dropout_train_accs, no_dropout_test_accs = [], []
# with_dropout_train_accs, with_dropout_test_accs = [], []
#
# # Entrenar modelo SIN Dropout
# print('\nEntrenando modelo SIN Dropout...')
# optimizer_no = torch.optim.Adam(model_no_dropout.parameters(), lr=0.001)
#
# for epoch in range(epochs):
#     train_loss, train_acc = train_epoch(model_no_dropout, train_loader, criterion, optimizer_no)
#     test_loss, test_acc = evaluate(model_no_dropout, test_loader, criterion)
#
#     no_dropout_train_accs.append(train_acc)
#     no_dropout_test_accs.append(test_acc)
#
#     if (epoch + 1) % 5 == 0:
#         gap = train_acc - test_acc
#         print(f'Epoch {epoch+1}: Train={train_acc:.4f}, Test={test_acc:.4f}, Gap={gap:.4f}')
#
# # Entrenar modelo CON Dropout
# print('\nEntrenando modelo CON Dropout...')
# optimizer_with = torch.optim.Adam(model_with_dropout.parameters(), lr=0.001)
#
# for epoch in range(epochs):
#     train_loss, train_acc = train_epoch(model_with_dropout, train_loader, criterion, optimizer_with)
#     test_loss, test_acc = evaluate(model_with_dropout, test_loader, criterion)
#
#     with_dropout_train_accs.append(train_acc)
#     with_dropout_test_accs.append(test_acc)
#
#     if (epoch + 1) % 5 == 0:
#         gap = train_acc - test_acc
#         print(f'Epoch {epoch+1}: Train={train_acc:.4f}, Test={test_acc:.4f}, Gap={gap:.4f}')

print()

# ============================================
# PASO 7: Visualizar Resultados
# ============================================
print("--- Paso 7: Visualizar Resultados ---")

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# # Plot 1: Accuracy comparison
# axes[0].plot(no_dropout_train_accs, 'b-', label='Sin Dropout - Train', linewidth=2)
# axes[0].plot(no_dropout_test_accs, 'b--', label='Sin Dropout - Test', linewidth=2)
# axes[0].plot(with_dropout_train_accs, 'g-', label='Con Dropout - Train', linewidth=2)
# axes[0].plot(with_dropout_test_accs, 'g--', label='Con Dropout - Test', linewidth=2)
# axes[0].set_xlabel('Epoch')
# axes[0].set_ylabel('Accuracy')
# axes[0].set_title('Comparación: Con vs Sin Dropout')
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)
#
# # Plot 2: Gap comparison
# no_dropout_gaps = [t - v for t, v in zip(no_dropout_train_accs, no_dropout_test_accs)]
# with_dropout_gaps = [t - v for t, v in zip(with_dropout_train_accs, with_dropout_test_accs)]
#
# axes[1].plot(no_dropout_gaps, 'b-', label='Sin Dropout', linewidth=2)
# axes[1].plot(with_dropout_gaps, 'g-', label='Con Dropout', linewidth=2)
# axes[1].axhline(y=0.03, color='r', linestyle='--', label='Umbral overfitting (3%)')
# axes[1].set_xlabel('Epoch')
# axes[1].set_ylabel('Gap (Train - Test)')
# axes[1].set_title('Gap Train-Test: Indicador de Overfitting')
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig('dropout_comparison.png', dpi=150)
# plt.show()
#
# # Resumen final
# print('\n' + '='*50)
# print('RESUMEN FINAL')
# print('='*50)
# print(f'Sin Dropout - Test Acc final: {no_dropout_test_accs[-1]:.4f}')
# print(f'Sin Dropout - Gap final: {no_dropout_gaps[-1]:.4f}')
# print(f'Con Dropout - Test Acc final: {with_dropout_test_accs[-1]:.4f}')
# print(f'Con Dropout - Gap final: {with_dropout_gaps[-1]:.4f}')
# print('='*50)

print()
print("Ejercicio completado!")
