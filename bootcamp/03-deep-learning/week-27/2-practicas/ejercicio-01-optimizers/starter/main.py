"""
Ejercicio 01: Comparación de Optimizadores
==========================================

Compara SGD, SGD+Momentum, Adam y AdamW entrenando en MNIST.
Sigue las instrucciones del README.md y descomenta cada sección.
"""

# ============================================
# PASO 1: Imports y Configuración
# ============================================
print("--- Paso 1: Configuración ---")

# Descomenta las siguientes líneas:
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import time

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Usando dispositivo: {device}')

print()

# ============================================
# PASO 2: Definir el Modelo
# ============================================
print("--- Paso 2: Modelo ---")

# Descomenta las siguientes líneas:
# class SimpleNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(784, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 10)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         return self.fc3(x)

# print('Modelo SimpleNet definido')

print()

# ============================================
# PASO 3: Cargar Datos
# ============================================
print("--- Paso 3: Datos ---")

# Descomenta las siguientes líneas:
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1000)

# print(f'Train: {len(train_dataset)} imágenes')
# print(f'Test: {len(test_dataset)} imágenes')

print()

# ============================================
# PASO 4: Función de Entrenamiento
# ============================================
print("--- Paso 4: Función de Entrenamiento ---")

# Descomenta las siguientes líneas:
# def train_with_optimizer(optimizer_name, optimizer, model, epochs=5):
#     """Entrena el modelo y retorna historial de métricas."""
#     criterion = nn.CrossEntropyLoss()
#     history = {'loss': [], 'acc': [], 'time': 0}
#
#     start_time = time.time()
#
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         for x, y in train_loader:
#             x, y = x.to(device), y.to(device)
#
#             optimizer.zero_grad()
#             output = model(x)
#             loss = criterion(output, y)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#             correct += (output.argmax(1) == y).sum().item()
#             total += y.size(0)
#
#         epoch_loss = running_loss / len(train_loader)
#         epoch_acc = correct / total
#         history['loss'].append(epoch_loss)
#         history['acc'].append(epoch_acc)
#
#         print(f'{optimizer_name} - Epoch {epoch+1}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}')
#
#     history['time'] = time.time() - start_time
#     return history

# print('Función train_with_optimizer definida')

print()

# ============================================
# PASO 5: Comparar Optimizadores
# ============================================
print("--- Paso 5: Comparación ---")

# Descomenta las siguientes líneas:
# optimizers_config = {
#     'SGD': lambda model: optim.SGD(model.parameters(), lr=0.01),
#     'SGD+Momentum': lambda model: optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
#     'Adam': lambda model: optim.Adam(model.parameters(), lr=0.001),
#     'AdamW': lambda model: optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01),
# }

# results = {}

# for name, opt_fn in optimizers_config.items():
#     print(f'\n{"="*50}')
#     print(f'Entrenando con {name}')
#     print("="*50)
#
#     # Crear modelo nuevo para cada optimizador
#     model = SimpleNet().to(device)
#     optimizer = opt_fn(model)
#
#     results[name] = train_with_optimizer(name, optimizer, model, epochs=5)

print()

# ============================================
# PASO 6: Visualizar Resultados
# ============================================
print("--- Paso 6: Visualización ---")

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # Gráfica de Loss
# for name, history in results.items():
#     axes[0].plot(history['loss'], label=name, marker='o')
# axes[0].set_xlabel('Epoch')
# axes[0].set_ylabel('Loss')
# axes[0].set_title('Comparación de Loss por Optimizador')
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)

# # Gráfica de Accuracy
# for name, history in results.items():
#     axes[1].plot(history['acc'], label=name, marker='o')
# axes[1].set_xlabel('Epoch')
# axes[1].set_ylabel('Accuracy')
# axes[1].set_title('Comparación de Accuracy por Optimizador')
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('optimizers_comparison.png', dpi=150)
# plt.show()

# # Tabla resumen
# print('\n' + '='*60)
# print('RESUMEN DE RESULTADOS')
# print('='*60)
# print(f'{"Optimizador":<15} {"Loss Final":<12} {"Acc Final":<12} {"Tiempo (s)":<10}')
# print('-'*60)
# for name, history in results.items():
#     print(f'{name:<15} {history["loss"][-1]:<12.4f} {history["acc"][-1]:<12.4f} {history["time"]:<10.2f}')

print()
print("=" * 50)
print("Ejercicio completado!")
print("=" * 50)
