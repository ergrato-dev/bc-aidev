"""
Ejercicio 02: Learning Rate Schedules
=====================================

Compara StepLR, CosineAnnealingLR y OneCycleLR.
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
# from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Usando dispositivo: {device}')

print()

# ============================================
# PASO 2: Modelo y Datos
# ============================================
print("--- Paso 2: Modelo y Datos ---")

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

# # Cargar datos
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
# train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# print(f'Train: {len(train_dataset)} imágenes')
# print(f'Batches por época: {len(train_loader)}')

print()

# ============================================
# PASO 3: Función de Entrenamiento con Scheduler
# ============================================
print("--- Paso 3: Función de Entrenamiento ---")

# Descomenta las siguientes líneas:
# def train_with_scheduler(scheduler_name, model, optimizer, scheduler, epochs=10, step_per_batch=False):
#     """
#     Entrena con un scheduler específico.
#
#     Args:
#         step_per_batch: Si True, hace scheduler.step() cada batch (para OneCycleLR)
#     """
#     criterion = nn.CrossEntropyLoss()
#     history = {'loss': [], 'lr': []}
#
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
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
#
#             # OneCycleLR hace step por batch
#             if step_per_batch:
#                 scheduler.step()
#
#         # Los demás schedulers hacen step por época
#         if not step_per_batch:
#             scheduler.step()
#
#         epoch_loss = running_loss / len(train_loader)
#         current_lr = optimizer.param_groups[0]['lr']
#
#         history['loss'].append(epoch_loss)
#         history['lr'].append(current_lr)
#
#         print(f'{scheduler_name} - Epoch {epoch+1}: loss={epoch_loss:.4f}, lr={current_lr:.6f}')
#
#     return history

# print('Función train_with_scheduler definida')

print()

# ============================================
# PASO 4: Configurar Schedulers
# ============================================
print("--- Paso 4: Configurar Schedulers ---")

# Descomenta las siguientes líneas:
# EPOCHS = 10
# LR_INICIAL = 0.1

# schedulers_config = {
#     'StepLR': {
#         'scheduler': lambda opt: StepLR(opt, step_size=3, gamma=0.5),
#         'step_per_batch': False
#     },
#     'CosineAnnealing': {
#         'scheduler': lambda opt: CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=0.001),
#         'step_per_batch': False
#     },
#     'OneCycleLR': {
#         'scheduler': lambda opt: OneCycleLR(
#             opt,
#             max_lr=0.1,
#             epochs=EPOCHS,
#             steps_per_epoch=len(train_loader)
#         ),
#         'step_per_batch': True  # ¡Importante!
#     },
# }

# print(f'Configurados {len(schedulers_config)} schedulers')
# print(f'Épocas: {EPOCHS}, LR inicial: {LR_INICIAL}')

print()

# ============================================
# PASO 5: Entrenar con Cada Scheduler
# ============================================
print("--- Paso 5: Entrenamiento ---")

# Descomenta las siguientes líneas:
# results = {}

# for name, config in schedulers_config.items():
#     print(f'\n{"="*50}')
#     print(f'Entrenando con {name}')
#     print("="*50)
#
#     # Nuevo modelo y optimizador para cada experimento
#     model = SimpleNet().to(device)
#     optimizer = optim.SGD(model.parameters(), lr=LR_INICIAL)
#     scheduler = config['scheduler'](optimizer)
#
#     results[name] = train_with_scheduler(
#         name, model, optimizer, scheduler,
#         epochs=EPOCHS,
#         step_per_batch=config['step_per_batch']
#     )

print()

# ============================================
# PASO 6: Visualizar Resultados
# ============================================
print("--- Paso 6: Visualización ---")

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # Gráfica de Learning Rate
# for name, history in results.items():
#     axes[0].plot(history['lr'], label=name, marker='o')
# axes[0].set_xlabel('Epoch')
# axes[0].set_ylabel('Learning Rate')
# axes[0].set_title('Evolución del Learning Rate')
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)
# axes[0].set_yscale('log')

# # Gráfica de Loss
# for name, history in results.items():
#     axes[1].plot(history['loss'], label=name, marker='o')
# axes[1].set_xlabel('Epoch')
# axes[1].set_ylabel('Loss')
# axes[1].set_title('Evolución del Loss')
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('lr_schedules_comparison.png', dpi=150)
# plt.show()

# # Resumen
# print('\n' + '='*50)
# print('RESUMEN')
# print('='*50)
# for name, history in results.items():
#     print(f'{name}: Loss final={history["loss"][-1]:.4f}, LR final={history["lr"][-1]:.6f}')

print()
print("=" * 50)
print("Ejercicio completado!")
print("=" * 50)
