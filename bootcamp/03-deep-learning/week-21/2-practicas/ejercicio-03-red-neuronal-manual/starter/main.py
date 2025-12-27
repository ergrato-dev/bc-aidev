"""
Ejercicio 03: Red Neuronal Manual
Bootcamp IA: Zero to Hero | Semana 21

Descomenta cada sección según avances en el ejercicio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print("=" * 60)
print("EJERCICIO 03: RED NEURONAL MANUAL")
print("=" * 60)

# ============================================
# PASO 1: Estructura de nn.Module
# ============================================
print("\n--- Paso 1: Estructura de nn.Module ---")

# class RedSimple(nn.Module):
#     def __init__(self):
#         super().__init__()  # ¡Siempre llamar al padre!
#         self.fc1 = nn.Linear(10, 5)
#
#     def forward(self, x):
#         return self.fc1(x)

# Crear instancia
# modelo_simple = RedSimple()
# print(f"Modelo simple:\n{modelo_simple}")

# Probar forward
# x = torch.randn(1, 10)
# output = modelo_simple(x)  # Llama forward() automáticamente
# print(f"\nInput shape: {x.shape}")
# print(f"Output shape: {output.shape}")

print()

# ============================================
# PASO 2: Definir Arquitectura Completa
# ============================================
print("\n--- Paso 2: Definir Arquitectura Completa ---")

# class RedNeuronal(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#
#         # Definir capas
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
#         self.fc3 = nn.Linear(hidden_size // 2, output_size)
#
#         # Capas de regularización
#         self.dropout = nn.Dropout(0.2)
#         self.bn1 = nn.BatchNorm1d(hidden_size)
#
#     def forward(self, x):
#         # Capa 1: Linear -> BatchNorm -> ReLU -> Dropout
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         # Capa 2: Linear -> ReLU -> Dropout
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#
#         # Capa de salida (sin activación para CrossEntropyLoss)
#         x = self.fc3(x)
#         return x

# Crear modelo para clasificación MNIST-like
# model = RedNeuronal(input_size=784, hidden_size=256, output_size=10)
# print(f"Arquitectura:\n{model}")

print()

# ============================================
# PASO 3: Inspeccionar el Modelo
# ============================================
print("\n--- Paso 3: Inspeccionar el Modelo ---")

# # Total de parámetros
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total parámetros: {total_params:,}")
# print(f"Parámetros entrenables: {trainable_params:,}")

# # Ver cada capa y sus parámetros
# print("\nParámetros por capa:")
# for name, param in model.named_parameters():
#     print(f"  {name}: {param.shape}")

# # Ver submódulos
# print("\nSubmódulos:")
# for name, module in model.named_modules():
#     if name:  # Excluir el módulo raíz
#         print(f"  {name}: {type(module).__name__}")

print()

# ============================================
# PASO 4: Loss y Optimizer
# ============================================
print("\n--- Paso 4: Loss y Optimizer ---")

# # Función de pérdida para clasificación multiclase
# criterion = nn.CrossEntropyLoss()

# # Optimizador Adam
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# print(f"Criterion: {criterion}")
# print(f"Optimizer: {type(optimizer).__name__}")
# print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

# # Alternativas de optimizadores
# # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

print()

# ============================================
# PASO 5: Training Loop Completo
# ============================================
print("\n--- Paso 5: Training Loop Completo ---")

# # Datos sintéticos para demostración
# torch.manual_seed(42)
# X_train = torch.randn(1000, 784)  # 1000 muestras, 784 features
# y_train = torch.randint(0, 10, (1000,))  # 10 clases

# X_test = torch.randn(200, 784)
# y_test = torch.randint(0, 10, (200,))

# # Hiperparámetros
# epochs = 20
# batch_size = 32

# # Training loop
# model.train()  # Modo entrenamiento
# history = {'loss': [], 'accuracy': []}

# for epoch in range(epochs):
#     epoch_loss = 0.0
#     correct = 0
#     total = 0
#
#     # Mini-batch training
#     for i in range(0, len(X_train), batch_size):
#         # Obtener batch
#         X_batch = X_train[i:i+batch_size]
#         y_batch = y_train[i:i+batch_size]
#
#         # 1. Limpiar gradientes
#         optimizer.zero_grad()
#
#         # 2. Forward pass
#         outputs = model(X_batch)
#
#         # 3. Calcular pérdida
#         loss = criterion(outputs, y_batch)
#
#         # 4. Backward pass
#         loss.backward()
#
#         # 5. Actualizar parámetros
#         optimizer.step()
#
#         # Estadísticas
#         epoch_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += y_batch.size(0)
#         correct += predicted.eq(y_batch).sum().item()
#
#     # Métricas de época
#     avg_loss = epoch_loss / (len(X_train) // batch_size)
#     accuracy = 100. * correct / total
#     history['loss'].append(avg_loss)
#     history['accuracy'].append(accuracy)
#
#     if (epoch + 1) % 5 == 0:
#         print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {accuracy:.2f}%")

print()

# ============================================
# PASO 6: train() vs eval()
# ============================================
print("\n--- Paso 6: train() vs eval() ---")

# # Modo evaluación
# model.eval()
# print(f"model.training = {model.training}")  # False

# # Evaluar en test set
# with torch.no_grad():  # Desactivar gradientes
#     outputs = model(X_test)
#     _, predicted = outputs.max(1)
#     test_correct = predicted.eq(y_test).sum().item()
#     test_accuracy = 100. * test_correct / len(y_test)
#     print(f"Test Accuracy: {test_accuracy:.2f}%")

# # Volver a modo entrenamiento
# model.train()
# print(f"model.training = {model.training}")  # True

# # Diferencia práctica:
# # - train(): Dropout activo, BatchNorm usa stats del batch
# # - eval(): Dropout desactivado, BatchNorm usa running stats

print()

# ============================================
# PASO 7: Guardar y Cargar Modelo
# ============================================
print("\n--- Paso 7: Guardar y Cargar Modelo ---")

# import os

# # Crear directorio si no existe
# os.makedirs('checkpoints', exist_ok=True)

# # Guardar solo los pesos (recomendado)
# torch.save(model.state_dict(), 'checkpoints/modelo_pesos.pth')
# print("✓ Pesos guardados en 'checkpoints/modelo_pesos.pth'")

# # Cargar pesos
# modelo_nuevo = RedNeuronal(784, 256, 10)  # Crear arquitectura
# modelo_nuevo.load_state_dict(torch.load('checkpoints/modelo_pesos.pth'))
# modelo_nuevo.eval()
# print("✓ Pesos cargados en nuevo modelo")

# # Guardar checkpoint completo (para continuar entrenamiento)
# checkpoint = {
#     'epoch': epochs,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': history['loss'][-1],
#     'accuracy': history['accuracy'][-1]
# }
# torch.save(checkpoint, 'checkpoints/checkpoint.pth')
# print("✓ Checkpoint guardado")

# # Cargar checkpoint
# checkpoint = torch.load('checkpoints/checkpoint.pth')
# modelo_nuevo.load_state_dict(checkpoint['model_state_dict'])
# optimizer_nuevo = optim.Adam(modelo_nuevo.parameters())
# optimizer_nuevo.load_state_dict(checkpoint['optimizer_state_dict'])
# print(f"✓ Checkpoint cargado - Época {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")

# # Limpiar archivos de prueba
# # os.remove('checkpoints/modelo_pesos.pth')
# # os.remove('checkpoints/checkpoint.pth')

print()

# ============================================
# VERIFICACIÓN FINAL
# ============================================
print("\n" + "=" * 60)
print("✅ Ejercicio completado!")
print("=" * 60)
print(
    """
Resumen de lo aprendido:
1. Crear clases que heredan de nn.Module
2. Definir __init__ con capas y forward con flujo
3. Inspeccionar parámetros y estructura del modelo
4. Configurar criterion (loss) y optimizer
5. Implementar training loop: zero_grad → forward → loss → backward → step
6. Usar train() para entrenamiento, eval() para inferencia
7. Guardar/cargar modelos con state_dict()
"""
)
