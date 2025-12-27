"""
Ejercicio 02: CNN en PyTorch
============================

Construir CNNs usando PyTorch nn.Module.

Instrucciones:
- Lee cada paso en el README.md
- Descomenta las secciones correspondientes
- Ejecuta y observa las dimensiones
"""

import torch
import torch.nn as nn

# ============================================
# PASO 1: Capa Convolucional Básica
# ============================================
print("--- Paso 1: Capa Convolucional Básica ---")

# Crear una capa convolucional
# - in_channels: canales de entrada
# - out_channels: número de filtros
# - kernel_size: tamaño del filtro
# - stride: paso del deslizamiento
# - padding: relleno para mantener tamaño

# Descomenta las siguientes líneas:
# conv = nn.Conv2d(
#     in_channels=1,
#     out_channels=32,
#     kernel_size=3,
#     stride=1,
#     padding=1
# )
#
# # Entrada: (batch, channels, height, width)
# x = torch.randn(4, 1, 28, 28)
# output = conv(x)
#
# print(f"Input shape: {x.shape}")
# print(f"Output shape: {output.shape}")
# print(f"Kernel shape: {conv.weight.shape}")
# print(f"Bias shape: {conv.bias.shape}")
# print(f"Parámetros conv: {conv.weight.numel() + conv.bias.numel()}")

print()

# ============================================
# PASO 2: Pooling y Activaciones
# ============================================
print("--- Paso 2: Pooling y Activaciones ---")

# Bloque típico: Conv -> ReLU -> Pool
# El pooling reduce las dimensiones espaciales

# Descomenta las siguientes líneas:
# block = nn.Sequential(
#     nn.Conv2d(1, 32, kernel_size=3, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2)
# )
#
# x = torch.randn(4, 1, 28, 28)
# output = block(x)
#
# print(f"Input: {x.shape}")
# print(f"Output: {output.shape}")
# print(f"Reducción espacial: 28 -> 14 (MaxPool 2×2)")

print()

# ============================================
# PASO 3: Calcular Dimensiones de Flatten
# ============================================
print("--- Paso 3: Calcular Dimensiones de Flatten ---")

# Función para calcular el tamaño después de flatten
# Útil para definir la primera capa fully connected

# Descomenta las siguientes líneas:
# def calculate_flatten_size(input_shape, conv_layers):
#     """Calcula el tamaño después de flatten."""
#     x = torch.randn(1, *input_shape)
#     with torch.no_grad():
#         output = conv_layers(x)
#     return output.numel()
#
# # Ejemplo con dos bloques conv
# features = nn.Sequential(
#     # Bloque 1: 1 -> 32, 28 -> 14
#     nn.Conv2d(1, 32, 3, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2),
#     # Bloque 2: 32 -> 64, 14 -> 7
#     nn.Conv2d(32, 64, 3, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2)
# )
#
# flat_size = calculate_flatten_size((1, 28, 28), features)
# print(f"Tamaño flatten: {flat_size}")
# print(f"Cálculo: 64 canales × 7 × 7 = {64 * 7 * 7}")

print()

# ============================================
# PASO 4: CNN Completa como Clase
# ============================================
print("--- Paso 4: CNN Completa como Clase ---")

# CNN completa heredando de nn.Module
# Separamos features (extractor) y classifier

# Descomenta las siguientes líneas:
# class SimpleCNN(nn.Module):
#     """CNN simple para clasificación de imágenes 28×28."""
#
#     def __init__(self, num_classes=10):
#         super().__init__()
#
#         # Extractor de características
#         self.features = nn.Sequential(
#             # Bloque 1: 1 -> 32 canales, 28 -> 14
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             # Bloque 2: 32 -> 64 canales, 14 -> 7
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#
#         # Clasificador
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 7 * 7, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x
#
# # Crear modelo y verificar
# model = SimpleCNN(num_classes=10)
# x = torch.randn(4, 1, 28, 28)
# output = model(x)
# print(f"Input: {x.shape}")
# print(f"Output: {output.shape}")

print()

# ============================================
# PASO 5: Inspeccionar el Modelo
# ============================================
print("--- Paso 5: Inspeccionar el Modelo ---")

# Ver arquitectura y contar parámetros

# Descomenta las siguientes líneas:
# def count_parameters(model):
#     """Cuenta parámetros entrenables."""
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# model = SimpleCNN(num_classes=10)
#
# # Ver arquitectura
# print("Arquitectura:")
# print(model)
# print()
#
# # Contar parámetros
# total = count_parameters(model)
# print(f"Total parámetros: {total:,}")
# print()
#
# # Parámetros por capa
# print("Parámetros por capa:")
# for name, param in model.named_parameters():
#     print(f"  {name}: {param.shape} = {param.numel():,}")

print()

# ============================================
# PASO 6: Forward Pass y Dimensiones
# ============================================
print("--- Paso 6: Forward Pass y Dimensiones ---")

# Trazar dimensiones en cada capa

# Descomenta las siguientes líneas:
# def trace_dimensions(model, input_tensor):
#     """Muestra dimensiones en cada capa."""
#     x = input_tensor.clone()
#     print(f"Input: {x.shape}")
#
#     # Features
#     for i, layer in enumerate(model.features):
#         x = layer(x)
#         layer_name = layer.__class__.__name__
#         print(f"  features[{i}] {layer_name}: {x.shape}")
#
#     # Classifier
#     for i, layer in enumerate(model.classifier):
#         x = layer(x)
#         layer_name = layer.__class__.__name__
#         print(f"  classifier[{i}] {layer_name}: {x.shape}")
#
#     return x
#
# model = SimpleCNN(num_classes=10)
# x = torch.randn(1, 1, 28, 28)
# print("\nFlujo de datos:")
# output = trace_dimensions(model, x)

print()

# ============================================
# PASO 7: Batch Normalization
# ============================================
print("--- Paso 7: Batch Normalization ---")

# BatchNorm acelera entrenamiento y mejora estabilidad
# Se coloca después de Conv, antes de ReLU

# Descomenta las siguientes líneas:
# class CNNWithBatchNorm(nn.Module):
#     """CNN con Batch Normalization."""
#
#     def __init__(self, num_classes=10):
#         super().__init__()
#
#         self.features = nn.Sequential(
#             # Bloque 1 con BatchNorm
#             nn.Conv2d(1, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             # Bloque 2 con BatchNorm
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             # Bloque 3 con BatchNorm
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 3 * 3, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x
#
# # Comparar modelos
# model_simple = SimpleCNN(num_classes=10)
# model_bn = CNNWithBatchNorm(num_classes=10)
#
# print(f"SimpleCNN parámetros: {count_parameters(model_simple):,}")
# print(f"CNN+BatchNorm parámetros: {count_parameters(model_bn):,}")
#
# # BatchNorm agrega parámetros (gamma, beta por canal)
# x = torch.randn(4, 1, 28, 28)
# output = model_bn(x)
# print(f"\nOutput shape: {output.shape}")

print()

# ============================================
# RESUMEN
# ============================================
print("=" * 50)
print("RESUMEN")
print("=" * 50)
print(
    """
Componentes de una CNN en PyTorch:

1. nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride)
   - Extrae características espaciales

2. nn.MaxPool2d(kernel_size, stride)
   - Reduce dimensiones espaciales

3. nn.BatchNorm2d(num_features)
   - Normaliza activaciones, acelera entrenamiento

4. nn.ReLU()
   - No-linealidad

5. nn.Flatten()
   - Convierte tensor 3D a vector 1D

6. nn.Linear(in_features, out_features)
   - Clasificación final

Patrón típico:
  [Conv -> BatchNorm -> ReLU -> Pool] × N -> Flatten -> FC
"""
)
