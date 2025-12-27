"""
Ejercicio 01: Tensores en PyTorch
Bootcamp IA: Zero to Hero | Semana 21

Descomenta cada sección según avances en el ejercicio.
"""

import numpy as np
import torch

print("=" * 60)
print("EJERCICIO 01: TENSORES EN PYTORCH")
print("=" * 60)

# ============================================
# PASO 1: Creación de Tensores
# ============================================
print("\n--- Paso 1: Creación de Tensores ---")

# Desde lista Python
# t1 = torch.tensor([1, 2, 3, 4])
# print(f"Desde lista: {t1}")
# print(f"  dtype: {t1.dtype}")

# Especificando dtype
# t2 = torch.tensor([1, 2, 3], dtype=torch.float32)
# print(f"Con dtype float32: {t2}")

# Tensores de ceros y unos
# zeros = torch.zeros(3, 4)
# print(f"Zeros (3x4):\n{zeros}")

# ones = torch.ones(2, 3)
# print(f"Ones (2x3):\n{ones}")

# Tensores aleatorios
# rand = torch.rand(3, 3)  # Uniforme [0, 1)
# print(f"Random uniforme:\n{rand}")

# randn = torch.randn(3, 3)  # Normal (0, 1)
# print(f"Random normal:\n{randn}")

# Secuencias
# arange = torch.arange(0, 10, 2)
# print(f"Arange(0, 10, 2): {arange}")

# linspace = torch.linspace(0, 1, 5)
# print(f"Linspace(0, 1, 5): {linspace}")

print()

# ============================================
# PASO 2: Atributos de Tensores
# ============================================
print("\n--- Paso 2: Atributos de Tensores ---")

# t = torch.rand(2, 3, 4)
# print(f"Tensor shape: {t.shape}")
# print(f"Tensor dtype: {t.dtype}")
# print(f"Tensor device: {t.device}")
# print(f"Dimensiones: {t.dim()}")
# print(f"Total elementos: {t.numel()}")

# Cambiar dtype
# t_int = t.int()
# print(f"Convertido a int: {t_int.dtype}")

# t_double = t.double()
# print(f"Convertido a double: {t_double.dtype}")

print()

# ============================================
# PASO 3: Operaciones Matemáticas
# ============================================
print("\n--- Paso 3: Operaciones Matemáticas ---")

# a = torch.tensor([1.0, 2.0, 3.0])
# b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise
# print(f"a + b = {a + b}")
# print(f"a - b = {a - b}")
# print(f"a * b = {a * b}")
# print(f"a / b = {a / b}")
# print(f"a ** 2 = {a ** 2}")

# Reducciones
# print(f"Suma: {a.sum()}")
# print(f"Media: {a.mean()}")
# print(f"Max: {a.max()}")
# print(f"Min: {a.min()}")

# Producto punto
# dot = torch.dot(a, b)
# print(f"Producto punto: {dot}")

# Multiplicación de matrices
# A = torch.rand(2, 3)
# B = torch.rand(3, 4)
# C = A @ B  # o torch.matmul(A, B)
# print(f"A (2x3) @ B (3x4) = C {C.shape}")

print()

# ============================================
# PASO 4: Indexing y Slicing
# ============================================
print("\n--- Paso 4: Indexing y Slicing ---")

# t = torch.arange(12).reshape(3, 4)
# print(f"Tensor original:\n{t}")

# Indexing básico
# print(f"t[0] (primera fila): {t[0]}")
# print(f"t[1, 2] (elemento): {t[1, 2]}")
# print(f"t[-1] (última fila): {t[-1]}")

# Slicing
# print(f"t[:2] (primeras 2 filas):\n{t[:2]}")
# print(f"t[:, 1] (segunda columna): {t[:, 1]}")
# print(f"t[1:, 2:] (submatriz):\n{t[1:, 2:]}")

# Boolean indexing
# mask = t > 5
# print(f"Máscara (t > 5):\n{mask}")
# print(f"Elementos > 5: {t[mask]}")

print()

# ============================================
# PASO 5: Reshape y Manipulación
# ============================================
print("\n--- Paso 5: Reshape y Manipulación ---")

# t = torch.arange(12)
# print(f"Original: {t}")

# Reshape
# reshaped = t.reshape(3, 4)
# print(f"Reshape (3, 4):\n{reshaped}")

# Con -1 (inferir dimensión)
# auto = t.reshape(-1, 3)
# print(f"Reshape (-1, 3):\n{auto}")

# Flatten
# flat = reshaped.flatten()
# print(f"Flatten: {flat}")

# Squeeze (elimina dims de tamaño 1)
# t = torch.rand(1, 3, 1, 4)
# print(f"Original shape: {t.shape}")
# squeezed = t.squeeze()
# print(f"Squeezed shape: {squeezed.shape}")

# Unsqueeze (añade dim)
# t = torch.rand(3, 4)
# unsqueezed = t.unsqueeze(0)
# print(f"Unsqueezed shape: {unsqueezed.shape}")

# Concatenar
# a = torch.rand(2, 3)
# b = torch.rand(2, 3)
# concat_0 = torch.cat([a, b], dim=0)
# print(f"Cat dim=0: {concat_0.shape}")
# concat_1 = torch.cat([a, b], dim=1)
# print(f"Cat dim=1: {concat_1.shape}")

print()

# ============================================
# PASO 6: NumPy ↔ PyTorch
# ============================================
print("\n--- Paso 6: NumPy <-> PyTorch ---")

# NumPy → PyTorch
# np_arr = np.array([1.0, 2.0, 3.0])
# tensor_from_np = torch.from_numpy(np_arr)
# print(f"NumPy array: {np_arr}")
# print(f"Tensor from NumPy: {tensor_from_np}")

# ⚠️ Comparten memoria!
# np_arr[0] = 100
# print(f"Después de modificar NumPy:")
# print(f"  NumPy: {np_arr}")
# print(f"  Tensor: {tensor_from_np}")

# PyTorch → NumPy
# tensor = torch.tensor([4.0, 5.0, 6.0])
# np_from_tensor = tensor.numpy()
# print(f"Tensor: {tensor}")
# print(f"NumPy from tensor: {np_from_tensor}")

# Copia sin compartir memoria
# np_arr = np.array([1.0, 2.0, 3.0])
# tensor_copy = torch.tensor(np_arr)  # Usa torch.tensor() para copiar
# np_arr[0] = 999
# print(f"Con copia - NumPy modificado: {np_arr}")
# print(f"Con copia - Tensor sin modificar: {tensor_copy}")

print()

# ============================================
# PASO 7: Dispositivos (CPU/GPU)
# ============================================
print("\n--- Paso 7: Dispositivos (CPU/GPU) ---")

# Detectar dispositivo disponible
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Dispositivo disponible: {device}")

# Información de CUDA si está disponible
# if torch.cuda.is_available():
#     print(f"GPU: {torch.cuda.get_device_name(0)}")
#     print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Crear tensor en dispositivo específico
# t_device = torch.rand(3, 3, device=device)
# print(f"Tensor en {device}: {t_device.device}")

# Mover tensor entre dispositivos
# t_cpu = torch.rand(3, 3)
# print(f"Original en: {t_cpu.device}")

# t_to_device = t_cpu.to(device)
# print(f"Movido a: {t_to_device.device}")

# Volver a CPU
# t_back_cpu = t_to_device.cpu()
# print(f"De vuelta en: {t_back_cpu.device}")

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
1. Crear tensores con diferentes métodos
2. Entender atributos: shape, dtype, device
3. Operaciones matemáticas element-wise y matriciales
4. Indexing y slicing como NumPy
5. Reshape, squeeze, unsqueeze, cat
6. Conversión NumPy <-> PyTorch
7. Mover tensores entre CPU y GPU
"""
)
