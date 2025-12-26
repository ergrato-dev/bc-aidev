"""
Ejercicio 03: MLP con NumPy
===========================
Implementar forward pass de un MLP.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================
# PASO 1: Funciones de Activación
# ============================================
print('--- Paso 1: Activaciones ---')

# Descomenta las siguientes líneas:
# def sigmoid(z):
#     return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
# 
# def relu(z):
#     return np.maximum(0, z)
# 
# def softmax(z):
#     exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
#     return exp_z / np.sum(exp_z, axis=0, keepdims=True)
# 
# print("Activaciones definidas ✓")

print()


# ============================================
# PASO 2: Inicialización de Pesos
# ============================================
print('--- Paso 2: Inicialización ---')

# Descomenta las siguientes líneas:
# def initialize_weights(layer_sizes):
#     """
#     Inicializa pesos con Xavier/He initialization.
#     
#     Args:
#         layer_sizes: lista [input, hidden1, ..., output]
#     
#     Returns:
#         parameters: dict con W1, b1, W2, b2, etc.
#     """
#     parameters = {}
#     L = len(layer_sizes) - 1  # número de capas
#     
#     for l in range(1, L + 1):
#         n_in = layer_sizes[l - 1]
#         n_out = layer_sizes[l]
#         
#         # He initialization para ReLU
#         parameters[f'W{l}'] = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
#         parameters[f'b{l}'] = np.zeros((n_out, 1))
#         
#         print(f"W{l}: {parameters[f'W{l}'].shape}, b{l}: {parameters[f'b{l}'].shape}")
#     
#     return parameters
# 
# # Probar: red 3 -> 4 -> 2
# params = initialize_weights([3, 4, 2])

print()


# ============================================
# PASO 3: Forward Pass de una Capa
# ============================================
print('--- Paso 3: Forward de una Capa ---')

# Descomenta las siguientes líneas:
# def linear_forward(A_prev, W, b):
#     """
#     Calcula Z = W·A + b
#     
#     Args:
#         A_prev: activación de capa anterior (n_prev, m)
#         W: pesos (n_current, n_prev)
#         b: bias (n_current, 1)
#     
#     Returns:
#         Z: pre-activación (n_current, m)
#     """
#     Z = np.dot(W, A_prev) + b
#     return Z
# 
# def activation_forward(Z, activation):
#     """Aplica función de activación."""
#     if activation == 'relu':
#         return relu(Z)
#     elif activation == 'sigmoid':
#         return sigmoid(Z)
#     elif activation == 'softmax':
#         return softmax(Z)
#     else:
#         return Z  # linear
# 
# # Probar
# A_test = np.random.randn(3, 5)  # 3 features, 5 samples
# W_test = np.random.randn(4, 3)  # 4 neuronas
# b_test = np.zeros((4, 1))
# 
# Z = linear_forward(A_test, W_test, b_test)
# A = activation_forward(Z, 'relu')
# 
# print(f"Input shape: {A_test.shape}")
# print(f"Z shape: {Z.shape}")
# print(f"A shape (after ReLU): {A.shape}")

print()


# ============================================
# PASO 4: Forward Pass Completo
# ============================================
print('--- Paso 4: Forward Completo ---')

# Descomenta las siguientes líneas:
# def forward_propagation(X, parameters, activations):
#     """
#     Forward pass completo a través de la red.
#     
#     Args:
#         X: input data (n_features, m_samples)
#         parameters: dict con W1, b1, W2, b2, etc.
#         activations: lista de funciones de activación por capa
#     
#     Returns:
#         A: output final
#         cache: valores intermedios para backprop
#     """
#     cache = {'A0': X}
#     A = X
#     L = len(activations)
#     
#     for l in range(1, L + 1):
#         A_prev = A
#         W = parameters[f'W{l}']
#         b = parameters[f'b{l}']
#         
#         Z = linear_forward(A_prev, W, b)
#         A = activation_forward(Z, activations[l - 1])
#         
#         cache[f'Z{l}'] = Z
#         cache[f'A{l}'] = A
#     
#     return A, cache
# 
# print("forward_propagation definida ✓")

print()


# ============================================
# PASO 5: Probar con Red Simple
# ============================================
print('--- Paso 5: Red Simple ---')

# Descomenta las siguientes líneas:
# # Arquitectura: 2 -> 4 -> 1 (para clasificación binaria)
# layer_sizes = [2, 4, 1]
# activations = ['relu', 'sigmoid']
# 
# params = initialize_weights(layer_sizes)
# 
# # Input de prueba (2 features, 4 samples)
# X = np.array([[0, 0, 1, 1],
#               [0, 1, 0, 1]])
# 
# # Forward pass
# output, cache = forward_propagation(X, params, activations)
# 
# print(f"Input X:\n{X}")
# print(f"Output shape: {output.shape}")
# print(f"Output:\n{output}")

print()


# ============================================
# PASO 6: Red Multicapa
# ============================================
print('--- Paso 6: Red Multicapa ---')

# Descomenta las siguientes líneas:
# # Arquitectura más profunda: 5 -> 10 -> 8 -> 4 -> 2
# layer_sizes_deep = [5, 10, 8, 4, 2]
# activations_deep = ['relu', 'relu', 'relu', 'softmax']
# 
# params_deep = initialize_weights(layer_sizes_deep)
# 
# # Input de prueba
# X_deep = np.random.randn(5, 10)  # 5 features, 10 samples
# 
# output_deep, cache_deep = forward_propagation(X_deep, params_deep, activations_deep)
# 
# print(f"\nInput shape: {X_deep.shape}")
# print(f"Output shape: {output_deep.shape}")
# print(f"\nSoftmax outputs (suman 1 por columna):")
# print(output_deep[:, :3])  # Primeras 3 muestras
# print(f"Suma por columna: {output_deep.sum(axis=0)[:3]}")

print()


# ============================================
# PASO 7: Visualizar Cache
# ============================================
print('--- Paso 7: Cache ---')

# Descomenta las siguientes líneas:
# print("Contenido del cache (para backprop):")
# for key in cache:
#     print(f"  {key}: shape {cache[key].shape}")

print()


# ============================================
# PASO 8: Clase MLP
# ============================================
print('--- Paso 8: Clase MLP ---')

# Descomenta las siguientes líneas:
# class MLP:
#     """Multi-Layer Perceptron (solo forward pass)."""
#     
#     def __init__(self, layer_sizes, activations):
#         self.layer_sizes = layer_sizes
#         self.activations = activations
#         self.parameters = initialize_weights(layer_sizes)
#         self.cache = None
#     
#     def forward(self, X):
#         """Forward pass."""
#         output, self.cache = forward_propagation(X, self.parameters, self.activations)
#         return output
#     
#     def predict(self, X):
#         """Predice clases."""
#         output = self.forward(X)
#         if output.shape[0] == 1:
#             return (output > 0.5).astype(int)
#         else:
#             return np.argmax(output, axis=0)
# 
# # Usar
# mlp = MLP([2, 4, 1], ['relu', 'sigmoid'])
# predictions = mlp.predict(X)
# print(f"Predicciones: {predictions.flatten()}")

print()


# ============================================
# PASO 9: Resumen
# ============================================
print('--- Paso 9: Resumen ---')

# Descomenta las siguientes líneas:
# print("""
# RESUMEN: FORWARD PASS EN MLP
# ============================
# 
# 1. Inicializar pesos (He/Xavier)
# 2. Para cada capa l:
#    - Z = W·A_prev + b  (combinación lineal)
#    - A = activation(Z)  (no-linealidad)
# 3. Guardar cache para backprop
# 
# Dimensiones:
# - X: (n_features, n_samples)
# - W_l: (n_l, n_{l-1})
# - b_l: (n_l, 1)
# - Z_l, A_l: (n_l, n_samples)
# 
# SIGUIENTE: Implementar backpropagation en el proyecto
# """)

print()
print('=== Ejercicio completado ===')
