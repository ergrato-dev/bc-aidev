"""
Ejercicio 01: Perceptrón Simple
===============================
Implementar un perceptrón desde cero.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================
# PASO 1: Función de Activación Step
# ============================================
print('--- Paso 1: Función Step ---')

# Descomenta las siguientes líneas:
# def step(z):
#     """Función escalón: 1 si z >= 0, 0 si no."""
#     return np.where(z >= 0, 1, 0)
# 
# # Probar
# z_test = np.array([-2, -1, 0, 1, 2])
# print(f"z = {z_test}")
# print(f"step(z) = {step(z_test)}")

print()


# ============================================
# PASO 2: Clase Perceptron
# ============================================
print('--- Paso 2: Clase Perceptron ---')

# Descomenta las siguientes líneas:
# class Perceptron:
#     """Implementación del perceptrón de Rosenblatt."""
#     
#     def __init__(self, n_features, learning_rate=0.1):
#         self.weights = np.zeros(n_features)
#         self.bias = 0.0
#         self.lr = learning_rate
#         self.errors_ = []
#     
#     def predict(self, X):
#         """Predice la clase para X."""
#         z = np.dot(X, self.weights) + self.bias
#         return step(z)
#     
#     def fit(self, X, y, epochs=100):
#         """Entrena el perceptrón."""
#         for epoch in range(epochs):
#             errors = 0
#             for xi, yi in zip(X, y):
#                 pred = self.predict(xi.reshape(1, -1))[0]
#                 error = yi - pred
#                 if error != 0:
#                     self.weights += self.lr * error * xi
#                     self.bias += self.lr * error
#                     errors += 1
#             self.errors_.append(errors)
#             if errors == 0:
#                 print(f"Convergió en época {epoch + 1}")
#                 break
#         return self
# 
# print("Clase Perceptron definida ✓")

print()


# ============================================
# PASO 3: Compuerta AND
# ============================================
print('--- Paso 3: Compuerta AND ---')

# Descomenta las siguientes líneas:
# X = np.array([
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ])
# y_and = np.array([0, 0, 0, 1])
# 
# p_and = Perceptron(n_features=2, learning_rate=0.1)
# p_and.fit(X, y_and, epochs=20)
# 
# print(f"Pesos: {p_and.weights}")
# print(f"Bias: {p_and.bias}")
# print(f"Predicciones: {p_and.predict(X)}")
# print(f"Esperado:     {y_and}")

print()


# ============================================
# PASO 4: Compuerta OR
# ============================================
print('--- Paso 4: Compuerta OR ---')

# Descomenta las siguientes líneas:
# y_or = np.array([0, 1, 1, 1])
# 
# p_or = Perceptron(n_features=2, learning_rate=0.1)
# p_or.fit(X, y_or, epochs=20)
# 
# print(f"Pesos: {p_or.weights}")
# print(f"Bias: {p_or.bias}")
# print(f"Predicciones: {p_or.predict(X)}")
# print(f"Esperado:     {y_or}")

print()


# ============================================
# PASO 5: Compuerta NAND
# ============================================
print('--- Paso 5: Compuerta NAND ---')

# Descomenta las siguientes líneas:
# y_nand = np.array([1, 1, 1, 0])
# 
# p_nand = Perceptron(n_features=2, learning_rate=0.1)
# p_nand.fit(X, y_nand, epochs=20)
# 
# print(f"Pesos: {p_nand.weights}")
# print(f"Bias: {p_nand.bias}")
# print(f"Predicciones: {p_nand.predict(X)}")
# print(f"Esperado:     {y_nand}")

print()


# ============================================
# PASO 6: XOR (Falla)
# ============================================
print('--- Paso 6: XOR (Esperado: Falla) ---')

# Descomenta las siguientes líneas:
# y_xor = np.array([0, 1, 1, 0])
# 
# p_xor = Perceptron(n_features=2, learning_rate=0.1)
# p_xor.fit(X, y_xor, epochs=100)
# 
# print(f"Predicciones: {p_xor.predict(X)}")
# print(f"Esperado:     {y_xor}")
# accuracy = np.mean(p_xor.predict(X) == y_xor)
# print(f"Accuracy: {accuracy}")  # < 1.0 porque XOR no es linealmente separable

print()


# ============================================
# PASO 7: Visualizar Frontera de Decisión
# ============================================
print('--- Paso 7: Visualización ---')

# Descomenta las siguientes líneas:
# def plot_decision_boundary(perceptron, X, y, title):
#     fig, ax = plt.subplots(figsize=(6, 5))
#     
#     # Mesh
#     x_min, x_max = -0.5, 1.5
#     y_min, y_max = -0.5, 1.5
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
#                          np.linspace(y_min, y_max, 100))
#     Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     
#     ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
#     ax.scatter(X[y==0, 0], X[y==0, 1], c='red', s=100, marker='o', label='0')
#     ax.scatter(X[y==1, 0], X[y==1, 1], c='blue', s=100, marker='s', label='1')
#     
#     ax.set_xlabel('x₁')
#     ax.set_ylabel('x₂')
#     ax.set_title(title)
#     ax.legend()
#     plt.tight_layout()
#     plt.show()
# 
# # Visualizar AND
# plot_decision_boundary(p_and, X, y_and, "Perceptrón - AND")
# 
# # Visualizar OR
# plot_decision_boundary(p_or, X, y_or, "Perceptrón - OR")
# 
# # Visualizar XOR (frontera incorrecta)
# plot_decision_boundary(p_xor, X, y_xor, "Perceptrón - XOR (Falla)")

print()


# ============================================
# PASO 8: Curva de Errores
# ============================================
print('--- Paso 8: Curva de Errores ---')

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# 
# # AND converge rápido
# axes[0].plot(p_and.errors_, 'b-o')
# axes[0].set_xlabel('Época')
# axes[0].set_ylabel('Errores')
# axes[0].set_title('AND - Convergencia')
# 
# # XOR no converge
# axes[1].plot(p_xor.errors_, 'r-o')
# axes[1].set_xlabel('Época')
# axes[1].set_ylabel('Errores')
# axes[1].set_title('XOR - No Converge')
# 
# plt.tight_layout()
# plt.show()

print()
print('=== Ejercicio completado ===')
