"""
Ejercicio 02: Funciones de Activación
=====================================
Implementar y visualizar funciones de activación.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================
# PASO 1: Sigmoid
# ============================================
print('--- Paso 1: Sigmoid ---')

# Descomenta las siguientes líneas:
# def sigmoid(z):
#     """Función sigmoid: σ(z) = 1 / (1 + e^(-z))"""
#     return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
# 
# def sigmoid_derivative(z):
#     """Derivada: σ'(z) = σ(z) * (1 - σ(z))"""
#     s = sigmoid(z)
#     return s * (1 - s)
# 
# # Probar
# z = np.array([-2, -1, 0, 1, 2])
# print(f"z = {z}")
# print(f"sigmoid(z) = {sigmoid(z)}")
# print(f"sigmoid'(z) = {sigmoid_derivative(z)}")

print()


# ============================================
# PASO 2: Tanh
# ============================================
print('--- Paso 2: Tanh ---')

# Descomenta las siguientes líneas:
# def tanh(z):
#     """Función tanh: (e^z - e^(-z)) / (e^z + e^(-z))"""
#     return np.tanh(z)
# 
# def tanh_derivative(z):
#     """Derivada: tanh'(z) = 1 - tanh²(z)"""
#     return 1 - np.tanh(z) ** 2
# 
# print(f"tanh(z) = {tanh(z)}")
# print(f"tanh'(z) = {tanh_derivative(z)}")

print()


# ============================================
# PASO 3: ReLU
# ============================================
print('--- Paso 3: ReLU ---')

# Descomenta las siguientes líneas:
# def relu(z):
#     """ReLU: max(0, z)"""
#     return np.maximum(0, z)
# 
# def relu_derivative(z):
#     """Derivada: 1 si z > 0, 0 si z <= 0"""
#     return (z > 0).astype(float)
# 
# print(f"relu(z) = {relu(z)}")
# print(f"relu'(z) = {relu_derivative(z)}")

print()


# ============================================
# PASO 4: Leaky ReLU
# ============================================
print('--- Paso 4: Leaky ReLU ---')

# Descomenta las siguientes líneas:
# def leaky_relu(z, alpha=0.01):
#     """Leaky ReLU: z si z > 0, alpha*z si z <= 0"""
#     return np.where(z > 0, z, alpha * z)
# 
# def leaky_relu_derivative(z, alpha=0.01):
#     """Derivada: 1 si z > 0, alpha si z <= 0"""
#     return np.where(z > 0, 1, alpha)
# 
# print(f"leaky_relu(z) = {leaky_relu(z)}")
# print(f"leaky_relu'(z) = {leaky_relu_derivative(z)}")

print()


# ============================================
# PASO 5: Softmax
# ============================================
print('--- Paso 5: Softmax ---')

# Descomenta las siguientes líneas:
# def softmax(z):
#     """Softmax: e^zi / Σe^zj (probabilidades que suman 1)"""
#     exp_z = np.exp(z - np.max(z))  # Restar max para estabilidad
#     return exp_z / np.sum(exp_z)
# 
# logits = np.array([2.0, 1.0, 0.1])
# probs = softmax(logits)
# print(f"Logits: {logits}")
# print(f"Softmax: {probs}")
# print(f"Suma: {np.sum(probs):.4f}")

print()


# ============================================
# PASO 6: Visualizar Funciones
# ============================================
print('--- Paso 6: Visualizar Funciones ---')

# Descomenta las siguientes líneas:
# z = np.linspace(-5, 5, 200)
# 
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# 
# # Sigmoid
# axes[0, 0].plot(z, sigmoid(z), 'b-', linewidth=2)
# axes[0, 0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
# axes[0, 0].axvline(0, color='gray', linestyle='--', alpha=0.5)
# axes[0, 0].set_title('Sigmoid', fontsize=14)
# axes[0, 0].set_ylim(-0.1, 1.1)
# axes[0, 0].grid(True, alpha=0.3)
# 
# # Tanh
# axes[0, 1].plot(z, tanh(z), 'purple', linewidth=2)
# axes[0, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
# axes[0, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
# axes[0, 1].set_title('Tanh', fontsize=14)
# axes[0, 1].set_ylim(-1.2, 1.2)
# axes[0, 1].grid(True, alpha=0.3)
# 
# # ReLU
# axes[0, 2].plot(z, relu(z), 'g-', linewidth=2)
# axes[0, 2].axhline(0, color='gray', linestyle='--', alpha=0.5)
# axes[0, 2].axvline(0, color='gray', linestyle='--', alpha=0.5)
# axes[0, 2].set_title('ReLU', fontsize=14)
# axes[0, 2].set_ylim(-1, 5)
# axes[0, 2].grid(True, alpha=0.3)
# 
# # Derivadas
# axes[1, 0].plot(z, sigmoid_derivative(z), 'b-', linewidth=2)
# axes[1, 0].set_title("Sigmoid' (máx=0.25)", fontsize=14)
# axes[1, 0].grid(True, alpha=0.3)
# 
# axes[1, 1].plot(z, tanh_derivative(z), 'purple', linewidth=2)
# axes[1, 1].set_title("Tanh' (máx=1.0)", fontsize=14)
# axes[1, 1].grid(True, alpha=0.3)
# 
# axes[1, 2].plot(z, relu_derivative(z), 'g-', linewidth=2)
# axes[1, 2].set_title("ReLU' (0 o 1)", fontsize=14)
# axes[1, 2].set_ylim(-0.1, 1.2)
# axes[1, 2].grid(True, alpha=0.3)
# 
# plt.tight_layout()
# plt.show()

print()


# ============================================
# PASO 7: Comparar Derivadas (Vanishing Gradient)
# ============================================
print('--- Paso 7: Vanishing Gradient ---')

# Descomenta las siguientes líneas:
# # El problema: derivadas pequeñas se multiplican en capas profundas
# n_layers = [1, 5, 10, 20]
# 
# print("Gradiente máximo después de N capas:")
# print("-" * 40)
# for n in n_layers:
#     # Sigmoid: derivada máxima = 0.25
#     sigmoid_grad = 0.25 ** n
#     # Tanh: derivada máxima = 1.0
#     tanh_grad = 1.0 ** n
#     # ReLU: derivada = 1 (si activa)
#     relu_grad = 1.0 ** n
#     
#     print(f"{n:2d} capas: Sigmoid={sigmoid_grad:.2e}, Tanh={tanh_grad:.2e}, ReLU={relu_grad:.2e}")
# 
# print("\n⚠️ Sigmoid causa vanishing gradient en redes profundas!")
# print("✅ ReLU mantiene el gradiente constante")

print()


# ============================================
# PASO 8: Dying ReLU
# ============================================
print('--- Paso 8: Dying ReLU ---')

# Descomenta las siguientes líneas:
# # Problema: si z < 0 siempre, la neurona "muere"
# z_negative = np.array([-5, -3, -1, -0.5])
# 
# print("Si z siempre es negativo:")
# print(f"z = {z_negative}")
# print(f"ReLU(z) = {relu(z_negative)}")      # Todo ceros
# print(f"ReLU'(z) = {relu_derivative(z_negative)}")  # Todo ceros - ¡no aprende!
# 
# print("\nSolución: Leaky ReLU")
# print(f"LeakyReLU(z) = {leaky_relu(z_negative, 0.1)}")
# print(f"LeakyReLU'(z) = {leaky_relu_derivative(z_negative, 0.1)}")

print()


# ============================================
# PASO 9: Resumen
# ============================================
print('--- Paso 9: Resumen ---')

# Descomenta las siguientes líneas:
# print("""
# RESUMEN DE FUNCIONES DE ACTIVACIÓN
# ===================================
# 
# | Función    | Rango      | Derivada Máx | Uso Principal          |
# |------------|------------|--------------|------------------------|
# | Sigmoid    | (0, 1)     | 0.25         | Salida binaria         |
# | Tanh       | (-1, 1)    | 1.0          | Capas ocultas/RNN      |
# | ReLU       | [0, ∞)     | 1.0          | DEFAULT capas ocultas  |
# | Leaky ReLU | (-∞, ∞)    | 1.0          | Evitar dying ReLU      |
# | Softmax    | (0, 1)     | -            | Salida multiclase      |
# 
# RECOMENDACIONES:
# - Capas ocultas: ReLU (simple y efectivo)
# - Salida binaria: Sigmoid
# - Salida multiclase: Softmax
# - Si ReLU falla: Leaky ReLU
# """)

print()
print('=== Ejercicio completado ===')
