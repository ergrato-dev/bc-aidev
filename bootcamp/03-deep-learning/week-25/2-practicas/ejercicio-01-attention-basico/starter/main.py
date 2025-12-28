"""
Ejercicio 01: Scaled Dot-Product Attention
==========================================

Implementa el mecanismo de atención básico paso a paso.
"""

# ============================================
# PASO 1: Importaciones
# ============================================
print("--- Paso 1: Importaciones ---")

# Descomenta las siguientes líneas:
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

print("Importaciones completadas")
print()

# ============================================
# PASO 2: Crear datos de ejemplo
# ============================================
print("--- Paso 2: Crear Q, K, V de ejemplo ---")

# Descomenta las siguientes líneas:
# torch.manual_seed(42)
#
# # Dimensiones
# batch_size = 1
# seq_len = 4  # 4 tokens
# d_k = 8      # dimensión de embedding
#
# # Crear Q, K, V aleatorios
# Q = torch.randn(batch_size, seq_len, d_k)
# K = torch.randn(batch_size, seq_len, d_k)
# V = torch.randn(batch_size, seq_len, d_k)
#
# print(f"Q shape: {Q.shape}")
# print(f"K shape: {K.shape}")
# print(f"V shape: {V.shape}")

print()

# ============================================
# PASO 3: Calcular scores (Q · K^T)
# ============================================
print("--- Paso 3: Calcular scores ---")

# El producto punto mide similitud entre queries y keys
# Descomenta las siguientes líneas:
# scores = torch.matmul(Q, K.transpose(-2, -1))
# print(f"Scores shape: {scores.shape}")
# print(f"Scores:\n{scores[0]}")

print()

# ============================================
# PASO 4: Escalar por sqrt(d_k)
# ============================================
print("--- Paso 4: Escalar scores ---")

# Sin escalar, valores grandes hacen que softmax sature
# Descomenta las siguientes líneas:
# d_k_value = K.size(-1)
# scaled_scores = scores / (d_k_value ** 0.5)
# print(f"Factor de escala: sqrt({d_k_value}) = {d_k_value ** 0.5:.4f}")
# print(f"Scaled scores:\n{scaled_scores[0]}")

print()

# ============================================
# PASO 5: Aplicar Softmax
# ============================================
print("--- Paso 5: Softmax para obtener pesos ---")

# Softmax convierte scores a probabilidades (suman 1)
# Descomenta las siguientes líneas:
# attention_weights = F.softmax(scaled_scores, dim=-1)
# print(f"Attention weights shape: {attention_weights.shape}")
# print(f"Weights:\n{attention_weights[0]}")
# print(f"Suma por fila: {attention_weights[0].sum(dim=-1)}")

print()

# ============================================
# PASO 6: Multiplicar por Values
# ============================================
print("--- Paso 6: Weighted sum de Values ---")

# Los pesos de atención ponderan los values
# Descomenta las siguientes líneas:
# output = torch.matmul(attention_weights, V)
# print(f"Output shape: {output.shape}")
# print(f"Output:\n{output[0]}")

print()

# ============================================
# PASO 7: Función completa
# ============================================
print("--- Paso 7: Función scaled_dot_product_attention ---")

# Descomenta la función completa:
# def scaled_dot_product_attention(Q, K, V, mask=None):
#     """
#     Scaled Dot-Product Attention.
#
#     Args:
#         Q: Queries (batch, seq_len, d_k)
#         K: Keys (batch, seq_len, d_k)
#         V: Values (batch, seq_len, d_v)
#         mask: Optional mask (batch, seq_len, seq_len)
#
#     Returns:
#         output: Weighted values (batch, seq_len, d_v)
#         weights: Attention weights (batch, seq_len, seq_len)
#     """
#     d_k = K.size(-1)
#
#     # 1. Calcular scores
#     scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
#
#     # 2. Aplicar máscara (opcional)
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, float('-inf'))
#
#     # 3. Softmax
#     weights = F.softmax(scores, dim=-1)
#
#     # 4. Weighted sum
#     output = torch.matmul(weights, V)
#
#     return output, weights

# Prueba la función:
# output, weights = scaled_dot_product_attention(Q, K, V)
# print(f"Output shape: {output.shape}")
# print(f"Weights shape: {weights.shape}")

print()

# ============================================
# PASO 8: Visualizar atención
# ============================================
print("--- Paso 8: Visualizar pesos de atención ---")

# Descomenta para visualizar:
# tokens = ['[CLS]', 'Hello', 'World', '[SEP]']
#
# plt.figure(figsize=(6, 5))
# plt.imshow(weights[0].detach().numpy(), cmap='Blues')
# plt.colorbar(label='Attention Weight')
# plt.xticks(range(len(tokens)), tokens)
# plt.yticks(range(len(tokens)), tokens)
# plt.xlabel('Key (attending to)')
# plt.ylabel('Query (from)')
# plt.title('Attention Weights')
# plt.tight_layout()
# plt.savefig('attention_visualization.png', dpi=150)
# plt.show()
# print("Visualización guardada en 'attention_visualization.png'")

print()
print("=" * 50)
print("¡Ejercicio completado!")
print("=" * 50)
