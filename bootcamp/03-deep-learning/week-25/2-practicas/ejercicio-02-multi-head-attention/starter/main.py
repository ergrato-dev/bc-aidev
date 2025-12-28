"""
Ejercicio 02: Multi-Head Attention
==================================

Implementa Multi-Head Attention con múltiples heads en paralelo.
"""

# ============================================
# PASO 1: Importaciones
# ============================================
print("--- Paso 1: Importaciones ---")

# Descomenta las siguientes líneas:
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

print("Importaciones completadas")
print()

# ============================================
# PASO 2: Función de atención base
# ============================================
print("--- Paso 2: Scaled Dot-Product Attention ---")

# Descomenta la función (del ejercicio anterior):
# def scaled_dot_product_attention(Q, K, V, mask=None):
#     """Scaled Dot-Product Attention."""
#     d_k = K.size(-1)
#     scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
#
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, float('-inf'))
#
#     weights = F.softmax(scores, dim=-1)
#     output = torch.matmul(weights, V)
#
#     return output, weights

print("Función de atención definida")
print()

# ============================================
# PASO 3: Clase MultiHeadAttention - Estructura
# ============================================
print("--- Paso 3: Estructura de MultiHeadAttention ---")

# Descomenta la clase:
# class MultiHeadAttention(nn.Module):
#     """
#     Multi-Head Attention Layer.
#
#     Args:
#         d_model: Dimensión del modelo (embedding size)
#         num_heads: Número de heads de atención
#         dropout: Probabilidad de dropout
#     """
#
#     def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
#         super().__init__()
#
#         assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
#
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads  # Dimensión por head
#
#         # Proyecciones lineales para Q, K, V
#         self.W_q = nn.Linear(d_model, d_model)
#         self.W_k = nn.Linear(d_model, d_model)
#         self.W_v = nn.Linear(d_model, d_model)
#
#         # Proyección de salida
#         self.W_o = nn.Linear(d_model, d_model)
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, Q, K, V, mask=None):
#         batch_size = Q.size(0)
#         seq_len = Q.size(1)
#
#         # Paso 1: Proyecciones lineales
#         Q = self.W_q(Q)
#         K = self.W_k(K)
#         V = self.W_v(V)
#
#         # Paso 2: Reshape para múltiples heads
#         # De (batch, seq, d_model) a (batch, num_heads, seq, d_k)
#         Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#
#         # Paso 3: Atención para todos los heads en paralelo
#         attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
#
#         # Paso 4: Concatenar heads
#         # De (batch, num_heads, seq, d_k) a (batch, seq, d_model)
#         attn_output = attn_output.transpose(1, 2).contiguous()
#         attn_output = attn_output.view(batch_size, seq_len, self.d_model)
#
#         # Paso 5: Proyección final
#         output = self.W_o(attn_output)
#         output = self.dropout(output)
#
#         return output, attn_weights

print("Clase definida")
print()

# ============================================
# PASO 4: Probar la implementación
# ============================================
print("--- Paso 4: Probar MultiHeadAttention ---")

# Descomenta para probar:
# torch.manual_seed(42)
#
# # Configuración
# batch_size = 2
# seq_len = 4
# d_model = 64
# num_heads = 8
#
# # Crear datos de entrada
# x = torch.randn(batch_size, seq_len, d_model)
#
# # Crear capa de Multi-Head Attention
# mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
#
# # Forward pass (self-attention: Q=K=V=x)
# output, weights = mha(x, x, x)
#
# print(f"Input shape: {x.shape}")
# print(f"Output shape: {output.shape}")
# print(f"Weights shape: {weights.shape}")
# print(f"d_k per head: {d_model // num_heads}")

print()

# ============================================
# PASO 5: Verificar dimensiones
# ============================================
print("--- Paso 5: Verificar dimensiones ---")

# Descomenta para verificar:
# print("Verificaciones:")
# print(f"  - Output tiene misma forma que input: {output.shape == x.shape}")
# print(f"  - Weights: (batch, heads, seq, seq) = {weights.shape}")
# print(f"  - Weights suman 1 por fila: {torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))}")

print()

# ============================================
# PASO 6: Comparar con PyTorch oficial
# ============================================
print("--- Paso 6: Comparar con nn.MultiheadAttention ---")

# Descomenta para comparar:
# # PyTorch espera (seq, batch, d_model)
# mha_pytorch = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
#
# output_pytorch, weights_pytorch = mha_pytorch(x, x, x)
#
# print(f"Output PyTorch shape: {output_pytorch.shape}")
# print(f"Nuestra implementación funciona correctamente!")

print()

# ============================================
# PASO 7: Visualizar diferentes heads
# ============================================
print("--- Paso 7: Visualizar atención por head ---")

# Descomenta para visualizar:
# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(2, 4, figsize=(12, 6))
# tokens = ['Token1', 'Token2', 'Token3', 'Token4']
#
# for i, ax in enumerate(axes.flat):
#     if i < num_heads:
#         im = ax.imshow(weights[0, i].detach().numpy(), cmap='Blues')
#         ax.set_title(f'Head {i+1}')
#         ax.set_xticks(range(len(tokens)))
#         ax.set_yticks(range(len(tokens)))
#         ax.set_xticklabels(tokens, fontsize=8)
#         ax.set_yticklabels(tokens, fontsize=8)
#
# plt.suptitle('Attention Patterns por Head')
# plt.tight_layout()
# plt.savefig('multi_head_attention.png', dpi=150)
# plt.show()
# print("Visualización guardada en 'multi_head_attention.png'")

print()
print("=" * 50)
print("¡Ejercicio completado!")
print("=" * 50)
