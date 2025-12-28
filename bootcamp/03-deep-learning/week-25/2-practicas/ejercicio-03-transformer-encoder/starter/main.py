"""
Ejercicio 03: Transformer Encoder
=================================

Construye un Transformer Encoder completo con todos sus componentes.
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
# PASO 2: Positional Encoding
# ============================================
print("--- Paso 2: Positional Encoding ---")

# Descomenta la clase:
# class PositionalEncoding(nn.Module):
#     """
#     Añade información posicional a los embeddings.
#
#     PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
#     PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#     """
#
#     def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout)
#
#         # Crear matriz de positional encoding
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#
#         pe[:, 0::2] = torch.sin(position * div_term)  # Posiciones pares
#         pe[:, 1::2] = torch.cos(position * div_term)  # Posiciones impares
#
#         pe = pe.unsqueeze(0)  # (1, max_len, d_model)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         """x: (batch, seq_len, d_model)"""
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)

print("PositionalEncoding definido")
print()

# ============================================
# PASO 3: Feed-Forward Network
# ============================================
print("--- Paso 3: Feed-Forward Network ---")

# Descomenta la clase:
# class FeedForward(nn.Module):
#     """
#     Feed-Forward Network: Linear -> ReLU -> Dropout -> Linear
#
#     Típicamente d_ff = 4 * d_model
#     """
#
#     def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
#         super().__init__()
#         self.linear1 = nn.Linear(d_model, d_ff)
#         self.linear2 = nn.Linear(d_ff, d_model)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         return self.linear2(self.dropout(F.relu(self.linear1(x))))

print("FeedForward definido")
print()

# ============================================
# PASO 4: Multi-Head Attention (simplificado)
# ============================================
print("--- Paso 4: Multi-Head Attention ---")

# Descomenta la clase:
# class MultiHeadAttention(nn.Module):
#     """Multi-Head Attention (versión simplificada)."""
#
#     def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
#         super().__init__()
#         assert d_model % num_heads == 0
#
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
#
#         self.W_q = nn.Linear(d_model, d_model)
#         self.W_k = nn.Linear(d_model, d_model)
#         self.W_v = nn.Linear(d_model, d_model)
#         self.W_o = nn.Linear(d_model, d_model)
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, Q, K, V, mask=None):
#         batch_size, seq_len, _ = Q.size()
#
#         # Proyecciones
#         Q = self.W_q(Q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         K = self.W_k(K).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         V = self.W_v(V).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#
#         # Scaled dot-product attention
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))
#         weights = F.softmax(scores, dim=-1)
#         weights = self.dropout(weights)
#
#         # Output
#         attn_output = torch.matmul(weights, V)
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
#
#         return self.W_o(attn_output)

print("MultiHeadAttention definido")
print()

# ============================================
# PASO 5: Encoder Layer
# ============================================
print("--- Paso 5: Encoder Layer ---")

# Descomenta la clase:
# class EncoderLayer(nn.Module):
#     """
#     Una capa del Transformer Encoder.
#
#     Arquitectura:
#     1. Multi-Head Self-Attention + Add & Norm
#     2. Feed-Forward + Add & Norm
#     """
#
#     def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
#         super().__init__()
#
#         self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
#         self.feed_forward = FeedForward(d_model, d_ff, dropout)
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, mask=None):
#         # Self-Attention con residual
#         attn_output = self.self_attention(x, x, x, mask)
#         x = self.norm1(x + self.dropout(attn_output))
#
#         # Feed-Forward con residual
#         ff_output = self.feed_forward(x)
#         x = self.norm2(x + self.dropout(ff_output))
#
#         return x

print("EncoderLayer definido")
print()

# ============================================
# PASO 6: Transformer Encoder Completo
# ============================================
print("--- Paso 6: Transformer Encoder ---")

# Descomenta la clase:
# class TransformerEncoder(nn.Module):
#     """
#     Transformer Encoder completo.
#
#     Componentes:
#     1. Token Embedding
#     2. Positional Encoding
#     3. N x Encoder Layers
#     """
#
#     def __init__(
#         self,
#         vocab_size: int,
#         d_model: int = 512,
#         num_heads: int = 8,
#         num_layers: int = 6,
#         d_ff: int = 2048,
#         max_len: int = 512,
#         dropout: float = 0.1
#     ):
#         super().__init__()
#
#         self.d_model = d_model
#
#         # Embedding + Positional Encoding
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
#
#         # Stack de Encoder Layers
#         self.layers = nn.ModuleList([
#             EncoderLayer(d_model, num_heads, d_ff, dropout)
#             for _ in range(num_layers)
#         ])
#
#         self.norm = nn.LayerNorm(d_model)
#
#     def forward(self, x, mask=None):
#         # x: (batch, seq_len) - índices de tokens
#
#         # Embedding + scale + positional encoding
#         x = self.embedding(x) * math.sqrt(self.d_model)
#         x = self.pos_encoding(x)
#
#         # Pasar por todas las capas
#         for layer in self.layers:
#             x = layer(x, mask)
#
#         return self.norm(x)

print("TransformerEncoder definido")
print()

# ============================================
# PASO 7: Probar el Encoder
# ============================================
print("--- Paso 7: Probar el Encoder ---")

# Descomenta para probar:
# torch.manual_seed(42)
#
# # Configuración
# vocab_size = 10000
# d_model = 256
# num_heads = 8
# num_layers = 4
# d_ff = 1024
# batch_size = 2
# seq_len = 16
#
# # Crear encoder
# encoder = TransformerEncoder(
#     vocab_size=vocab_size,
#     d_model=d_model,
#     num_heads=num_heads,
#     num_layers=num_layers,
#     d_ff=d_ff
# )
#
# # Datos de entrada (índices de tokens)
# tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
#
# # Forward pass
# output = encoder(tokens)
#
# print(f"Input tokens shape: {tokens.shape}")
# print(f"Output shape: {output.shape}")
# print(f"Número de parámetros: {sum(p.numel() for p in encoder.parameters()):,}")

print()

# ============================================
# PASO 8: Contar parámetros por componente
# ============================================
print("--- Paso 8: Análisis de parámetros ---")

# Descomenta para analizar:
# def count_parameters(model, name=""):
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"{name}: {total:,} total, {trainable:,} trainable")
#     return total
#
# print("\nDesglose de parámetros:")
# count_parameters(encoder.embedding, "Embedding")
# count_parameters(encoder.layers[0].self_attention, "Self-Attention (1 capa)")
# count_parameters(encoder.layers[0].feed_forward, "Feed-Forward (1 capa)")
# count_parameters(encoder.layers[0], "EncoderLayer (1 capa)")
# print(f"\nTotal encoder: {sum(p.numel() for p in encoder.parameters()):,}")

print()
print("=" * 50)
print("¡Ejercicio completado!")
print("=" * 50)
