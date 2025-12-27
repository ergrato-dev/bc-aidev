"""
Ejercicio 01: RNN Básica desde Cero
===================================
Implementa una RNN paso a paso.
"""

import torch
import torch.nn as nn

# ============================================
# PASO 1: Celda RNN Manual
# ============================================
print("--- Paso 1: Celda RNN Manual ---")

# Dimensiones
input_size = 4
hidden_size = 8

# Descomenta las siguientes líneas:
# W_xh = torch.randn(hidden_size, input_size)
# W_hh = torch.randn(hidden_size, hidden_size)
# b_h = torch.zeros(hidden_size)

# x_t = torch.randn(input_size)
# h_prev = torch.zeros(hidden_size)

# # Forward de una celda: h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
# h_t = torch.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
# print(f'h_t shape: {h_t.shape}')
# print(f'h_t values (primeros 4): {h_t[:4]}')

print()

# ============================================
# PASO 2: Procesar una Secuencia
# ============================================
print("--- Paso 2: Procesar una Secuencia ---")

# Descomenta las siguientes líneas:
# seq_len = 5
# sequence = torch.randn(seq_len, input_size)

# # Procesar secuencia completa
# h = torch.zeros(hidden_size)
# outputs = []

# for t in range(seq_len):
#     x_t = sequence[t]
#     h = torch.tanh(W_xh @ x_t + W_hh @ h + b_h)
#     outputs.append(h.clone())
#     print(f't={t}: h norm = {h.norm():.4f}')

# outputs = torch.stack(outputs)
# print(f'Outputs shape: {outputs.shape}')

print()

# ============================================
# PASO 3: RNNCell de PyTorch
# ============================================
print("--- Paso 3: RNNCell de PyTorch ---")

# Descomenta las siguientes líneas:
# rnn_cell = nn.RNNCell(input_size=4, hidden_size=8)

# # Una entrada (batch_size=1)
# x = torch.randn(1, 4)
# h = torch.zeros(1, 8)

# # Forward
# h_new = rnn_cell(x, h)
# print(f'h_new shape: {h_new.shape}')

# # Procesar secuencia con RNNCell
# sequence = torch.randn(5, 1, 4)  # (seq, batch, features)
# h = torch.zeros(1, 8)
# for t in range(5):
#     h = rnn_cell(sequence[t], h)
# print(f'Final h: {h.shape}')

print()

# ============================================
# PASO 4: nn.RNN Completa
# ============================================
print("--- Paso 4: nn.RNN Completa ---")

# Descomenta las siguientes líneas:
# rnn = nn.RNN(
#     input_size=4,
#     hidden_size=8,
#     num_layers=1,
#     batch_first=True
# )

# # Batch de secuencias
# x = torch.randn(2, 5, 4)  # (batch, seq, features)

# # Forward
# outputs, h_n = rnn(x)

# print(f'Input shape: {x.shape}')
# print(f'Outputs shape: {outputs.shape}')  # (batch, seq, hidden)
# print(f'h_n shape: {h_n.shape}')          # (num_layers, batch, hidden)

# # El último output es igual a h_n
# print(f'outputs[:, -1] == h_n[0]: {torch.allclose(outputs[:, -1], h_n[0])}')

print()

# ============================================
# PASO 5: Modelo Simple con Capa de Salida
# ============================================
print("--- Paso 5: Modelo Simple con Capa de Salida ---")

# Descomenta las siguientes líneas:
# class SimpleRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # x: (batch, seq, features)
#         out, h_n = self.rnn(x)
#         # Usar último estado para clasificación
#         out = self.fc(out[:, -1, :])
#         return out

# model = SimpleRNN(input_size=4, hidden_size=8, output_size=2)
# x = torch.randn(2, 5, 4)
# y = model(x)
# print(f'Input: {x.shape}')
# print(f'Output: {y.shape}')

# # Contar parámetros
# total_params = sum(p.numel() for p in model.parameters())
# print(f'Total parámetros: {total_params}')

print()

# ============================================
# RESUMEN
# ============================================
print("--- Resumen ---")
print("✅ RNN procesa secuencias manteniendo estado oculto")
print("✅ Los pesos se comparten en todos los pasos temporales")
print("✅ nn.RNN es más eficiente que iterar con RNNCell")
print("✅ El último h_t se usa típicamente para clasificación")
