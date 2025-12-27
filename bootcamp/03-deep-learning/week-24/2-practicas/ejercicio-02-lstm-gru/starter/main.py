"""
Ejercicio 02: LSTM y GRU
========================
Comparar arquitecturas LSTM y GRU.
"""

import torch
import torch.nn as nn

# ============================================
# PASO 1: LSTMCell
# ============================================
print("--- Paso 1: LSTMCell ---")

input_size = 4
hidden_size = 8

# Descomenta las siguientes líneas:
# lstm_cell = nn.LSTMCell(input_size, hidden_size)

# x = torch.randn(1, input_size)
# h = torch.zeros(1, hidden_size)
# c = torch.zeros(1, hidden_size)

# # LSTM retorna (h_new, c_new)
# h_new, c_new = lstm_cell(x, (h, c))
# print(f'h_new: {h_new.shape}')
# print(f'c_new: {c_new.shape}')

print()

# ============================================
# PASO 2: nn.LSTM Completa
# ============================================
print("--- Paso 2: nn.LSTM Completa ---")

# Descomenta las siguientes líneas:
# lstm = nn.LSTM(
#     input_size=4,
#     hidden_size=8,
#     num_layers=2,
#     batch_first=True,
#     dropout=0.2
# )

# x = torch.randn(2, 10, 4)  # (batch, seq, features)
# outputs, (h_n, c_n) = lstm(x)

# print(f'Input: {x.shape}')
# print(f'Outputs: {outputs.shape}')  # (batch, seq, hidden)
# print(f'h_n: {h_n.shape}')          # (num_layers, batch, hidden)
# print(f'c_n: {c_n.shape}')          # (num_layers, batch, hidden)

print()

# ============================================
# PASO 3: nn.GRU
# ============================================
print("--- Paso 3: nn.GRU ---")

# Descomenta las siguientes líneas:
# gru = nn.GRU(
#     input_size=4,
#     hidden_size=8,
#     num_layers=2,
#     batch_first=True
# )

# x = torch.randn(2, 10, 4)
# outputs, h_n = gru(x)

# print(f'Outputs: {outputs.shape}')
# print(f'h_n: {h_n.shape}')
# print('Nota: GRU no tiene c_n (no hay cell state)')

print()

# ============================================
# PASO 4: Comparar Parámetros
# ============================================
print("--- Paso 4: Comparar Parámetros ---")

# Descomenta las siguientes líneas:
# # Crear modelos con mismas dimensiones
# rnn = nn.RNN(10, 20, num_layers=1)
# lstm = nn.LSTM(10, 20, num_layers=1)
# gru = nn.GRU(10, 20, num_layers=1)

# rnn_params = sum(p.numel() for p in rnn.parameters())
# lstm_params = sum(p.numel() for p in lstm.parameters())
# gru_params = sum(p.numel() for p in gru.parameters())

# print(f'RNN params:  {rnn_params}')
# print(f'GRU params:  {gru_params}')
# print(f'LSTM params: {lstm_params}')
# print(f'Ratio GRU/RNN:  {gru_params/rnn_params:.2f}x')
# print(f'Ratio LSTM/RNN: {lstm_params/rnn_params:.2f}x')

print()

# ============================================
# PASO 5: Bidireccional
# ============================================
print("--- Paso 5: Bidireccional ---")

# Descomenta las siguientes líneas:
# bi_lstm = nn.LSTM(
#     input_size=4,
#     hidden_size=8,
#     bidirectional=True,
#     batch_first=True
# )

# x = torch.randn(2, 5, 4)
# out, (h_n, c_n) = bi_lstm(x)

# print(f'Input: {x.shape}')
# print(f'Output: {out.shape}')   # (2, 5, 16) - 2*hidden
# print(f'h_n: {h_n.shape}')      # (2, 2, 8) - 2 direcciones

# # Separar direcciones
# forward_out = out[:, :, :8]
# backward_out = out[:, :, 8:]
# print(f'Forward: {forward_out.shape}, Backward: {backward_out.shape}')

print()

# ============================================
# RESUMEN
# ============================================
print("--- Resumen ---")
print("✅ LSTM tiene 4 gates y cell state separado")
print("✅ GRU tiene 2 gates y es ~25% más rápido")
print("✅ Ambos resuelven vanishing gradient")
print("✅ Bidireccional duplica el output size")
