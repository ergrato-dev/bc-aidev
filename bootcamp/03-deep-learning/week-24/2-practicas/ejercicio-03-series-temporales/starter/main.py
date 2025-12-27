"""
Ejercicio 03: Series Temporales con RNN
=======================================
Preparar y entrenar LSTM para predicción.
"""

import numpy as np
import torch
import torch.nn as nn

# ============================================
# PASO 1: Crear Dataset Sintético
# ============================================
print("--- Paso 1: Crear Dataset Sintético ---")

# Descomenta las siguientes líneas:
# np.random.seed(42)
# t = np.linspace(0, 100, 1000)
# data = np.sin(t) + 0.1 * np.random.randn(1000)
# print(f'Data shape: {data.shape}')
# print(f'Min: {data.min():.2f}, Max: {data.max():.2f}')

print()

# ============================================
# PASO 2: Crear Ventanas Deslizantes
# ============================================
print("--- Paso 2: Crear Ventanas Deslizantes ---")


def create_sequences(data, seq_length):
    """Crear secuencias X, y para entrenamiento."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# Descomenta las siguientes líneas:
# seq_length = 20
# X, y = create_sequences(data, seq_length)
# print(f'X shape: {X.shape}')  # (980, 20)
# print(f'y shape: {y.shape}')  # (980,)

print()

# ============================================
# PASO 3: Normalizar Datos
# ============================================
print("--- Paso 3: Normalizar Datos ---")

# Descomenta las siguientes líneas:
# # Normalización manual (MinMax)
# data_min = data.min()
# data_max = data.max()
# data_scaled = (data - data_min) / (data_max - data_min)

# X, y = create_sequences(data_scaled, seq_length)
# print(f'Scaled range: [{data_scaled.min():.2f}, {data_scaled.max():.2f}]')

print()

# ============================================
# PASO 4: Preparar Tensores
# ============================================
print("--- Paso 4: Preparar Tensores ---")

# Descomenta las siguientes líneas:
# # Split train/test
# split = int(len(X) * 0.8)
# X_train, X_test = X[:split], X[split:]
# y_train, y_test = y[:split], y[split:]

# # Convertir a tensores
# X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # (N, seq, 1)
# y_train = torch.FloatTensor(y_train).unsqueeze(-1)  # (N, 1)
# X_test = torch.FloatTensor(X_test).unsqueeze(-1)
# y_test = torch.FloatTensor(y_test).unsqueeze(-1)

# print(f'X_train: {X_train.shape}')
# print(f'y_train: {y_train.shape}')

print()

# ============================================
# PASO 5: Definir Modelo LSTM
# ============================================
print("--- Paso 5: Definir Modelo LSTM ---")

# Descomenta las siguientes líneas:
# class LSTMPredictor(nn.Module):
#     def __init__(self, input_size=1, hidden_size=32, num_layers=1):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_size, hidden_size,
#             num_layers=num_layers,
#             batch_first=True
#         )
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         out, (h_n, c_n) = self.lstm(x)
#         # Usar último hidden state
#         out = self.fc(out[:, -1, :])
#         return out

# model = LSTMPredictor(hidden_size=32)
# print(model)
# total_params = sum(p.numel() for p in model.parameters())
# print(f'Total params: {total_params}')

print()

# ============================================
# PASO 6: Entrenar
# ============================================
print("--- Paso 6: Entrenar ---")

# Descomenta las siguientes líneas:
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# epochs = 50
# for epoch in range(epochs):
#     model.train()
#     pred = model(X_train)
#     loss = criterion(pred, y_train)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if (epoch + 1) % 10 == 0:
#         model.eval()
#         with torch.no_grad():
#             test_pred = model(X_test)
#             test_loss = criterion(test_pred, y_test)
#         print(f'Epoch {epoch+1}: Train Loss={loss.item():.4f}, Test Loss={test_loss.item():.4f}')

print()

# ============================================
# PASO 7: Evaluar
# ============================================
print("--- Paso 7: Evaluar ---")

# Descomenta las siguientes líneas:
# model.eval()
# with torch.no_grad():
#     predictions = model(X_test)
#     mse = criterion(predictions, y_test).item()
#     mae = torch.abs(predictions - y_test).mean().item()

# print(f'Test MSE: {mse:.4f}')
# print(f'Test MAE: {mae:.4f}')

# # Desnormalizar para ver valores reales
# pred_real = predictions.numpy() * (data_max - data_min) + data_min
# y_real = y_test.numpy() * (data_max - data_min) + data_min
# mae_real = np.abs(pred_real - y_real).mean()
# print(f'MAE (escala original): {mae_real:.4f}')

print()

# ============================================
# RESUMEN
# ============================================
print("--- Resumen ---")
print("✅ Ventanas deslizantes convierten serie en supervisado")
print("✅ Normalización es crítica para convergencia")
print("✅ LSTM captura patrones temporales")
print("✅ MSE/MAE evalúan calidad de predicción")
