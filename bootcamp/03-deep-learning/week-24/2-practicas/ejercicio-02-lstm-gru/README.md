# 🧠 Ejercicio 02: LSTM y GRU

## 🎯 Objetivo

Comparar LSTM y GRU en términos de arquitectura, parámetros y uso práctico.

---

## 📋 Conceptos Clave

- LSTM: 4 gates (forget, input, cell, output)
- GRU: 2 gates (reset, update)
- Cell state vs Hidden state

---

## 🔢 Pasos

### Paso 1: LSTMCell Manual

```python
import torch
import torch.nn as nn

# Dimensiones
input_size = 4
hidden_size = 8

# Una celda LSTM
lstm_cell = nn.LSTMCell(input_size, hidden_size)

x = torch.randn(1, input_size)
h = torch.zeros(1, hidden_size)
c = torch.zeros(1, hidden_size)

h_new, c_new = lstm_cell(x, (h, c))
print(f'h_new: {h_new.shape}, c_new: {c_new.shape}')
```

---

### Paso 2: nn.LSTM Completa

```python
lstm = nn.LSTM(
    input_size=4,
    hidden_size=8,
    num_layers=2,
    batch_first=True,
    dropout=0.2
)

x = torch.randn(2, 10, 4)  # (batch, seq, features)
outputs, (h_n, c_n) = lstm(x)

print(f'Outputs: {outputs.shape}')  # (2, 10, 8)
print(f'h_n: {h_n.shape}')          # (2, 2, 8)
print(f'c_n: {c_n.shape}')          # (2, 2, 8)
```

---

### Paso 3: GRU

```python
gru = nn.GRU(
    input_size=4,
    hidden_size=8,
    num_layers=2,
    batch_first=True
)

x = torch.randn(2, 10, 4)
outputs, h_n = gru(x)

print(f'Outputs: {outputs.shape}')
print(f'h_n: {h_n.shape}')
# Nota: GRU no tiene c_n
```

---

### Paso 4: Comparar Parámetros

```python
lstm = nn.LSTM(10, 20, num_layers=1)
gru = nn.GRU(10, 20, num_layers=1)

lstm_params = sum(p.numel() for p in lstm.parameters())
gru_params = sum(p.numel() for p in gru.parameters())

print(f'LSTM params: {lstm_params}')
print(f'GRU params: {gru_params}')
print(f'Ratio LSTM/GRU: {lstm_params/gru_params:.2f}')
```

---

### Paso 5: Bidireccional

```python
bi_lstm = nn.LSTM(4, 8, bidirectional=True, batch_first=True)
x = torch.randn(2, 5, 4)
out, (h_n, c_n) = bi_lstm(x)

print(f'Output: {out.shape}')  # (2, 5, 16) - 2*hidden
print(f'h_n: {h_n.shape}')     # (2, 2, 8)
```

---

## ✅ Checklist

- [ ] Usé LSTMCell y GRUCell
- [ ] Usé nn.LSTM y nn.GRU
- [ ] Comparé número de parámetros
- [ ] Implementé versiones bidireccionales

---

## 📁 Archivos

```
ejercicio-02-lstm-gru/
├── README.md
└── starter/
    └── main.py
```
