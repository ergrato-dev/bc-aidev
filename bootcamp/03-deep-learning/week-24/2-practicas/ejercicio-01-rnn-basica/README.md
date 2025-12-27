# 🔄 Ejercicio 01: RNN Básica desde Cero

## 🎯 Objetivo

Implementar una RNN simple paso a paso para entender el flujo de información y el estado oculto.

---

## 📋 Conceptos Clave

- Estado oculto ($h_t$)
- Propagación temporal
- Pesos compartidos en el tiempo

---

## 🔢 Pasos

### Paso 1: Celda RNN Manual

Implementaremos la fórmula:

$$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

```python
import torch
import torch.nn as nn

# Dimensiones
input_size = 4
hidden_size = 8

# Pesos (normalmente se inicializan aleatoriamente)
W_xh = torch.randn(hidden_size, input_size)
W_hh = torch.randn(hidden_size, hidden_size)
b_h = torch.zeros(hidden_size)

# Una entrada
x_t = torch.randn(input_size)

# Estado oculto inicial
h_prev = torch.zeros(hidden_size)

# Forward de una celda
h_t = torch.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
print(f'h_t shape: {h_t.shape}')
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

---

### Paso 2: Procesar una Secuencia

Iteramos sobre cada elemento de la secuencia:

```python
# Secuencia de 5 pasos temporales
seq_len = 5
sequence = torch.randn(seq_len, input_size)

# Procesar secuencia
h = torch.zeros(hidden_size)
outputs = []

for t in range(seq_len):
    x_t = sequence[t]
    h = torch.tanh(W_xh @ x_t + W_hh @ h + b_h)
    outputs.append(h)

outputs = torch.stack(outputs)
print(f'Outputs shape: {outputs.shape}')  # (5, 8)
```

---

### Paso 3: RNNCell de PyTorch

PyTorch proporciona `nn.RNNCell`:

```python
rnn_cell = nn.RNNCell(input_size=4, hidden_size=8)

# Una entrada
x = torch.randn(1, 4)  # (batch, input)
h = torch.zeros(1, 8)  # (batch, hidden)

# Forward
h_new = rnn_cell(x, h)
print(f'h_new: {h_new.shape}')
```

---

### Paso 4: nn.RNN Completa

Para secuencias completas:

```python
rnn = nn.RNN(
    input_size=4,
    hidden_size=8,
    num_layers=1,
    batch_first=True
)

# Batch de secuencias
x = torch.randn(2, 5, 4)  # (batch, seq, features)

# Forward
outputs, h_n = rnn(x)

print(f'Outputs: {outputs.shape}')  # (2, 5, 8)
print(f'h_n: {h_n.shape}')          # (1, 2, 8)
```

---

### Paso 5: Predicción Simple

Añadimos capa de salida:

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        # Usar último estado
        out = self.fc(out[:, -1, :])
        return out

model = SimpleRNN(4, 8, 2)
x = torch.randn(2, 5, 4)
y = model(x)
print(f'Output: {y.shape}')  # (2, 2)
```

---

## ✅ Checklist

- [ ] Implementé celda RNN manual
- [ ] Procesé secuencia paso a paso
- [ ] Usé RNNCell de PyTorch
- [ ] Usé nn.RNN completa
- [ ] Creé modelo con capa de salida

---

## 📁 Archivos

```
ejercicio-01-rnn-basica/
├── README.md
└── starter/
    └── main.py
```
