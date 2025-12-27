# 📈 Ejercicio 03: Series Temporales con RNN

## 🎯 Objetivo

Preparar datos de series temporales y entrenar un modelo LSTM para predicción.

---

## 📋 Conceptos Clave

- Ventanas deslizantes (sliding windows)
- Normalización de series temporales
- Predicción multi-step

---

## 🔢 Pasos

### Paso 1: Crear Dataset Sintético

```python
import torch
import numpy as np

# Serie temporal: seno con ruido
t = np.linspace(0, 100, 1000)
data = np.sin(t) + 0.1 * np.random.randn(1000)
```

---

### Paso 2: Crear Ventanas

```python
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 20
X, y = create_sequences(data, seq_length)
```

---

### Paso 3: Normalizar

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
X, y = create_sequences(data_scaled, seq_length)
```

---

### Paso 4: Modelo LSTM

```python
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

---

### Paso 5: Entrenar

```python
model = LSTMPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    pred = model(X_tensor)
    loss = criterion(pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## ✅ Checklist

- [ ] Creé dataset sintético
- [ ] Implementé ventanas deslizantes
- [ ] Normalicé los datos
- [ ] Entrené modelo LSTM
- [ ] Visualicé predicciones

---

## 📁 Archivos

```
ejercicio-03-series-temporales/
├── README.md
└── starter/
    └── main.py
```
