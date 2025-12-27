# 🌡️ Proyecto: Predictor de Temperatura con LSTM

## 🎯 Objetivo

Construir un modelo LSTM para predecir temperaturas futuras basándose en datos históricos.

**Meta**: Alcanzar MAE < 2°C en el conjunto de test.

---

## 📋 Descripción

Usarás datos de temperatura sintéticos (patrón estacional + tendencia + ruido) para entrenar un modelo LSTM que prediga la temperatura del día siguiente.

---

## 🗂️ Estructura

```
predictor-temperatura/
├── README.md
├── starter/
│   └── main.py      # TODO: Implementar
└── solution/
    └── main.py      # Solución completa
```

---

## 📝 Requisitos

### Datos
- Generar 3 años de datos diarios de temperatura
- Patrón: estacional (seno anual) + tendencia + ruido
- Split: 80% train, 20% test

### Modelo
- Arquitectura LSTM con al menos 2 capas
- Dropout para regularización
- Ventana de entrada: 30 días

### Entrenamiento
- Loss: MSE
- Optimizer: Adam
- Early stopping opcional

### Evaluación
- **MAE < 2°C** en test set
- Visualizar predicciones vs valores reales

---

## 🔧 Funciones a Implementar

```python
def generate_temperature_data(days: int) -> np.ndarray:
    """Generar datos sintéticos de temperatura."""
    # TODO: Implementar
    pass

def create_sequences(data: np.ndarray, seq_len: int) -> tuple:
    """Crear ventanas deslizantes."""
    # TODO: Implementar
    pass

class TemperatureLSTM(nn.Module):
    """Modelo LSTM para predicción de temperatura."""
    # TODO: Implementar
    pass

def train_model(model, train_loader, val_loader, epochs: int):
    """Entrenar el modelo."""
    # TODO: Implementar
    pass

def evaluate_model(model, test_loader, scaler) -> float:
    """Evaluar y retornar MAE en escala original."""
    # TODO: Implementar
    pass
```

---

## 📊 Criterios de Éxito

| Criterio | Requisito |
|----------|-----------|
| MAE Test | < 2°C |
| Código | Documentado y limpio |
| Visualización | Gráfico pred vs real |

---

## 💡 Hints

1. **Normalización**: Usa MinMaxScaler o StandardScaler
2. **Ventana**: 30 días captura bien el patrón semanal
3. **Hidden size**: 64-128 suele funcionar bien
4. **Learning rate**: Empieza con 0.001
5. **Epochs**: 50-100 con early stopping

---

## 📚 Recursos

- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Time Series Forecasting](https://machinelearningmastery.com/time-series-forecasting/)
