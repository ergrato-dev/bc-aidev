# 👗 Proyecto: Clasificador Fashion-MNIST

## 🎯 Objetivo

Construir un clasificador de imágenes de ropa usando PyTorch desde cero, implementando el training loop completo y alcanzando **≥88% accuracy** en el test set.

---

## 📋 Descripción

**Fashion-MNIST** es un dataset de Zalando con 70,000 imágenes de 10 categorías de ropa:

| Clase | Descripción     |
| ----- | --------------- |
| 0     | T-shirt/top     |
| 1     | Trouser         |
| 2     | Pullover        |
| 3     | Dress           |
| 4     | Coat            |
| 5     | Sandal          |
| 6     | Shirt           |
| 7     | Sneaker         |
| 8     | Bag             |
| 9     | Ankle boot      |

- **Imágenes**: 28x28 píxeles, escala de grises
- **Train**: 60,000 imágenes
- **Test**: 10,000 imágenes

---

## 📂 Estructura del Proyecto

```
clasificador-fashion-mnist/
├── README.md
├── starter/
│   └── main.py          # Plantilla con TODOs
└── solution/
    └── main.py          # Solución de referencia
```

---

## 🛠️ Requisitos Técnicos

### Arquitectura del Modelo

- Input: 784 (28×28 aplanado)
- Al menos 2 capas ocultas
- Dropout para regularización
- Output: 10 clases

### Training

- Optimizador: Adam
- Loss: CrossEntropyLoss
- Mínimo 10 epochs
- Batch size: 64

### Evaluación

- Accuracy en test ≥ 88%
- Visualización de loss y accuracy
- Matriz de confusión (opcional)

---

## 📝 Tareas

### 1. Cargar Datos (15 min)

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(...)
test_dataset = datasets.FashionMNIST(...)
```

### 2. Definir Modelo (20 min)

```python
class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Definir capas
    
    def forward(self, x):
        # TODO: Implementar forward pass
        pass
```

### 3. Implementar Training Loop (30 min)

```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    # TODO: Implementar
    pass

def evaluate(model, loader, criterion, device):
    model.eval()
    # TODO: Implementar
    pass
```

### 4. Entrenar y Evaluar (30 min)

- Entrenar por múltiples epochs
- Registrar métricas
- Evaluar en test set

### 5. Visualización (25 min)

- Gráficas de loss y accuracy
- Ejemplos de predicciones
- Matriz de confusión

---

## ✅ Criterios de Evaluación

| Criterio                    | Puntos |
| --------------------------- | ------ |
| Carga correcta de datos     | 15     |
| Arquitectura del modelo     | 20     |
| Training loop completo      | 25     |
| Accuracy ≥ 88%              | 20     |
| Visualizaciones             | 10     |
| Código limpio y documentado | 10     |
| **Total**                   | **100** |

---

## 💡 Hints

1. **Normalización**: Las imágenes originales están en [0, 255], normaliza a [-1, 1]
2. **Flatten**: Usa `x.view(x.size(0), -1)` o `nn.Flatten()`
3. **Device**: Mueve modelo y datos al mismo dispositivo
4. **Debugging**: Imprime shapes en forward para verificar dimensiones

---

## 📚 Recursos

- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [PyTorch DataLoader Tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

---

## 🚀 Extensiones (Opcional)

- Implementar learning rate scheduler
- Agregar más capas o usar arquitectura diferente
- Data augmentation
- Early stopping
- Guardar mejor modelo durante entrenamiento
