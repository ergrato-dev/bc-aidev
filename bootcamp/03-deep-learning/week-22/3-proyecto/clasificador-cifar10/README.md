# 🎯 Proyecto: Clasificador de Imágenes CIFAR-10

## 📋 Descripción

Construirás una CNN desde cero para clasificar imágenes del dataset CIFAR-10, aplicando todos los conceptos de convolución, pooling y arquitecturas aprendidos esta semana.

**CIFAR-10** contiene 60,000 imágenes a color (32×32×3) en 10 clases:
- ✈️ Avión (airplane)
- 🚗 Automóvil (automobile)
- 🐦 Pájaro (bird)
- 🐱 Gato (cat)
- 🦌 Ciervo (deer)
- 🐕 Perro (dog)
- 🐸 Rana (frog)
- 🐴 Caballo (horse)
- 🚢 Barco (ship)
- 🚚 Camión (truck)

---

## 🎯 Objetivos

1. **Diseñar** una arquitectura CNN apropiada para CIFAR-10
2. **Implementar** la red usando PyTorch
3. **Entrenar** el modelo con técnicas de regularización
4. **Evaluar** y analizar el rendimiento
5. **Visualizar** filtros y predicciones

---

## 📊 Requisitos del Modelo

### Arquitectura Mínima

Tu CNN debe incluir:

- [ ] Al menos **3 bloques convolucionales**
- [ ] **Batch Normalization** después de cada convolución
- [ ] **MaxPooling** o stride > 1 para reducir dimensiones
- [ ] **Dropout** para regularización
- [ ] **Clasificador** con al menos una capa oculta

### Métricas Objetivo

| Métrica | Mínimo | Deseable |
|---------|--------|----------|
| Test Accuracy | ≥ 70% | ≥ 75% |
| Training completo | ≤ 20 épocas | ≤ 15 épocas |

---

## 🏗️ Arquitectura Sugerida

```
Input: 32×32×3 (imagen RGB)
│
├── Bloque 1
│   ├── Conv2d(3, 32, 3, padding=1)
│   ├── BatchNorm2d(32)
│   ├── ReLU
│   └── MaxPool2d(2)  → 16×16×32
│
├── Bloque 2
│   ├── Conv2d(32, 64, 3, padding=1)
│   ├── BatchNorm2d(64)
│   ├── ReLU
│   └── MaxPool2d(2)  → 8×8×64
│
├── Bloque 3
│   ├── Conv2d(64, 128, 3, padding=1)
│   ├── BatchNorm2d(128)
│   ├── ReLU
│   └── MaxPool2d(2)  → 4×4×128
│
├── Flatten → 2048
│
├── Clasificador
│   ├── Linear(2048, 256)
│   ├── ReLU
│   ├── Dropout(0.5)
│   └── Linear(256, 10)
│
└── Output: 10 clases
```

---

## 📝 Entregables

### 1. Código (70%)

**Archivo**: `starter/main.py`

Debe contener:

```python
# Imports necesarios
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Definición del modelo
class CIFAR10CNN(nn.Module):
    # TODO: Implementar arquitectura

# 2. Data augmentation y loaders
# TODO: Implementar transformaciones

# 3. Funciones de entrenamiento
# TODO: train_epoch, evaluate

# 4. Loop de entrenamiento
# TODO: Entrenar el modelo

# 5. Evaluación final
# TODO: Métricas y visualizaciones
```

### 2. Resultados (20%)

Generar los siguientes archivos:

- `training_curves.png` - Curvas de loss y accuracy
- `confusion_matrix.png` - Matriz de confusión
- `sample_predictions.png` - Ejemplos de predicciones
- `model_cifar10.pth` - Modelo entrenado

### 3. Análisis (10%)

Responder en comentarios del código:

1. ¿Por qué CIFAR-10 es más difícil que MNIST?
2. ¿Qué efecto tiene el data augmentation?
3. ¿Qué clases confunde más el modelo?

---

## 🔧 Configuración Recomendada

```python
# Hiperparámetros sugeridos
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
WEIGHT_DECAY = 1e-4

# Data Augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])
```

---

## 📈 Rúbrica de Evaluación

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| **Arquitectura** | 25 | CNN con ≥3 bloques, BatchNorm, Dropout |
| **Data Augmentation** | 15 | Flip, Crop, Normalización correcta |
| **Entrenamiento** | 20 | Loop correcto, scheduler, early stopping |
| **Accuracy ≥70%** | 20 | Alcanzar métrica objetivo |
| **Visualizaciones** | 10 | Curvas, matriz confusión, predicciones |
| **Código limpio** | 10 | Comentarios, organización, type hints |
| **Total** | 100 | |

---

## 💡 Tips

1. **Empieza simple**: Primero haz que funcione, luego optimiza
2. **Data augmentation es clave**: Sin él, accuracy baja ~5-10%
3. **Learning rate scheduling**: Reduce LR cuando loss se estanca
4. **Monitorea overfitting**: Si train_acc >> test_acc, agrega regularización
5. **GPU**: Usa CUDA si está disponible (`torch.cuda.is_available()`)

---

## 🔗 Recursos

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch CIFAR-10 Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Data Augmentation en PyTorch](https://pytorch.org/vision/stable/transforms.html)

---

## 📅 Tiempo Estimado

| Fase | Tiempo |
|------|--------|
| Setup y datos | 15 min |
| Diseño arquitectura | 30 min |
| Implementación | 45 min |
| Entrenamiento | 20 min |
| Evaluación y visualización | 10 min |
| **Total** | ~2 horas |

---

## 🚀 ¡Comienza!

1. Abre `starter/main.py`
2. Implementa cada sección marcada con TODO
3. Ejecuta y entrena tu modelo
4. Genera las visualizaciones requeridas
5. ¡Alcanza el 70% de accuracy!

**¡Buena suerte! 🍀**
