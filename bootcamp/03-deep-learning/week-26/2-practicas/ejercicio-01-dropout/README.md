# 🎲 Ejercicio 01: Implementar Dropout

## 🎯 Objetivo

Implementar y comparar modelos con diferentes configuraciones de Dropout para combatir overfitting.

---

## 📋 Descripción

En este ejercicio aprenderás:

1. Cómo Dropout reduce overfitting
2. Elegir valores apropiados de p
3. Diferencia entre `model.train()` y `model.eval()`
4. Comparar rendimiento con/sin Dropout

---

## 🔧 Requisitos

```bash
pip install torch torchvision matplotlib
```

---

## 📝 Pasos del Ejercicio

### Paso 1: Cargar MNIST y Crear DataLoaders

Cargamos el dataset MNIST y creamos loaders para entrenamiento y test.

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Modelo SIN Dropout (Baseline)

Creamos un modelo MLP simple sin regularización.

```python
model_no_dropout = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

Este modelo tenderá a hacer overfitting con suficientes épocas.

**Descomenta** la sección del Paso 2 en `starter/main.py`.

---

### Paso 3: Modelo CON Dropout

Agregamos capas de Dropout después de cada activación ReLU.

```python
model_with_dropout = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(0.5),      # 50% dropout
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),      # 30% dropout
    nn.Linear(256, 10)
)
```

**Descomenta** la sección del Paso 3 en `starter/main.py`.

---

### Paso 4: Función de Entrenamiento

Implementamos el loop de entrenamiento que registra métricas.

```python
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()  # Activa Dropout
    total_loss, correct, total = 0, 0, 0
    
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (output.argmax(1) == y).sum().item()
        total += y.size(0)
    
    return total_loss / len(train_loader), correct / total
```

**Descomenta** la sección del Paso 4 en `starter/main.py`.

---

### Paso 5: Función de Evaluación

Evaluamos en test set con Dropout desactivado.

```python
def evaluate(model, test_loader, criterion):
    model.eval()  # Desactiva Dropout
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            total_loss += criterion(output, y).item()
            correct += (output.argmax(1) == y).sum().item()
            total += y.size(0)
    
    return total_loss / len(test_loader), correct / total
```

⚠️ **Importante**: `model.eval()` desactiva Dropout durante inferencia.

**Descomenta** la sección del Paso 5 en `starter/main.py`.

---

### Paso 6: Entrenar Ambos Modelos

Entrenamos y comparamos ambos modelos.

```python
epochs = 20
# Entrenar modelo sin dropout
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model_no_dropout, ...)
    test_loss, test_acc = evaluate(model_no_dropout, ...)
    # Guardar métricas...

# Entrenar modelo con dropout
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model_with_dropout, ...)
    test_loss, test_acc = evaluate(model_with_dropout, ...)
    # Guardar métricas...
```

**Descomenta** la sección del Paso 6 en `starter/main.py`.

---

### Paso 7: Visualizar Resultados

Graficamos las curvas de aprendizaje para comparar.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
axes[0].plot(no_dropout_train_accs, label='Sin Dropout - Train')
axes[0].plot(no_dropout_test_accs, label='Sin Dropout - Test')
axes[0].plot(with_dropout_train_accs, label='Con Dropout - Train')
axes[0].plot(with_dropout_test_accs, label='Con Dropout - Test')
axes[0].set_title('Accuracy: Dropout Comparison')
axes[0].legend()
```

**Descomenta** la sección del Paso 7 en `starter/main.py`.

---

## ✅ Criterios de Éxito

| Métrica | Sin Dropout | Con Dropout |
|---------|-------------|-------------|
| Gap Train-Test | > 5% | < 3% |
| Test Accuracy | ~97% | ~98% |
| Overfitting | Visible | Reducido |

---

## 🔍 Preguntas de Reflexión

1. ¿Por qué el gap train-test es menor con Dropout?
2. ¿Qué pasaría si usamos `p=0.9`?
3. ¿Por qué es importante `model.eval()` en inferencia?

---

## 📚 Recursos

- [PyTorch Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
- [Dropout Paper (Srivastava 2014)](https://jmlr.org/papers/v15/srivastava14a.html)
