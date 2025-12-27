# 🧠 Ejercicio 03: Red Neuronal Manual

## 🎯 Objetivo

Construir una red neuronal completa usando `nn.Module` e implementar el training loop manualmente.

---

## 📋 Instrucciones

Este ejercicio te guiará para construir tu primera red neuronal en PyTorch. Abre `starter/main.py` y descomenta cada sección.

---

## Paso 1: Estructura de nn.Module

La clase base para redes neuronales en PyTorch:

```python
import torch.nn as nn

class MiRed(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc1(x)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

## Paso 2: Definir Arquitectura

Construye una red con múltiples capas:

```python
class RedNeuronal(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

**Descomenta** la sección del Paso 2.

---

## Paso 3: Inspeccionar el Modelo

Ver parámetros y estructura:

```python
model = RedNeuronal(784, 128, 10)

# Ver estructura
print(model)

# Ver parámetros
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

**Descomenta** la sección del Paso 3.

---

## Paso 4: Loss y Optimizer

Configurar función de pérdida y optimizador:

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Descomenta** la sección del Paso 4.

---

## Paso 5: Training Loop

El ciclo de entrenamiento completo:

```python
for epoch in range(epochs):
    optimizer.zero_grad()       # 1. Limpiar gradientes
    output = model(data)        # 2. Forward pass
    loss = criterion(output, target)  # 3. Calcular loss
    loss.backward()             # 4. Backward pass
    optimizer.step()            # 5. Actualizar pesos
```

**Descomenta** la sección del Paso 5.

---

## Paso 6: train() vs eval()

Cambiar entre modos de entrenamiento y evaluación:

```python
model.train()  # Activa dropout, BN usa stats del batch
model.eval()   # Desactiva dropout, BN usa stats guardadas

with torch.no_grad():
    predictions = model(test_data)
```

**Descomenta** la sección del Paso 6.

---

## Paso 7: Guardar y Cargar

Persistir modelos entrenados:

```python
# Guardar
torch.save(model.state_dict(), 'modelo.pth')

# Cargar
model.load_state_dict(torch.load('modelo.pth'))
```

**Descomenta** la sección del Paso 7.

---

## ✅ Verificación

Al completar todos los pasos, deberías:
- Crear clases que hereden de `nn.Module`
- Implementar el training loop completo
- Entender `train()` vs `eval()`
- Guardar y cargar modelos

---

## 📚 Recursos

- [nn.Module Tutorial](https://pytorch.org/tutorials/beginner/nn_tutorial.html)
- [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
