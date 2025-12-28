# 🏃 Ejercicio 01: Comparación de Optimizadores

## 🎯 Objetivo

Comparar el rendimiento de diferentes optimizadores (SGD, SGD+Momentum, Adam, AdamW) entrenando el mismo modelo en MNIST.

---

## 📋 Instrucciones

### Paso 1: Configuración Inicial

Primero importamos las librerías necesarias y configuramos el dispositivo:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
```

**Abre `starter/main.py`** y descomenta la sección de imports.

---

### Paso 2: Definir el Modelo

Usaremos una red simple para que las diferencias entre optimizadores sean más evidentes:

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
```

**Descomenta** la definición del modelo en `starter/main.py`.

---

### Paso 3: Cargar Datos

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)
```

**Descomenta** la sección de carga de datos.

---

### Paso 4: Función de Entrenamiento

Esta función entrena un modelo y registra la pérdida por época:

```python
def train_with_optimizer(optimizer_name, optimizer, model, epochs=5):
    criterion = nn.CrossEntropyLoss()
    history = {'loss': [], 'acc': [], 'time': 0}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            correct += (output.argmax(1) == y).sum().item()
            total += y.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        
        print(f'{optimizer_name} - Epoch {epoch+1}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}')
    
    history['time'] = time.time() - start_time
    return history
```

**Descomenta** la función de entrenamiento.

---

### Paso 5: Comparar Optimizadores

Ahora entrenamos con cada optimizador y guardamos los resultados:

```python
# Configuración de optimizadores a comparar
optimizers_config = {
    'SGD': lambda model: optim.SGD(model.parameters(), lr=0.01),
    'SGD+Momentum': lambda model: optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'Adam': lambda model: optim.Adam(model.parameters(), lr=0.001),
    'AdamW': lambda model: optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01),
}

results = {}

for name, opt_fn in optimizers_config.items():
    print(f'\n{"="*50}')
    print(f'Entrenando con {name}')
    print("="*50)
    
    # Crear modelo nuevo para cada optimizador
    model = SimpleNet().to(device)
    optimizer = opt_fn(model)
    
    results[name] = train_with_optimizer(name, optimizer, model, epochs=5)
```

**Descomenta** la sección de comparación.

---

### Paso 6: Visualizar Resultados

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfica de Loss
for name, history in results.items():
    axes[0].plot(history['loss'], label=name, marker='o')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Comparación de Loss por Optimizador')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfica de Accuracy
for name, history in results.items():
    axes[1].plot(history['acc'], label=name, marker='o')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Comparación de Accuracy por Optimizador')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimizers_comparison.png', dpi=150)
plt.show()

# Tabla resumen
print('\n' + '='*60)
print('RESUMEN DE RESULTADOS')
print('='*60)
print(f'{"Optimizador":<15} {"Loss Final":<12} {"Acc Final":<12} {"Tiempo (s)":<10}')
print('-'*60)
for name, history in results.items():
    print(f'{name:<15} {history["loss"][-1]:<12.4f} {history["acc"][-1]:<12.4f} {history["time"]:<10.2f}')
```

**Descomenta** la sección de visualización.

---

## 🔍 Preguntas de Reflexión

1. ¿Cuál optimizador convergió más rápido?
2. ¿Cuál alcanzó mejor accuracy final?
3. ¿Por qué SGD sin momentum oscila más?
4. ¿En qué casos preferirías AdamW sobre Adam?

---

## ✅ Checklist

- [ ] Imports y configuración funcionando
- [ ] Modelo definido correctamente
- [ ] Datos cargados
- [ ] Función de entrenamiento implementada
- [ ] Comparación de 4 optimizadores ejecutada
- [ ] Gráficas generadas
- [ ] Preguntas de reflexión respondidas

---

## 📚 Recursos

- [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html)
- [Adam Paper](https://arxiv.org/abs/1412.6980)
- [AdamW Paper](https://arxiv.org/abs/1711.05101)
