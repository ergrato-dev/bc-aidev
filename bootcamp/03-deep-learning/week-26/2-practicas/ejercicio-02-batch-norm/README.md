# 📊 Ejercicio 02: Batch Normalization

## 🎯 Objetivo

Implementar Batch Normalization y observar cómo acelera el entrenamiento y estabiliza la convergencia.

---

## 📋 Descripción

En este ejercicio aprenderás:

1. Cómo BatchNorm normaliza activaciones
2. Diferencia train vs eval mode
3. Efecto en velocidad de convergencia
4. Combinación con diferentes learning rates

---

## 🔧 Requisitos

```bash
pip install torch torchvision matplotlib
```

---

## 📝 Pasos del Ejercicio

### Paso 1: Preparar Dataset

Cargamos CIFAR-10 para un problema más desafiante.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10('data', train=False, transform=transform)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: CNN SIN BatchNorm

Creamos una CNN básica sin normalización.

```python
class CNNNoBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: CNN CON BatchNorm

Agregamos BatchNorm después de cada capa convolucional y fully connected.

```python
class CNNWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.bn3(self.fc1(x)))
        return self.fc2(x)
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Funciones de Entrenamiento

```python
def train_epoch(model, loader, criterion, optimizer):
    model.train()  # BatchNorm usa estadísticas del batch
    # ... training loop
    
def evaluate(model, loader, criterion):
    model.eval()   # BatchNorm usa running statistics
    # ... evaluation loop
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Comparar con Diferentes Learning Rates

BatchNorm permite usar learning rates más altos.

```python
# Sin BatchNorm: necesita lr pequeño
optimizer_no_bn = Adam(model_no_bn.parameters(), lr=0.001)

# Con BatchNorm: puede usar lr más alto
optimizer_with_bn = Adam(model_with_bn.parameters(), lr=0.01)
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Visualizar Convergencia

Comparamos las curvas de loss para ver la diferencia en velocidad.

**Descomenta** la sección del Paso 6.

---

## ✅ Criterios de Éxito

| Métrica | Sin BatchNorm | Con BatchNorm |
|---------|---------------|---------------|
| Épocas para 60% acc | ~15 | ~5 |
| LR máximo estable | 0.001 | 0.01 |
| Estabilidad | Variable | Consistente |

---

## 📚 Recursos

- [PyTorch BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
