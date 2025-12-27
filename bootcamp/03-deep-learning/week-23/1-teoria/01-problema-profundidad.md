# 📉 El Problema de la Profundidad en Redes Neuronales

## 🎯 Objetivos

- Comprender por qué las redes muy profundas fallan
- Entender vanishing/exploding gradients
- Conocer el problema de degradación

---

## 1. La Promesa de las Redes Profundas

### Intuición

Más capas = más capacidad de aprender representaciones complejas.

```
Red Superficial (3 capas):
Input → [Conv] → [Conv] → [Conv] → Output
         ↓        ↓        ↓
      Bordes   Texturas  Formas

Red Profunda (20+ capas):
Input → [...muchas capas...] → Output
         ↓
      Características muy abstractas
      (ojos, caras, objetos completos)
```

### El Problema Real

En la práctica, agregar más capas **no siempre mejora** el rendimiento.

```
Accuracy vs Profundidad (sin ResNet):

100% ┤
 95% ┤         ●───●
 90% ┤    ●───●     ╲
 85% ┤   ╱            ╲
 80% ┤  ●              ●───●
 75% ┤ ╱                    ╲
 70% ┼─●───────────────────────●
     └─┬───┬───┬───┬───┬───┬───┬
       8  16  20  32  56  110 152
                Capas
```

---

## 2. Vanishing Gradient

### ¿Qué es?

Durante backpropagation, los gradientes se multiplican capa por capa. Si son < 1, se vuelven exponencialmente pequeños.

### Matemáticamente

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial a_n} \cdot \frac{\partial a_n}{\partial a_{n-1}} \cdot ... \cdot \frac{\partial a_2}{\partial W_1}$$

Si cada término $|\frac{\partial a_i}{\partial a_{i-1}}| < 1$:

$$\text{gradiente} \approx (0.5)^{20} = 0.00000095$$

### Visualización

```
Capa 20:  Gradiente = 1.0        ████████████████████
Capa 15:  Gradiente = 0.1        ██████████
Capa 10:  Gradiente = 0.01       █████
Capa 5:   Gradiente = 0.001      ██
Capa 1:   Gradiente = 0.0001     ▪  (casi cero!)
```

### Consecuencias

- Las primeras capas **no aprenden**
- El modelo se comporta como una red superficial
- Entrenar se vuelve imposible

---

## 3. Exploding Gradient

### El Problema Opuesto

Si los gradientes son > 1, crecen exponencialmente.

$$\text{gradiente} \approx (2.0)^{20} = 1,048,576$$

### Síntomas

```python
# Durante entrenamiento verás:
Epoch 1: Loss = 2.34
Epoch 2: Loss = 15.67
Epoch 3: Loss = nan  # ¡Explotó!
```

### Soluciones Comunes

```python
# 1. Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Inicialización cuidadosa (He, Xavier)
nn.init.kaiming_normal_(layer.weight, mode='fan_out')

# 3. Batch Normalization
nn.BatchNorm2d(num_features)
```

---

## 4. El Problema de Degradación

### Descubrimiento (He et al., 2015)

Incluso con BatchNorm y buena inicialización, redes más profundas tienen **peor** accuracy que redes menos profundas.

> "Esto no es overfitting (training error también es peor)"

### Experimento Original

| Modelo | Train Error | Test Error |
|--------|-------------|------------|
| 20 capas | 5.5% | 8.2% |
| 56 capas | 8.3% | 9.6% |

**¿Por qué?** La red de 56 capas debería al menos igualar a la de 20 (podría aprender identidad en capas extra).

### La Paradoja

```
Si una red de 20 capas es óptima:
- Red de 56 capas podría copiar las 20 primeras
- Y hacer identidad en las 36 restantes: y = x
- Pero esto NO ocurre naturalmente
```

---

## 5. Por Qué la Identidad es Difícil

### Aprender y = x

Parece simple, pero para una red neuronal es sorprendentemente difícil:

```python
# Una capa intenta aprender: y = x
# Tiene que aprender:
W = [[1, 0, 0, ...],
     [0, 1, 0, ...],
     [0, 0, 1, ...]]
b = [0, 0, 0, ...]

# Esto requiere ajuste preciso de muchos parámetros
```

### Con ReLU

```
y = ReLU(Wx + b)

Para que y = x:
- W debe ser identidad
- b debe ser 0
- Y x debe ser > 0 (por ReLU)

¡Muy restrictivo!
```

---

## 6. La Solución: Conexiones Residuales

### Idea Clave de ResNet

En lugar de aprender $H(x)$, aprende el **residuo** $F(x) = H(x) - x$

```
Tradicional:           Residual:
    x                      x ─────────────┐
    │                      │              │
    ▼                      ▼              │
 ┌─────┐                ┌─────┐           │
 │  H  │                │  F  │           │
 └─────┘                └─────┘           │
    │                      │              │
    ▼                      ▼              │
   H(x)                   F(x) + x ◄──────┘
```

### ¿Por Qué Funciona?

**Si la transformación óptima es identidad:**

- Tradicional: Aprender $H(x) = x$ (difícil)
- Residual: Aprender $F(x) = 0$ (¡fácil! solo poner pesos en 0)

### Gradientes

```
Tradicional:
∂L/∂x = ∂L/∂H × ∂H/∂x    (puede → 0)

Residual:
∂L/∂x = ∂L/∂(F+x) × (∂F/∂x + 1)
      = ∂L/∂(F+x) × ∂F/∂x + ∂L/∂(F+x)
                             ↑
                    ¡Siempre hay gradiente directo!
```

---

## 7. Comparativa: Con y Sin Residuales

### Sin Skip Connections (VGG-style)

```
Profundidad  | Train Acc | Test Acc
-------------|-----------|----------
    18       |   95.0%   |  92.0%
    34       |   93.5%   |  90.2%   ← Peor
    50       |   91.8%   |  88.5%   ← Aún peor
```

### Con Skip Connections (ResNet)

```
Profundidad  | Train Acc | Test Acc
-------------|-----------|----------
    18       |   95.5%   |  93.0%
    34       |   96.0%   |  94.5%   ← Mejor
    50       |   96.5%   |  95.2%   ← Aún mejor
   152       |   97.0%   |  95.8%   ← ¡Funciona!
```

---

## 8. Código: Verificar el Problema

```python
import torch
import torch.nn as nn

def check_gradient_flow(model, input_shape):
    """Verifica si hay vanishing gradient."""
    x = torch.randn(1, *input_shape, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    print("Gradientes por capa:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            status = "✓" if grad_norm > 1e-6 else "✗ VANISHING"
            print(f"  {name}: {grad_norm:.2e} {status}")

# Ejemplo con red profunda sin residuales
class DeepCNN(nn.Module):
    def __init__(self, num_layers=20):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = 3 if i == 0 else 64
            layers.append(nn.Conv2d(in_ch, 64, 3, padding=1))
            layers.append(nn.ReLU())
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=[2, 3])  # Global Average Pool
        return self.fc(x)

# Verificar
model = DeepCNN(num_layers=50)
check_gradient_flow(model, (3, 32, 32))
# Verás gradientes muy pequeños en primeras capas
```

---

## ✅ Resumen

| Problema | Causa | Solución |
|----------|-------|----------|
| Vanishing Gradient | Multiplicación de gradientes < 1 | Skip connections, BatchNorm |
| Exploding Gradient | Multiplicación de gradientes > 1 | Gradient clipping, inicialización |
| Degradación | Dificultad para aprender identidad | Conexiones residuales |

**Próximo**: Implementación de ResNet y bloques residuales.

---

## 🔗 Navegación

[← README](../README.md) | [Siguiente: ResNet →](02-resnet-conexiones-residuales.md)
