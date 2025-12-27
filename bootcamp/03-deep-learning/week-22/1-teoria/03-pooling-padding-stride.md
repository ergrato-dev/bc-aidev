# 📐 Pooling, Padding y Stride

## 🎯 Objetivos

- Entender el propósito y tipos de pooling
- Dominar el concepto de padding y sus modos
- Comprender el efecto del stride
- Calcular dimensiones de salida en cualquier configuración

---

## 📋 Contenido

### 1. Pooling (Submuestreo)

El pooling **reduce la dimensionalidad espacial** manteniendo la información más relevante.

#### ¿Por qué Pooling?

| Beneficio | Descripción |
|-----------|-------------|
| **Reduce cómputo** | Menos píxeles = menos operaciones |
| **Reduce parámetros** | Capas siguientes más pequeñas |
| **Invarianza espacial** | Pequeños desplazamientos no afectan |
| **Reduce overfitting** | Menos parámetros = mejor generalización |

---

### 2. Tipos de Pooling

#### 2.1 Max Pooling

Toma el **valor máximo** de cada región:

```
Entrada (4×4):           Max Pool 2×2, stride=2:
┌───┬───┬───┬───┐        ┌───┬───┐
│ 1 │ 3 │ 2 │ 1 │        │ 4 │ 4 │
├───┼───┼───┼───┤   →    ├───┼───┤
│ 2 │ 4 │ 1 │ 4 │        │ 6 │ 5 │
├───┼───┼───┼───┤        └───┴───┘
│ 6 │ 2 │ 3 │ 1 │
├───┼───┼───┼───┤
│ 1 │ 4 │ 2 │ 5 │
└───┴───┴───┴───┘

Región [0:2, 0:2]:       Región [0:2, 2:4]:
[1, 3]                   [2, 1]
[2, 4] → max = 4         [1, 4] → max = 4
```

**Uso**: El más común en CNNs. Preserva features más prominentes.

```python
import torch.nn as nn

max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
# Reduce dimensiones a la mitad
```

#### 2.2 Average Pooling

Calcula el **promedio** de cada región:

```
Entrada:                 Avg Pool 2×2:
┌───┬───┬───┬───┐        ┌─────┬─────┐
│ 1 │ 3 │ 2 │ 2 │        │ 2.5 │ 2.0 │
├───┼───┼───┼───┤   →    ├─────┼─────┤
│ 2 │ 4 │ 2 │ 2 │        │ 3.25│ 2.75│
├───┼───┼───┼───┤        └─────┴─────┘
│ 4 │ 2 │ 3 │ 1 │
├───┼───┼───┼───┤
│ 5 │ 2 │ 4 │ 3 │
└───┴───┴───┴───┘

(1+3+2+4)/4 = 2.5
```

**Uso**: Útil en algunas arquitecturas, menos agresivo que max pooling.

```python
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
```

#### 2.3 Global Pooling

Reduce **todo el feature map a un solo valor** por canal:

```
Entrada (C×H×W):         Global Pool:
┌─────────────┐          ┌───┐
│ Canal 1     │          │v1 │
│  H × W      │    →     ├───┤
├─────────────┤          │v2 │
│ Canal 2     │          ├───┤
│  H × W      │          │...│
└─────────────┘          └───┘
                         (C×1×1)
```

```python
# Global Average Pooling
gap = nn.AdaptiveAvgPool2d(1)  # Output: (batch, channels, 1, 1)

# Global Max Pooling
gmp = nn.AdaptiveMaxPool2d(1)
```

**Uso**: Reemplaza capas FC finales en arquitecturas modernas.

---

### 3. Comparación de Pooling

```python
import torch
import torch.nn as nn

# Crear tensor de ejemplo
x = torch.tensor([
    [1., 3., 2., 1.],
    [2., 4., 1., 4.],
    [6., 2., 3., 1.],
    [1., 4., 2., 5.]
]).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 4, 4)

# Aplicar diferentes poolings
max_pool = nn.MaxPool2d(2, 2)
avg_pool = nn.AvgPool2d(2, 2)
global_avg = nn.AdaptiveAvgPool2d(1)
global_max = nn.AdaptiveMaxPool2d(1)

print(f"Input:\n{x.squeeze()}")
print(f"\nMax Pool 2×2:\n{max_pool(x).squeeze()}")
print(f"\nAvg Pool 2×2:\n{avg_pool(x).squeeze()}")
print(f"\nGlobal Avg: {global_avg(x).item():.2f}")
print(f"Global Max: {global_max(x).item():.2f}")
```

---

### 4. Padding

El padding **añade valores alrededor de la imagen** para controlar el tamaño de salida.

#### 4.1 Sin Padding (Valid)

```
Entrada 5×5:             Kernel 3×3:        Salida 3×3:
┌───────────────┐        ┌─────────┐        ┌─────────┐
│ x x x x x     │        │ k k k   │        │ o o o   │
│ x x x x x     │   *    │ k k k   │   =    │ o o o   │
│ x x x x x     │        │ k k k   │        │ o o o   │
│ x x x x x     │        └─────────┘        └─────────┘
│ x x x x x     │        
└───────────────┘        5 - 3 + 1 = 3
```

#### 4.2 Same Padding

Añade padding para que **salida = entrada**:

```
Entrada 5×5 + pad=1:     Kernel 3×3:        Salida 5×5:
┌─────────────────┐      ┌─────────┐        ┌───────────┐
│ 0 0 0 0 0 0 0   │      │ k k k   │        │ o o o o o │
│ 0 x x x x x 0   │  *   │ k k k   │   =    │ o o o o o │
│ 0 x x x x x 0   │      │ k k k   │        │ o o o o o │
│ 0 x x x x x 0   │      └─────────┘        │ o o o o o │
│ 0 x x x x x 0   │                         │ o o o o o │
│ 0 x x x x x 0   │                         └───────────┘
│ 0 0 0 0 0 0 0   │      
└─────────────────┘      (5 - 3 + 2×1)/1 + 1 = 5
```

#### Calcular Padding para Same

```python
def same_padding(kernel_size: int) -> int:
    """Calcula padding para mantener dimensiones."""
    return kernel_size // 2

# Ejemplos
print(same_padding(3))  # 1 (para kernel 3×3)
print(same_padding(5))  # 2 (para kernel 5×5)
print(same_padding(7))  # 3 (para kernel 7×7)
```

#### En PyTorch

```python
import torch.nn as nn

# Padding explícito
conv_valid = nn.Conv2d(1, 32, kernel_size=3, padding=0)  # Valid
conv_same = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # Same

# Padding automático (PyTorch 1.9+)
conv_auto = nn.Conv2d(1, 32, kernel_size=3, padding='same')
```

---

### 5. Stride

El stride **controla el salto del kernel** entre posiciones.

#### Stride = 1 (Default)

```
Posición 1:    Posición 2:    Posición 3:
[x x x]○ ○    ○[x x x]○    ○ ○[x x x]
[x x x]○ ○    ○[x x x]○    ○ ○[x x x]
[x x x]○ ○    ○[x x x]○    ○ ○[x x x]
 ○ ○ ○ ○ ○     ○ ○ ○ ○ ○     ○ ○ ○ ○ ○
 ○ ○ ○ ○ ○     ○ ○ ○ ○ ○     ○ ○ ○ ○ ○

Se mueve 1 píxel a la vez
```

#### Stride = 2

```
Posición 1:    Posición 2:    (Sin posición 3)
[x x x]○ ○    ○ ○[x x x]
[x x x]○ ○    ○ ○[x x x]
[x x x]○ ○    ○ ○[x x x]
 ○ ○ ○ ○ ○     ○ ○ ○ ○ ○
 ○ ○ ○ ○ ○     ○ ○ ○ ○ ○

Se mueve 2 píxeles a la vez → reduce a la mitad
```

#### Efecto en Dimensiones

```python
def output_with_stride(W: int, K: int, P: int, S: int) -> int:
    """Calcula dimensión de salida."""
    return (W - K + 2 * P) // S + 1

# Imagen 32×32, kernel 3×3
print(output_with_stride(32, 3, 0, 1))  # 30 (stride=1)
print(output_with_stride(32, 3, 0, 2))  # 15 (stride=2)
print(output_with_stride(32, 3, 1, 1))  # 32 (same padding)
print(output_with_stride(32, 3, 1, 2))  # 16 (same + stride=2)
```

---

### 6. Stride vs Pooling para Reducir Dimensiones

Ambos reducen dimensionalidad, pero de forma diferente:

| Aspecto | Max Pooling | Stride > 1 |
|---------|-------------|------------|
| **Operación** | Selecciona máximo | Salta posiciones |
| **Parámetros** | 0 | Mismos que conv normal |
| **Información** | Preserva máximos | Puede perder info |
| **Uso moderno** | Tradicional | ResNet, arquitecturas recientes |

```python
# Reducir de 32×32 a 16×16

# Opción 1: Max Pooling
model_pool = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),  # 32×32
    nn.ReLU(),
    nn.MaxPool2d(2, 2)               # 16×16
)

# Opción 2: Strided Convolution
model_stride = nn.Sequential(
    nn.Conv2d(3, 64, 3, stride=2, padding=1),  # 16×16 directamente
    nn.ReLU()
)
```

---

### 7. Ejemplos Prácticos

#### Calcular Dimensiones de una CNN

```python
import torch
import torch.nn as nn

class ExampleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Entrada: 1×32×32
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # → 32×32×32
        self.pool1 = nn.MaxPool2d(2, 2)               # → 32×16×16
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # → 64×16×16
        self.pool2 = nn.MaxPool2d(2, 2)               # → 64×8×8
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # → 128×8×8
        self.pool3 = nn.MaxPool2d(2, 2)               # → 128×4×4
        
        # 128 × 4 × 4 = 2048
        self.fc = nn.Linear(128 * 4 * 4, 10)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Verificar dimensiones
model = ExampleCNN()
x = torch.randn(1, 1, 32, 32)
print(f"Input: {x.shape}")

# Ver cada paso
x1 = model.pool1(torch.relu(model.conv1(x)))
print(f"After conv1+pool1: {x1.shape}")

x2 = model.pool2(torch.relu(model.conv2(x1)))
print(f"After conv2+pool2: {x2.shape}")

x3 = model.pool3(torch.relu(model.conv3(x2)))
print(f"After conv3+pool3: {x3.shape}")
```

#### Tabla de Dimensiones

```
Capa            | Output Shape | Params
----------------|--------------|--------
Input           | 1×32×32      | -
Conv1 (32, 3×3) | 32×32×32     | 320
Pool1 (2×2)     | 32×16×16     | 0
Conv2 (64, 3×3) | 64×16×16     | 18,496
Pool2 (2×2)     | 64×8×8       | 0
Conv3 (128,3×3) | 128×8×8      | 73,856
Pool3 (2×2)     | 128×4×4      | 0
Flatten         | 2048         | 0
FC (10)         | 10           | 20,490
----------------|--------------|--------
Total           |              | 113,162
```

---

### 8. Padding Modes en PyTorch

```python
import torch.nn.functional as F

# Zeros (default)
x_zeros = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)

# Reflect
x_reflect = F.pad(x, (1, 1, 1, 1), mode='reflect')

# Replicate
x_replicate = F.pad(x, (1, 1, 1, 1), mode='replicate')

# Circular
x_circular = F.pad(x, (1, 1, 1, 1), mode='circular')
```

```
Original:        Zeros:           Reflect:         Replicate:
[a b c]          [0 a b c 0]      [b a b c b]      [a a b c c]
[d e f]    →     [0 d e f 0]      [e d e f e]      [d d e f f]
[g h i]          [0 g h i 0]      [h g h i h]      [g g h i i]
```

---

## 📊 Resumen de Fórmulas

### Tamaño de Salida

$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

### Padding para Same (stride=1)

$$P = \frac{K - 1}{2}$$

### Reducción por Pooling

$$O = \frac{W}{k}$$ (con stride = kernel_size)

---

## ✅ Checklist de Verificación

- [ ] Entiendo la diferencia entre max y average pooling
- [ ] Sé cuándo usar global pooling
- [ ] Puedo calcular padding para mantener dimensiones
- [ ] Entiendo el efecto del stride
- [ ] Puedo calcular las dimensiones de salida de cualquier capa

---

_Siguiente: [Arquitecturas Clásicas (LeNet, VGG)](04-arquitecturas-clasicas.md)_
