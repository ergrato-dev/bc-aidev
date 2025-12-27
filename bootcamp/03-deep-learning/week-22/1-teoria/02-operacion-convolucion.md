# 🔲 La Operación de Convolución

## 🎯 Objetivos

- Comprender matemáticamente la operación de convolución 2D
- Entender el rol de kernels/filtros
- Calcular dimensiones de salida
- Implementar convolución manualmente

---

## 📋 Contenido

### 1. ¿Qué es una Convolución?

La convolución es una operación matemática que combina dos funciones para producir una tercera. En CNNs, aplicamos un **kernel** (filtro pequeño) sobre una **imagen** para producir un **feature map**.

```
     Imagen          Kernel         Feature Map
    ┌───────────┐    ┌─────┐       ┌─────────┐
    │ 1 2 3 4 5 │    │ 1 0 │       │ ? ? ? ? │
    │ 6 7 8 9 0 │  * │ 0 1 │   =   │ ? ? ? ? │
    │ 1 2 3 4 5 │    └─────┘       │ ? ? ? ? │
    │ 6 7 8 9 0 │                  │ ? ? ? ? │
    └───────────┘                  └─────────┘
```

---

### 2. El Proceso Paso a Paso

#### Paso 1: Posicionar el Kernel

```
Imagen (5×5):                Kernel (3×3):
┌─────────────────┐          ┌─────────┐
│[1][2][3] 4  5   │          │ 1  0  1 │
│[4][5][6] 7  8   │    *     │ 0  1  0 │
│[7][8][9] 0  1   │          │ 1  0  1 │
│ 2  3  4  5  6   │          └─────────┘
│ 7  8  9  0  1   │
└─────────────────┘
```

#### Paso 2: Multiplicar Elemento a Elemento

```
Región:          Kernel:         Multiplicación:
1  2  3          1  0  1         1×1  2×0  3×1
4  5  6    ×     0  1  0    =    4×0  5×1  6×0
7  8  9          1  0  1         7×1  8×0  9×1
```

#### Paso 3: Sumar Todos los Valores

```
Resultado = 1×1 + 2×0 + 3×1 + 4×0 + 5×1 + 6×0 + 7×1 + 8×0 + 9×1
         = 1 + 0 + 3 + 0 + 5 + 0 + 7 + 0 + 9
         = 25
```

#### Paso 4: Deslizar y Repetir

```
Posición 1:        Posición 2:        Posición 3:
[1 2 3]4 5        1[2 3 4]5         1 2[3 4 5]
[4 5 6]7 8   →   4[5 6 7]8    →   4 5[6 7 8]
[7 8 9]0 1        7[8 9 0]1         7 8[9 0 1]
   ↓                 ↓                  ↓
  25                ?                   ?
```

---

### 3. Fórmula Matemática

Para una imagen $I$ y kernel $K$, la convolución en posición $(i, j)$:

$$
(I * K)_{i,j} = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} I_{i+m, j+n} \cdot K_{m,n}
$$

Donde:
- $k_h, k_w$: altura y ancho del kernel
- $I_{i+m, j+n}$: valor del píxel en la imagen
- $K_{m,n}$: valor del peso en el kernel

---

### 4. Tamaño de Salida

#### Sin Padding ni Stride

$$O = W - K + 1$$

```python
# Ejemplo: imagen 28×28, kernel 3×3
W, K = 28, 3
O = W - K + 1  # = 26

# El feature map será 26×26
```

#### Con Padding y Stride

$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

```python
def output_size(W: int, K: int, P: int = 0, S: int = 1) -> int:
    """Calcula tamaño de salida de convolución."""
    return (W - K + 2 * P) // S + 1

# Ejemplos
print(output_size(28, 3, P=0, S=1))  # 26 (sin padding)
print(output_size(28, 3, P=1, S=1))  # 28 (same padding)
print(output_size(28, 3, P=0, S=2))  # 13 (stride 2)
```

---

### 5. Kernels como Detectores de Features

Diferentes kernels detectan diferentes características:

#### Detector de Bordes Vertical

```python
import numpy as np

kernel_vertical = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])
# Detecta transiciones izquierda-derecha
```

#### Detector de Bordes Horizontal

```python
kernel_horizontal = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])
# Detecta transiciones arriba-abajo
```

#### Detector de Bordes (Sobel)

```python
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])
```

#### Filtro de Enfoque (Sharpen)

```python
kernel_sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])
```

#### Filtro de Desenfoque (Blur)

```python
kernel_blur = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]) / 9  # Promedio
```

---

### 6. Implementación Manual

```python
import numpy as np

def conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Aplica convolución 2D sin padding.
    
    Args:
        image: Imagen de entrada (H, W)
        kernel: Kernel/filtro (Kh, Kw)
    
    Returns:
        Feature map resultante
    """
    H, W = image.shape
    Kh, Kw = kernel.shape
    
    # Calcular dimensiones de salida
    out_h = H - Kh + 1
    out_w = W - Kw + 1
    
    # Inicializar salida
    output = np.zeros((out_h, out_w))
    
    # Aplicar convolución
    for i in range(out_h):
        for j in range(out_w):
            # Extraer región
            region = image[i:i+Kh, j:j+Kw]
            # Multiplicar y sumar
            output[i, j] = np.sum(region * kernel)
    
    return output

# Ejemplo de uso
image = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 0],
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 0],
    [1, 2, 3, 4, 5]
], dtype=float)

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=float)

result = conv2d(image, kernel)
print(f"Input shape: {image.shape}")
print(f"Kernel shape: {kernel.shape}")
print(f"Output shape: {result.shape}")
print(f"Result:\n{result}")
```

---

### 7. Convolución con Múltiples Canales

Las imágenes RGB tienen 3 canales. La convolución se extiende:

```
Imagen RGB (H×W×3):     Kernel (K×K×3):      Feature Map (H'×W'):
┌───────────────┐       ┌─────────┐          ┌───────────┐
│ R │ G │ B     │       │Kr│Kg│Kb│          │   Suma    │
│ H │ H │ H     │   *   │ K│ K│ K│    =     │   de los  │
│ × │ × │ ×     │       │ ×│ ×│ ×│          │  3 canales│
│ W │ W │ W     │       │ K│ K│ K│          │           │
└───────────────┘       └─────────┘          └───────────┘
```

```python
def conv2d_multichannel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolución 2D para imagen con múltiples canales.
    
    Args:
        image: (H, W, C) imagen con C canales
        kernel: (Kh, Kw, C) kernel con C canales
    
    Returns:
        Feature map (H', W')
    """
    H, W, C = image.shape
    Kh, Kw, Kc = kernel.shape
    assert C == Kc, "Canales deben coincidir"
    
    out_h = H - Kh + 1
    out_w = W - Kw + 1
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            # Extraer región 3D
            region = image[i:i+Kh, j:j+Kw, :]
            # Suma sobre todos los canales
            output[i, j] = np.sum(region * kernel)
    
    return output
```

---

### 8. Múltiples Filtros = Múltiples Feature Maps

```python
# En una capa Conv2d típica:
# - Entrada: (batch, C_in, H, W)
# - Pesos: (C_out, C_in, Kh, Kw) → C_out filtros
# - Salida: (batch, C_out, H', W')

import torch.nn as nn

# 3 canales de entrada (RGB), 64 filtros de salida, kernel 3×3
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)

# Esto crea 64 filtros, cada uno de 3×3×3
# Total parámetros: 64 × (3 × 3 × 3) + 64 bias = 1,792
```

---

### 9. Convolución en PyTorch

```python
import torch
import torch.nn as nn

# Crear capa convolucional
conv = nn.Conv2d(
    in_channels=1,      # Canales de entrada (grayscale=1, RGB=3)
    out_channels=32,    # Número de filtros
    kernel_size=3,      # Tamaño del kernel (3×3)
    stride=1,           # Paso del deslizamiento
    padding=1           # Padding para mantener tamaño
)

# Entrada: (batch, canales, alto, ancho)
x = torch.randn(4, 1, 28, 28)

# Aplicar convolución
output = conv(x)
print(f"Input: {x.shape}")      # [4, 1, 28, 28]
print(f"Output: {output.shape}")  # [4, 32, 28, 28]

# Ver los pesos
print(f"Kernel shape: {conv.weight.shape}")  # [32, 1, 3, 3]
print(f"Bias shape: {conv.bias.shape}")      # [32]
```

---

### 10. Visualización de Feature Maps

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Cargar imagen
image = Image.open('imagen.jpg').convert('L')  # Grayscale
transform = transforms.ToTensor()
x = transform(image).unsqueeze(0)  # Añadir batch dimension

# Crear filtros conocidos
kernels = {
    'vertical': torch.tensor([[[-1., 0., 1.], 
                               [-1., 0., 1.], 
                               [-1., 0., 1.]]]),
    'horizontal': torch.tensor([[[-1., -1., -1.], 
                                 [0., 0., 0.], 
                                 [1., 1., 1.]]]),
    'blur': torch.ones(1, 1, 3, 3) / 9,
}

# Aplicar cada filtro
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(x.squeeze(), cmap='gray')
axes[0].set_title('Original')

for idx, (name, kernel) in enumerate(kernels.items(), 1):
    conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    conv.weight.data = kernel.unsqueeze(0)
    
    with torch.no_grad():
        output = conv(x)
    
    axes[idx].imshow(output.squeeze(), cmap='gray')
    axes[idx].set_title(name)

plt.tight_layout()
plt.show()
```

---

## 📊 Resumen de Parámetros

| Parámetro | Símbolo | Descripción |
|-----------|---------|-------------|
| Kernel size | $K$ | Tamaño del filtro (típico: 3, 5, 7) |
| Stride | $S$ | Paso del deslizamiento (típico: 1, 2) |
| Padding | $P$ | Píxeles añadidos al borde |
| In channels | $C_{in}$ | Canales de entrada |
| Out channels | $C_{out}$ | Número de filtros |

### Número de Parámetros

$$\text{Params} = C_{out} \times (K \times K \times C_{in} + 1)$$

```python
def count_conv_params(in_ch: int, out_ch: int, kernel: int) -> int:
    """Cuenta parámetros de una capa Conv2d."""
    weights = out_ch * in_ch * kernel * kernel
    biases = out_ch
    return weights + biases

# Ejemplo
params = count_conv_params(3, 64, 3)  # RGB → 64 filtros, 3×3
print(f"Parámetros: {params}")  # 1,792
```

---

## ✅ Checklist de Verificación

- [ ] Puedo explicar paso a paso la operación de convolución
- [ ] Sé calcular el tamaño de salida con cualquier padding/stride
- [ ] Entiendo cómo diferentes kernels detectan diferentes features
- [ ] Puedo implementar convolución 2D manualmente
- [ ] Sé usar `nn.Conv2d` en PyTorch

---

_Siguiente: [Pooling, Padding y Stride](03-pooling-padding-stride.md)_
