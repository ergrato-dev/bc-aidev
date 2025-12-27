# 📖 Glosario - Semana 22: CNNs I

Términos clave ordenados alfabéticamente.

---

## A

### Activation Map
Ver **Feature Map**.

### Average Pooling
Operación de pooling que calcula el promedio de valores en una ventana. Menos común que Max Pooling pero útil para ciertas aplicaciones.

```python
nn.AvgPool2d(kernel_size=2, stride=2)
```

---

## B

### Batch Normalization (BatchNorm)
Técnica de normalización que estandariza las activaciones de cada capa durante el entrenamiento. Acelera convergencia y permite learning rates más altos.

```python
# Después de Conv2d
nn.BatchNorm2d(num_features)  # num_features = canales
```

**Fórmula:**
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

---

## C

### Canal (Channel)
Dimensión de profundidad en una imagen o feature map. Imágenes RGB tienen 3 canales; feature maps pueden tener cientos.

### Convolución
Operación matemática que aplica un filtro/kernel sobre una imagen, produciendo un mapa de características.

**Fórmula 2D:**
$$(I * K)[i,j] = \sum_m \sum_n I[i+m, j+n] \cdot K[m, n]$$

### Conv2d
Capa convolucional 2D en PyTorch.

```python
nn.Conv2d(
    in_channels,   # Canales entrada
    out_channels,  # Número de filtros
    kernel_size,   # Tamaño del kernel
    stride=1,      # Paso
    padding=0      # Relleno
)
```

---

## D

### Dilation
Separación entre elementos del kernel. Dilation > 1 aumenta el campo receptivo sin aumentar parámetros.

### Downsampling
Reducción de las dimensiones espaciales de un tensor, típicamente mediante pooling o stride > 1.

### Dropout
Técnica de regularización que desactiva neuronas aleatoriamente durante entrenamiento.

```python
nn.Dropout(p=0.5)      # Para capas FC
nn.Dropout2d(p=0.25)   # Para feature maps
```

---

## F

### Feature Map
Salida de una capa convolucional. Representa características detectadas (bordes, texturas, formas).

### Filtro
Ver **Kernel**.

### Flatten
Operación que convierte un tensor multidimensional en un vector 1D.

```python
nn.Flatten()  # (B, C, H, W) -> (B, C*H*W)
```

---

## G

### Global Average Pooling (GAP)
Pooling que reduce cada feature map a un único valor promediando todos los elementos. Elimina la necesidad de capas FC grandes.

```python
nn.AdaptiveAvgPool2d(1)  # Output: (B, C, 1, 1)
```

---

## K

### Kernel (Filtro)
Matriz de pesos que se desliza sobre la imagen en una convolución. Detecta patrones específicos como bordes o texturas.

**Ejemplos comunes (3×3):**
- Sobel (bordes verticales)
- Laplaciano (detección de bordes)
- Gaussian (suavizado)

---

## L

### LeNet-5
Primera CNN exitosa (LeCun, 1998). Arquitectura:
- Input 32×32
- 2 capas conv + pool
- 3 capas FC
- ~61,000 parámetros

---

## M

### Max Pooling
Operación que selecciona el valor máximo de una ventana. Reduce dimensiones y proporciona invariancia a pequeñas traslaciones.

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

---

## O

### Output Size (Tamaño de Salida)
Fórmula para calcular dimensiones después de convolución:

$$O = \frac{W - K + 2P}{S} + 1$$

Donde:
- $W$ = tamaño de entrada
- $K$ = tamaño del kernel
- $P$ = padding
- $S$ = stride

---

## P

### Padding
Relleno añadido alrededor de la imagen antes de convolución.

| Tipo | Descripción |
|------|-------------|
| `valid` | Sin padding (output más pequeño) |
| `same` | Padding para mantener tamaño |

```python
nn.Conv2d(..., padding='same')  # Mantiene dimensiones
nn.Conv2d(..., padding=1)       # Padding explícito
```

### Pooling
Operación de reducción de dimensionalidad que resume regiones del feature map.

---

## R

### Receptive Field (Campo Receptivo)
Región de la imagen de entrada que influye en una neurona específica de una capa posterior. Crece con la profundidad de la red.

### ReLU (Rectified Linear Unit)
Función de activación: $f(x) = \max(0, x)$

```python
nn.ReLU(inplace=True)
```

---

## S

### Stride
Paso o desplazamiento del kernel en cada movimiento. Stride > 1 reduce dimensiones.

```python
nn.Conv2d(..., stride=2)  # Reduce tamaño a la mitad
```

### Subsampling
Ver **Downsampling** o **Pooling**.

---

## T

### Transfer Learning
Técnica de usar modelos pre-entrenados (ej: VGG en ImageNet) y adaptarlos a nuevas tareas.

---

## V

### VGG-16
Arquitectura profunda (Simonyan & Zisserman, 2014) con:
- 13 capas conv (3×3)
- 3 capas FC
- ~138 millones de parámetros

Demostró que la profundidad mejora el rendimiento.

---

## W

### Weight Sharing
Característica de CNNs donde el mismo kernel se aplica en todas las posiciones de la imagen, reduciendo drásticamente el número de parámetros.

---

## Fórmulas Clave

| Operación | Fórmula |
|-----------|---------|
| Output size | $O = \frac{W - K + 2P}{S} + 1$ |
| Parámetros Conv | $K^2 \times C_{in} \times C_{out} + C_{out}$ |
| BatchNorm | $\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$ |

---

## 🔗 Navegación

[← Volver a la Semana](../README.md)
