# 🔲 Ejercicio 01: Convolución Manual

## 🎯 Objetivo

Implementar la operación de convolución 2D desde cero para entender su funcionamiento interno.

---

## 📋 Instrucciones

En este ejercicio implementarás convolución 2D manualmente usando NumPy, sin usar funciones de deep learning.

---

## Paso 1: Convolución Básica Sin Padding

La convolución desliza un kernel sobre la imagen, multiplicando elemento a elemento y sumando.

```python
import numpy as np

def conv2d_basic(image, kernel):
    """
    Convolución 2D básica sin padding.
    
    Args:
        image: Array 2D (H, W)
        kernel: Array 2D (Kh, Kw)
    
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
            region = image[i:i+Kh, j:j+Kw]
            output[i, j] = np.sum(region * kernel)
    
    return output
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1 para probar.

---

## Paso 2: Convolución con Padding

El padding añade ceros alrededor de la imagen para controlar el tamaño de salida.

```python
def conv2d_padding(image, kernel, padding=1):
    """
    Convolución 2D con zero padding.
    
    Args:
        image: Array 2D (H, W)
        kernel: Array 2D (Kh, Kw)
        padding: Cantidad de padding
    
    Returns:
        Feature map con padding aplicado
    """
    # Añadir padding
    padded = np.pad(image, padding, mode='constant', constant_values=0)
    
    # Aplicar convolución básica
    return conv2d_basic(padded, kernel)
```

**Descomenta** la sección del Paso 2 en `starter/main.py`.

---

## Paso 3: Convolución con Stride

El stride controla el salto del kernel entre posiciones.

```python
def conv2d_stride(image, kernel, stride=1):
    """
    Convolución 2D con stride.
    
    Args:
        image: Array 2D (H, W)
        kernel: Array 2D (Kh, Kw)
        stride: Paso del kernel
    
    Returns:
        Feature map con stride aplicado
    """
    H, W = image.shape
    Kh, Kw = kernel.shape
    
    out_h = (H - Kh) // stride + 1
    out_w = (W - Kw) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            region = image[h_start:h_start+Kh, w_start:w_start+Kw]
            output[i, j] = np.sum(region * kernel)
    
    return output
```

**Descomenta** la sección del Paso 3 en `starter/main.py`.

---

## Paso 4: Kernels de Detección de Bordes

Diferentes kernels detectan diferentes características:

```python
# Kernel para bordes verticales
kernel_vertical = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# Kernel para bordes horizontales
kernel_horizontal = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

# Kernel Sobel X
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Kernel de enfoque (sharpen)
kernel_sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])
```

**Descomenta** la sección del Paso 4 para aplicar estos filtros a una imagen.

---

## Paso 5: Visualización de Resultados

```python
import matplotlib.pyplot as plt

def visualize_convolution(image, kernel, title="Convolución"):
    """Visualiza imagen original y resultado de convolución."""
    result = conv2d_padding(image, kernel, padding=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    axes[1].imshow(kernel, cmap='RdBu', vmin=-2, vmax=2)
    axes[1].set_title('Kernel')
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            axes[1].text(j, i, f'{kernel[i,j]:.0f}', 
                        ha='center', va='center', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(result, cmap='gray')
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
```

**Descomenta** la sección del Paso 5 para visualizar los resultados.

---

## Paso 6: Convolución Completa con Padding y Stride

Combina todas las funcionalidades:

```python
def conv2d_full(image, kernel, padding=0, stride=1):
    """
    Convolución 2D completa con padding y stride.
    
    Args:
        image: Array 2D (H, W)
        kernel: Array 2D (Kh, Kw)
        padding: Zero padding
        stride: Paso del kernel
    
    Returns:
        Feature map resultante
    """
    # Aplicar padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    
    H, W = image.shape
    Kh, Kw = kernel.shape
    
    out_h = (H - Kh) // stride + 1
    out_w = (W - Kw) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            region = image[h_start:h_start+Kh, w_start:w_start+Kw]
            output[i, j] = np.sum(region * kernel)
    
    return output
```

**Descomenta** la sección del Paso 6 para probar la función completa.

---

## ✅ Verificación

Al completar el ejercicio deberías poder:

- [ ] Implementar convolución 2D desde cero
- [ ] Aplicar padding y stride
- [ ] Usar diferentes kernels para detectar bordes
- [ ] Visualizar los resultados de la convolución

---

## 🔗 Navegación

[← Teoría](../../1-teoria/02-operacion-convolucion.md) | [Siguiente Ejercicio →](../ejercicio-02-cnn-pytorch/)
