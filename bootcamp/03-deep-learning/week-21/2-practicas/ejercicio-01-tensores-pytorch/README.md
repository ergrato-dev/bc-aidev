# 🔢 Ejercicio 01: Tensores en PyTorch

## 🎯 Objetivo

Dominar la creación y manipulación de tensores en PyTorch, la estructura de datos fundamental para deep learning.

---

## 📋 Instrucciones

Este ejercicio te guiará a través de las operaciones básicas con tensores. Abre `starter/main.py` y descomenta cada sección según avances.

---

## Paso 1: Creación de Tensores

Los tensores son arrays multidimensionales, similares a NumPy pero con soporte GPU:

```python
import torch

# Desde lista Python
t1 = torch.tensor([1, 2, 3, 4])

# Tensor de ceros/unos
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)

# Tensor aleatorio
rand = torch.rand(3, 3)  # Uniforme [0, 1)
randn = torch.randn(3, 3)  # Normal (0, 1)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

## Paso 2: Atributos de Tensores

Cada tensor tiene propiedades importantes:

```python
t = torch.rand(2, 3, 4)

print(t.shape)      # torch.Size([2, 3, 4])
print(t.dtype)      # torch.float32
print(t.device)     # cpu
print(t.dim())      # 3 (dimensiones)
print(t.numel())    # 24 (total elementos)
```

**Descomenta** la sección del Paso 2.

---

## Paso 3: Operaciones Matemáticas

PyTorch soporta operaciones element-wise y matriciales:

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise
suma = a + b        # [5, 7, 9]
mult = a * b        # [4, 10, 18]

# Matriciales
A = torch.rand(2, 3)
B = torch.rand(3, 4)
C = A @ B           # Multiplicación de matrices [2, 4]
```

**Descomenta** la sección del Paso 3.

---

## Paso 4: Indexing y Slicing

Similar a NumPy:

```python
t = torch.arange(12).reshape(3, 4)

print(t[0])         # Primera fila
print(t[:, 1])      # Segunda columna
print(t[1:, 2:])    # Submatriz

# Boolean indexing
mask = t > 5
print(t[mask])      # Elementos > 5
```

**Descomenta** la sección del Paso 4.

---

## Paso 5: Reshape y Manipulación

Cambiar la forma de tensores:

```python
t = torch.arange(12)

# Reshape
reshaped = t.reshape(3, 4)
viewed = t.view(3, 4)

# Flatten
flat = reshaped.flatten()

# Squeeze/Unsqueeze
t = torch.rand(1, 3, 1)
squeezed = t.squeeze()      # [3]
unsqueezed = squeezed.unsqueeze(0)  # [1, 3]
```

**Descomenta** la sección del Paso 5.

---

## Paso 6: NumPy ↔ PyTorch

Conversión entre frameworks:

```python
import numpy as np

# NumPy → PyTorch
np_arr = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(np_arr)  # Comparte memoria!

# PyTorch → NumPy
tensor = torch.tensor([1.0, 2.0, 3.0])
np_arr = tensor.numpy()
```

**Descomenta** la sección del Paso 6.

---

## Paso 7: Dispositivos (CPU/GPU)

Mover tensores entre dispositivos:

```python
# Detectar GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Crear en dispositivo específico
t = torch.rand(3, 3, device=device)

# Mover entre dispositivos
t_cpu = t.cpu()
t_gpu = t.to('cuda')  # Si hay GPU
```

**Descomenta** la sección del Paso 7.

---

## ✅ Verificación

Al completar todos los pasos, deberías ver:
- Tensores creados con diferentes métodos
- Operaciones matemáticas correctas
- Manipulación de shapes funcionando
- Conversión NumPy ↔ PyTorch exitosa

---

## 📚 Recursos

- [PyTorch Tensor Tutorial](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)
- [Tensor Operations](https://pytorch.org/docs/stable/tensors.html)
