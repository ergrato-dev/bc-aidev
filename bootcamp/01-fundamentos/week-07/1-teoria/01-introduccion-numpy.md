# 📘 Introducción a NumPy

## 🎯 Objetivos

- Comprender qué es NumPy y su importancia en Data Science
- Entender la diferencia entre listas Python y ndarray
- Conocer la estructura interna del ndarray
- Instalar y configurar NumPy

---

## 📋 Contenido

1. [¿Qué es NumPy?](#1-qué-es-numpy)
2. [ndarray vs Listas Python](#2-ndarray-vs-listas-python)
3. [Anatomía del ndarray](#3-anatomía-del-ndarray)
4. [Tipos de Datos (dtypes)](#4-tipos-de-datos-dtypes)

---

## 1. ¿Qué es NumPy?

**NumPy** (Numerical Python) es la biblioteca fundamental para computación científica en Python. Proporciona:

- Arrays N-dimensionales eficientes (`ndarray`)
- Operaciones matemáticas vectorizadas
- Herramientas de álgebra lineal
- Generación de números aleatorios
- Integración con C/C++ y Fortran

### ¿Por qué NumPy es Fundamental?

![Stack de Data Science](../0-assets/01-numpy-ecosystem.svg)

```python
# NumPy es la base de casi todo en Data Science
import numpy as np      # Base
import pandas as pd     # Construido sobre NumPy
import matplotlib.pyplot as plt  # Usa arrays NumPy
import sklearn          # Espera arrays NumPy
import tensorflow as tf # Compatible con NumPy
```

### Instalación

```bash
# Con pip
pip install numpy

# Con conda
conda install numpy

# Verificar instalación
python -c "import numpy as np; print(np.__version__)"
```

---

## 2. ndarray vs Listas Python

### El Problema con las Listas

```python
# Lista Python - heterogénea y flexible
python_list = [1, 2.5, "texto", True, [1, 2]]

# Operación elemento a elemento requiere loop
numbers = [1, 2, 3, 4, 5]
doubled = []
for n in numbers:
    doubled.append(n * 2)
# O con list comprehension
doubled = [n * 2 for n in numbers]
```

### La Solución: ndarray

```python
import numpy as np

# ndarray - homogéneo y eficiente
numpy_array = np.array([1, 2, 3, 4, 5])

# Operación vectorizada - sin loop
doubled = numpy_array * 2  # array([2, 4, 6, 8, 10])
```

### Comparación de Rendimiento

```python
import numpy as np
import time

size = 1_000_000

# Lista Python
python_list = list(range(size))
start = time.time()
result = [x * 2 for x in python_list]
print(f"Lista Python: {time.time() - start:.4f}s")

# NumPy array
numpy_array = np.arange(size)
start = time.time()
result = numpy_array * 2
print(f"NumPy array: {time.time() - start:.4f}s")

# Resultado típico:
# Lista Python: 0.0800s
# NumPy array: 0.0010s  <- ¡80x más rápido!
```

### ¿Por qué NumPy es más Rápido?

| Aspecto           | Lista Python                 | ndarray NumPy            |
| ----------------- | ---------------------------- | ------------------------ |
| **Tipo de datos** | Heterogéneo (cualquier tipo) | Homogéneo (un solo tipo) |
| **Memoria**       | Punteros a objetos dispersos | Bloque contiguo          |
| **Operaciones**   | Loop en Python (lento)       | Código C optimizado      |
| **Vectorización** | No nativa                    | Nativa (SIMD)            |

---

## 3. Anatomía del ndarray

### Atributos Principales

```python
import numpy as np

# Crear un array 2D
matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# Atributos fundamentales
print(matrix.ndim)    # 2 - número de dimensiones
print(matrix.shape)   # (3, 4) - filas x columnas
print(matrix.size)    # 12 - total de elementos
print(matrix.dtype)   # int64 - tipo de datos
print(matrix.itemsize)  # 8 - bytes por elemento
print(matrix.nbytes)  # 96 - bytes totales (12 * 8)
```

### Visualización de Dimensiones

```python
# 0D - Escalar
scalar = np.array(42)
print(scalar.shape)  # ()

# 1D - Vector
vector = np.array([1, 2, 3, 4, 5])
print(vector.shape)  # (5,)

# 2D - Matriz
matrix = np.array([[1, 2], [3, 4], [5, 6]])
print(matrix.shape)  # (3, 2)

# 3D - Tensor
tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(tensor.shape)  # (2, 2, 2)
```

### Memoria Contigua

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

# En memoria: [1, 2, 3, 4, 5, 6] (C-order, row-major)
print(arr.flags)
# C_CONTIGUOUS : True   <- Filas contiguas
# F_CONTIGUOUS : False  <- Columnas no contiguas
```

---

## 4. Tipos de Datos (dtypes)

### dtypes Comunes

| dtype       | Descripción     | Rango/Precisión       |
| ----------- | --------------- | --------------------- |
| `int8`      | Entero 8-bit    | -128 a 127            |
| `int32`     | Entero 32-bit   | -2³¹ a 2³¹-1          |
| `int64`     | Entero 64-bit   | -2⁶³ a 2⁶³-1          |
| `float32`   | Flotante 32-bit | ~7 decimales          |
| `float64`   | Flotante 64-bit | ~15 decimales         |
| `bool`      | Booleano        | True/False            |
| `complex64` | Complejo 64-bit | Parte real+imaginaria |
| `str_`      | String Unicode  | Variable              |

### Especificar dtype

```python
import numpy as np

# dtype por defecto (int64 o float64)
arr1 = np.array([1, 2, 3])
print(arr1.dtype)  # int64

arr2 = np.array([1.0, 2.0, 3.0])
print(arr2.dtype)  # float64

# Especificar dtype explícitamente
arr3 = np.array([1, 2, 3], dtype=np.float32)
print(arr3.dtype)  # float32

arr4 = np.array([1, 2, 3], dtype='int8')
print(arr4.dtype)  # int8
```

### Conversión de dtype (astype)

```python
import numpy as np

# Array de enteros
integers = np.array([1, 2, 3, 4, 5])
print(integers.dtype)  # int64

# Convertir a float
floats = integers.astype(np.float32)
print(floats.dtype)  # float32
print(floats)  # [1. 2. 3. 4. 5.]

# Convertir a bool
bools = integers.astype(bool)
print(bools)  # [ True  True  True  True  True]

# ⚠️ Cuidado con pérdida de precisión
big_floats = np.array([1.7, 2.3, 3.9])
truncated = big_floats.astype(int)
print(truncated)  # [1 2 3] - ¡Se trunca, no redondea!
```

### Elegir el dtype Correcto

```python
import numpy as np

# Para imágenes (0-255)
image = np.zeros((100, 100, 3), dtype=np.uint8)
print(f"Memoria: {image.nbytes / 1024:.1f} KB")  # 29.3 KB

# Si usáramos float64
image_float = np.zeros((100, 100, 3), dtype=np.float64)
print(f"Memoria: {image_float.nbytes / 1024:.1f} KB")  # 234.4 KB
# ¡8x más memoria!
```

---

## ✅ Checklist de Verificación

- [ ] Entiendo por qué NumPy es más rápido que listas Python
- [ ] Conozco los atributos principales del ndarray (shape, dtype, ndim)
- [ ] Sé elegir el dtype apropiado para mi caso de uso
- [ ] Puedo convertir entre dtypes con astype()
- [ ] Entiendo el concepto de memoria contigua

---

## 📚 Recursos Adicionales

- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy for MATLAB Users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)
- [Array Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
