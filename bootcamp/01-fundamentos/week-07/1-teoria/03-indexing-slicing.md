# 📘 Indexing y Slicing en NumPy

## 🎯 Objetivos

- Dominar indexing básico en arrays 1D, 2D y 3D
- Aplicar slicing con start:stop:step
- Usar fancy indexing con arrays de índices
- Filtrar con boolean indexing
- Entender vistas vs copias

---

## 📋 Contenido

1. [Indexing Básico](#1-indexing-básico)
2. [Slicing](#2-slicing)
3. [Fancy Indexing](#3-fancy-indexing)
4. [Boolean Indexing](#4-boolean-indexing)
5. [Vistas vs Copias](#5-vistas-vs-copias)

---

## 1. Indexing Básico

### Arrays 1D

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# Índices positivos (desde el inicio)
print(arr[0])   # 10 (primer elemento)
print(arr[2])   # 30 (tercer elemento)
print(arr[-1])  # 50 (último elemento)
print(arr[-2])  # 40 (penúltimo)
```

### Arrays 2D (Matrices)

```python
import numpy as np

matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# Sintaxis: matrix[fila, columna]
print(matrix[0, 0])   # 1 (primera fila, primera columna)
print(matrix[1, 2])   # 7 (segunda fila, tercera columna)
print(matrix[-1, -1]) # 12 (última fila, última columna)

# Acceder a fila completa
print(matrix[1])      # [5 6 7 8]

# Acceder a columna completa
print(matrix[:, 2])   # [ 3  7 11]
```

### Arrays 3D

```python
import numpy as np

# Shape: (2, 3, 4) - 2 "capas", 3 filas, 4 columnas
tensor = np.arange(24).reshape((2, 3, 4))
print(tensor)
# [[[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]
#
#  [[12 13 14 15]
#   [16 17 18 19]
#   [20 21 22 23]]]

# tensor[capa, fila, columna]
print(tensor[0, 1, 2])  # 6
print(tensor[1, 2, 3])  # 23

# Primera capa completa
print(tensor[0])
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
```

---

## 2. Slicing

### Sintaxis: start:stop:step

```python
import numpy as np

arr = np.arange(10)  # [0 1 2 3 4 5 6 7 8 9]

# Básico
print(arr[2:7])     # [2 3 4 5 6] (índice 2 a 6)
print(arr[:5])      # [0 1 2 3 4] (inicio a 4)
print(arr[5:])      # [5 6 7 8 9] (5 al final)
print(arr[::2])     # [0 2 4 6 8] (cada 2)
print(arr[1::2])    # [1 3 5 7 9] (impares)

# Step negativo (reverso)
print(arr[::-1])    # [9 8 7 6 5 4 3 2 1 0]
print(arr[7:2:-1])  # [7 6 5 4 3]
```

### Slicing 2D

```python
import numpy as np

matrix = np.arange(20).reshape((4, 5))
print(matrix)
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]
#  [15 16 17 18 19]]

# Submatriz
print(matrix[1:3, 2:4])
# [[ 7  8]
#  [12 13]]

# Primeras 2 filas, todas las columnas
print(matrix[:2, :])
# [[0 1 2 3 4]
#  [5 6 7 8 9]]

# Todas las filas, últimas 2 columnas
print(matrix[:, -2:])
# [[ 3  4]
#  [ 8  9]
#  [13 14]
#  [18 19]]

# Filas pares
print(matrix[::2, :])
# [[ 0  1  2  3  4]
#  [10 11 12 13 14]]
```

### Slicing 3D

```python
import numpy as np

tensor = np.arange(24).reshape((2, 3, 4))

# Segunda capa, primeras 2 filas, columnas 1-2
print(tensor[1, :2, 1:3])
# [[13 14]
#  [17 18]]

# Todas las capas, primera fila, última columna
print(tensor[:, 0, -1])  # [ 3 15]
```

---

## 3. Fancy Indexing

Usar arrays de índices para acceder a múltiples elementos.

### Con array de índices

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60, 70])

# Seleccionar índices específicos
indices = [0, 2, 5]
print(arr[indices])  # [10 30 60]

# Con array NumPy
indices = np.array([1, 3, 4])
print(arr[indices])  # [20 40 50]

# Orden arbitrario y repeticiones
indices = [4, 2, 2, 0]
print(arr[indices])  # [50 30 30 10]
```

### Fancy indexing 2D

```python
import numpy as np

matrix = np.arange(12).reshape((3, 4))
print(matrix)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Seleccionar elementos específicos
rows = [0, 1, 2]
cols = [1, 2, 0]
print(matrix[rows, cols])  # [1 6 8]
# Equivale a: matrix[0,1], matrix[1,2], matrix[2,0]

# Seleccionar filas completas
print(matrix[[0, 2]])
# [[ 0  1  2  3]
#  [ 8  9 10 11]]

# Seleccionar columnas completas
print(matrix[:, [0, 3]])
# [[ 0  3]
#  [ 4  7]
#  [ 8 11]]
```

### np.ix\_() para productos cartesianos

```python
import numpy as np

matrix = np.arange(12).reshape((3, 4))

# Submatriz con filas [0,2] y columnas [1,3]
rows = [0, 2]
cols = [1, 3]

# Sin ix_ - obtiene elementos individuales
print(matrix[rows, cols])  # [1 11]

# Con ix_ - obtiene submatriz completa
print(matrix[np.ix_(rows, cols)])
# [[ 1  3]
#  [ 9 11]]
```

---

## 4. Boolean Indexing

Filtrar elementos usando condiciones.

### Filtrado básico

```python
import numpy as np

arr = np.array([1, 5, 3, 8, 2, 9, 4, 7])

# Crear máscara booleana
mask = arr > 5
print(mask)  # [False False False  True False  True False  True]

# Aplicar máscara
print(arr[mask])  # [8 9 7]

# Directamente
print(arr[arr > 5])  # [8 9 7]

# Condiciones múltiples
print(arr[(arr > 3) & (arr < 8)])  # [5 4 7]
print(arr[(arr < 3) | (arr > 7)])  # [1 2 8 9]

# Negación
print(arr[~(arr > 5)])  # [1 5 3 2 4]
```

### Con matrices

```python
import numpy as np

matrix = np.arange(12).reshape((3, 4))
print(matrix)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Filtrar elementos > 5
print(matrix[matrix > 5])  # [ 6  7  8  9 10 11]

# Modificar elementos que cumplen condición
matrix[matrix > 8] = 0
print(matrix)
# [[0 1 2 3]
#  [4 5 6 7]
#  [8 0 0 0]]
```

### np.where() - Condicional vectorizado

```python
import numpy as np

arr = np.array([1, 5, 3, 8, 2, 9])

# np.where(condición, valor_si_true, valor_si_false)
result = np.where(arr > 5, 1, 0)
print(result)  # [0 0 0 1 0 1]

# Obtener índices donde se cumple condición
indices = np.where(arr > 5)
print(indices)  # (array([3, 5]),)
print(arr[indices])  # [8 9]

# Reemplazar valores
cleaned = np.where(arr > 5, arr, -1)
print(cleaned)  # [-1 -1 -1  8 -1  9]
```

---

## 5. Vistas vs Copias

### Slicing crea VISTAS

```python
import numpy as np

original = np.arange(10)
print(original)  # [0 1 2 3 4 5 6 7 8 9]

# El slice es una VISTA del original
slice_view = original[2:6]
print(slice_view)  # [2 3 4 5]

# Modificar la vista AFECTA al original
slice_view[0] = 99
print(original)  # [ 0  1 99  3  4  5  6  7  8  9]
```

### Fancy indexing crea COPIAS

```python
import numpy as np

original = np.arange(10)

# Fancy indexing crea COPIA
fancy_copy = original[[2, 3, 4, 5]]
print(fancy_copy)  # [2 3 4 5]

# Modificar la copia NO afecta al original
fancy_copy[0] = 99
print(original)  # [0 1 2 3 4 5 6 7 8 9] - sin cambios
```

### Forzar copia explícita

```python
import numpy as np

original = np.arange(10)

# Copia explícita
explicit_copy = original[2:6].copy()
explicit_copy[0] = 99
print(original)  # [0 1 2 3 4 5 6 7 8 9] - sin cambios

# Verificar si es vista
print(np.shares_memory(original, original[2:6]))  # True (vista)
print(np.shares_memory(original, original[[2,3,4,5]]))  # False (copia)
```

### Resumen: Vistas vs Copias

| Operación                | Resultado | Comparte memoria |
| ------------------------ | --------- | ---------------- |
| `arr[2:6]` (slicing)     | Vista     | Sí               |
| `arr[[1,2,3]]` (fancy)   | Copia     | No               |
| `arr[arr > 5]` (boolean) | Copia     | No               |
| `arr.copy()`             | Copia     | No               |
| `arr.view()`             | Vista     | Sí               |
| `arr.reshape()`          | Vista\*   | Generalmente sí  |

---

## ✅ Checklist de Verificación

- [ ] Puedo indexar arrays 1D, 2D y 3D
- [ ] Domino la sintaxis start:stop:step
- [ ] Sé usar fancy indexing con arrays de índices
- [ ] Puedo filtrar con boolean indexing
- [ ] Entiendo cuándo se crea vista vs copia

---

## 📚 Recursos Adicionales

- [Indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html)
- [Boolean array indexing](https://numpy.org/doc/stable/reference/arrays.indexing.html#boolean-array-indexing)
