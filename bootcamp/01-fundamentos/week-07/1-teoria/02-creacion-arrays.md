# 📘 Creación de Arrays en NumPy

## 🎯 Objetivos

- Crear arrays desde secuencias Python
- Usar funciones de creación: zeros, ones, empty, full
- Generar rangos con arange y linspace
- Crear arrays con valores aleatorios
- Cambiar la forma de arrays con reshape

---

## 📋 Contenido

1. [Desde Secuencias Python](#1-desde-secuencias-python)
2. [Funciones de Inicialización](#2-funciones-de-inicialización)
3. [Rangos Numéricos](#3-rangos-numéricos)
4. [Arrays Aleatorios](#4-arrays-aleatorios)
5. [Reshape y Manipulación de Forma](#5-reshape-y-manipulación-de-forma)

---

## 1. Desde Secuencias Python

### np.array()

```python
import numpy as np

# Desde lista
list_1d = [1, 2, 3, 4, 5]
arr_1d = np.array(list_1d)
print(arr_1d)  # [1 2 3 4 5]

# Desde lista de listas (2D)
list_2d = [[1, 2, 3], [4, 5, 6]]
arr_2d = np.array(list_2d)
print(arr_2d)
# [[1 2 3]
#  [4 5 6]]

# Desde tupla
tuple_data = (1.5, 2.5, 3.5)
arr_tuple = np.array(tuple_data)
print(arr_tuple)  # [1.5 2.5 3.5]

# Especificando dtype
arr_float = np.array([1, 2, 3], dtype=float)
print(arr_float)  # [1. 2. 3.]
```

### np.asarray() vs np.array()

```python
import numpy as np

original = np.array([1, 2, 3])

# np.array() SIEMPRE crea una copia
copy = np.array(original)
copy[0] = 99
print(original)  # [1 2 3] - no cambió

# np.asarray() NO copia si ya es ndarray
view = np.asarray(original)
view[0] = 99
print(original)  # [99 2 3] - ¡cambió!

# Útil para aceptar listas o arrays como input
def process_data(data):
    arr = np.asarray(data)  # Sin copia innecesaria si ya es array
    return arr.mean()
```

---

## 2. Funciones de Inicialización

### np.zeros() - Array de ceros

```python
import numpy as np

# 1D
zeros_1d = np.zeros(5)
print(zeros_1d)  # [0. 0. 0. 0. 0.]

# 2D - shape como tupla
zeros_2d = np.zeros((3, 4))
print(zeros_2d)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# Con dtype específico
zeros_int = np.zeros((2, 2), dtype=int)
print(zeros_int)
# [[0 0]
#  [0 0]]
```

### np.ones() - Array de unos

```python
import numpy as np

ones_2d = np.ones((2, 3))
print(ones_2d)
# [[1. 1. 1.]
#  [1. 1. 1.]]

# Útil para máscaras o inicialización
weights = np.ones(100) * 0.5  # Array de 0.5
```

### np.full() - Array con valor específico

```python
import numpy as np

# Llenar con un valor
filled = np.full((3, 3), 7)
print(filled)
# [[7 7 7]
#  [7 7 7]
#  [7 7 7]]

# Con float
filled_pi = np.full((2, 4), 3.14159)
print(filled_pi)
```

### np.empty() - Array sin inicializar

```python
import numpy as np

# ⚠️ Contenido aleatorio (basura de memoria)
empty = np.empty((2, 3))
print(empty)  # Valores impredecibles

# Más rápido que zeros si vas a llenar todo después
# Útil cuando sabes que sobrescribirás todos los valores
```

### np.eye() e np.identity() - Matrices identidad

```python
import numpy as np

# Matriz identidad
identity = np.eye(4)
print(identity)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]

# Con offset (k)
eye_k1 = np.eye(4, k=1)  # Diagonal superior
print(eye_k1)
# [[0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]
#  [0. 0. 0. 0.]]
```

### np.diag() - Crear diagonal o extraer diagonal

```python
import numpy as np

# Crear matriz diagonal
diag_matrix = np.diag([1, 2, 3, 4])
print(diag_matrix)
# [[1 0 0 0]
#  [0 2 0 0]
#  [0 0 3 0]
#  [0 0 0 4]]

# Extraer diagonal de matriz existente
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
diagonal = np.diag(matrix)
print(diagonal)  # [1 5 9]
```

---

## 3. Rangos Numéricos

### np.arange() - Rango con paso

```python
import numpy as np

# Similar a range() pero devuelve array
arr = np.arange(10)
print(arr)  # [0 1 2 3 4 5 6 7 8 9]

# Con start y stop
arr = np.arange(5, 15)
print(arr)  # [ 5  6  7  8  9 10 11 12 13 14]

# Con step
arr = np.arange(0, 20, 2)
print(arr)  # [ 0  2  4  6  8 10 12 14 16 18]

# Con flotantes
arr = np.arange(0, 1, 0.1)
print(arr)  # [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]

# ⚠️ Con flotantes, usar linspace es más preciso
```

### np.linspace() - Puntos equiespaciados

```python
import numpy as np

# 5 puntos entre 0 y 10 (inclusive)
arr = np.linspace(0, 10, 5)
print(arr)  # [ 0.   2.5  5.   7.5 10. ]

# 100 puntos para gráficas suaves
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Sin incluir el endpoint
arr = np.linspace(0, 10, 5, endpoint=False)
print(arr)  # [0. 2. 4. 6. 8.]

# Retornar el step
arr, step = np.linspace(0, 10, 5, retstep=True)
print(f"Step: {step}")  # Step: 2.5
```

### np.logspace() - Escala logarítmica

```python
import numpy as np

# 5 puntos entre 10^0 y 10^4
arr = np.logspace(0, 4, 5)
print(arr)  # [1.e+00 1.e+01 1.e+02 1.e+03 1.e+04]
# Es decir: [1, 10, 100, 1000, 10000]

# Útil para parámetros de ML
learning_rates = np.logspace(-4, -1, 4)
print(learning_rates)  # [0.0001 0.001  0.01   0.1]
```

---

## 4. Arrays Aleatorios

### Generador moderno (recomendado)

```python
import numpy as np

# Crear generador con semilla para reproducibilidad
rng = np.random.default_rng(seed=42)

# Enteros aleatorios
random_int = rng.integers(0, 10, size=5)
print(random_int)  # [0 7 6 4 4]

# Flotantes uniformes [0, 1)
random_float = rng.random(size=(2, 3))
print(random_float)

# Distribución normal (media=0, std=1)
normal = rng.standard_normal(size=1000)
print(f"Media: {normal.mean():.3f}, Std: {normal.std():.3f}")

# Normal con parámetros específicos
normal_custom = rng.normal(loc=100, scale=15, size=1000)  # IQ distribution
```

### Funciones legacy (aún comunes)

```python
import numpy as np

# Fijar semilla para reproducibilidad
np.random.seed(42)

# Enteros aleatorios
arr = np.random.randint(0, 100, size=(3, 3))

# Flotantes uniformes [0, 1)
arr = np.random.random((2, 4))

# Distribución normal
arr = np.random.randn(100)  # Media=0, Std=1

# Elección aleatoria
choices = np.random.choice(['a', 'b', 'c'], size=10)

# Shuffle in-place
arr = np.arange(10)
np.random.shuffle(arr)
```

---

## 5. Reshape y Manipulación de Forma

### reshape() - Cambiar forma

```python
import numpy as np

arr = np.arange(12)
print(arr)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# Reshape a 2D
matrix = arr.reshape((3, 4))
print(matrix)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Reshape a 3D
tensor = arr.reshape((2, 2, 3))
print(tensor.shape)  # (2, 2, 3)

# Usar -1 para inferir dimensión
auto = arr.reshape((3, -1))  # NumPy calcula: 12/3 = 4
print(auto.shape)  # (3, 4)

auto = arr.reshape((-1, 6))  # 12/6 = 2
print(auto.shape)  # (2, 6)
```

### flatten() vs ravel()

```python
import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6]])

# flatten() - siempre crea copia
flat_copy = matrix.flatten()
flat_copy[0] = 99
print(matrix[0, 0])  # 1 - original no cambió

# ravel() - vista si es posible (más eficiente)
flat_view = matrix.ravel()
flat_view[0] = 99
print(matrix[0, 0])  # 99 - ¡original cambió!
```

### Transponer

```python
import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)  # (2, 3)

# Transponer
transposed = matrix.T
print(transposed.shape)  # (3, 2)
print(transposed)
# [[1 4]
#  [2 5]
#  [3 6]]

# Para arrays 1D, .T no hace nada
vector = np.array([1, 2, 3])
print(vector.T.shape)  # (3,) - sigue igual
```

### Expandir y reducir dimensiones

```python
import numpy as np

arr = np.array([1, 2, 3])
print(arr.shape)  # (3,)

# Añadir dimensión
row = arr[np.newaxis, :]
print(row.shape)  # (1, 3)

col = arr[:, np.newaxis]
print(col.shape)  # (3, 1)

# Equivalente con expand_dims
row = np.expand_dims(arr, axis=0)
col = np.expand_dims(arr, axis=1)

# Eliminar dimensiones de tamaño 1
squeezed = np.squeeze(row)
print(squeezed.shape)  # (3,)
```

---

## ✅ Checklist de Verificación

- [ ] Puedo crear arrays desde listas y tuplas
- [ ] Conozco la diferencia entre zeros, ones, empty y full
- [ ] Sé cuándo usar arange vs linspace
- [ ] Puedo generar arrays aleatorios reproducibles
- [ ] Entiendo reshape y el uso de -1
- [ ] Sé la diferencia entre flatten y ravel

---

## 📚 Recursos Adicionales

- [Array creation routines](https://numpy.org/doc/stable/reference/routines.array-creation.html)
- [Random sampling](https://numpy.org/doc/stable/reference/random/index.html)
