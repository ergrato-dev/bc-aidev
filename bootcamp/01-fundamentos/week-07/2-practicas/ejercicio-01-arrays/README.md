# 🏋️ Ejercicio 01: Creación de Arrays

## 🎯 Objetivo

Aprender a crear arrays NumPy usando diferentes métodos y entender sus atributos.

---

## 📋 Pasos

### Paso 1: Importar NumPy y crear arrays básicos

Importamos NumPy con el alias estándar `np` y creamos arrays desde listas Python.

```python
import numpy as np

# Desde lista
arr = np.array([1, 2, 3, 4, 5])
print(arr)  # [1 2 3 4 5]
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Explorar atributos del array

Los atributos más importantes son `shape`, `dtype`, `ndim` y `size`.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2, 3) - 2 filas, 3 columnas
print(arr.dtype)  # int64
print(arr.ndim)   # 2 dimensiones
print(arr.size)   # 6 elementos
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Funciones de inicialización

NumPy ofrece funciones para crear arrays con valores predefinidos.

```python
zeros = np.zeros((3, 4))       # Matriz 3x4 de ceros
ones = np.ones((2, 3))         # Matriz 2x3 de unos
full = np.full((2, 2), 7)      # Matriz 2x2 llena de 7
empty = np.empty((2, 3))       # Sin inicializar (valores basura)
eye = np.eye(4)                # Matriz identidad 4x4
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Rangos numéricos

`arange` y `linspace` generan secuencias de números.

```python
# arange: start, stop, step
arr = np.arange(0, 10, 2)  # [0 2 4 6 8]

# linspace: start, stop, num_points
arr = np.linspace(0, 1, 5)  # [0.   0.25 0.5  0.75 1.  ]
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Reshape y manipulación de forma

Cambiar la forma de un array sin modificar sus datos.

```python
arr = np.arange(12)
matrix = arr.reshape((3, 4))  # 3 filas, 4 columnas
auto = arr.reshape((3, -1))   # -1 calcula automáticamente
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Especificar dtype

Controlar el tipo de datos del array.

```python
arr_float = np.array([1, 2, 3], dtype=np.float32)
arr_int = np.zeros((2, 2), dtype=np.int8)
converted = arr_float.astype(int)  # Convertir tipo
```

**Descomenta** la sección del Paso 6.

---

## ✅ Verificación

Al completar deberías poder:

- [ ] Crear arrays desde listas
- [ ] Inspeccionar shape, dtype, ndim, size
- [ ] Usar zeros, ones, full, eye
- [ ] Generar rangos con arange y linspace
- [ ] Cambiar forma con reshape
- [ ] Especificar y convertir dtypes
