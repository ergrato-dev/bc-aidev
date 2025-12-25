# 📖 Glosario - Semana 07: NumPy

Términos técnicos clave de esta semana, ordenados alfabéticamente.

---

## A

### Array (ndarray)

Estructura de datos fundamental de NumPy. Colección homogénea de elementos del mismo tipo organizados en una cuadrícula n-dimensional.

```python
import numpy as np
arr = np.array([1, 2, 3, 4])  # Array 1D
matrix = np.array([[1, 2], [3, 4]])  # Array 2D
```

### Axis (Eje)

Dimensión de un array a lo largo de la cual se realizan operaciones. En un array 2D: `axis=0` son filas, `axis=1` son columnas.

```python
matrix = np.array([[1, 2], [3, 4]])
np.sum(matrix, axis=0)  # [4, 6] - suma columnas
np.sum(matrix, axis=1)  # [3, 7] - suma filas
```

---

## B

### Boolean Indexing

Técnica para seleccionar elementos usando una máscara de valores booleanos.

```python
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 3
arr[mask]  # array([4, 5])
```

### Broadcasting

Mecanismo que permite a NumPy realizar operaciones entre arrays de diferentes shapes, expandiendo automáticamente el array más pequeño.

```python
matrix = np.ones((3, 4))
row = np.array([1, 2, 3, 4])
matrix + row  # row se "expande" a cada fila
```

---

## C

### Contiguous (Memoria Contigua)

Array cuyos elementos están almacenados en posiciones de memoria consecutivas. C-contiguous (por filas) vs Fortran-contiguous (por columnas).

### Copy (Copia)

Nuevo array con sus propios datos, independiente del original.

```python
arr = np.array([1, 2, 3])
copy = arr.copy()  # Nueva memoria
copy[0] = 99  # No afecta a arr
```

---

## D

### Dtype (Data Type)

Tipo de datos de los elementos del array. Determina tamaño en memoria y operaciones permitidas.

```python
np.array([1, 2, 3], dtype=np.float32)  # 32-bit floats
np.array([1, 2, 3], dtype=np.int8)     # 8-bit integers
```

Tipos comunes: `int32`, `int64`, `float32`, `float64`, `bool`, `complex64`.

---

## E

### Element-wise (Elemento a Elemento)

Operaciones que se aplican independientemente a cada elemento del array.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a * b  # [4, 10, 18] - multiplicación elemento a elemento
```

---

## F

### Fancy Indexing

Indexación usando arrays de índices para seleccionar múltiples elementos.

```python
arr = np.array([10, 20, 30, 40, 50])
indices = [0, 2, 4]
arr[indices]  # array([10, 30, 50])
```

### Flatten

Convertir un array multidimensional en 1D, creando una copia.

```python
matrix = np.array([[1, 2], [3, 4]])
matrix.flatten()  # array([1, 2, 3, 4])
```

---

## I

### Indexing (Indexación)

Acceso a elementos individuales usando índices (base 0).

```python
arr = np.array([10, 20, 30])
arr[0]   # 10 (primer elemento)
arr[-1]  # 30 (último elemento)
```

---

## L

### Linspace

Función que genera array con números equiespaciados en un intervalo.

```python
np.linspace(0, 1, 5)  # [0., 0.25, 0.5, 0.75, 1.]
```

---

## M

### Matrix Multiplication (Producto Matricial)

Operación de álgebra lineal entre matrices usando `@` o `np.dot()`.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
A @ B  # Producto matricial (diferente de A * B)
```

---

## N

### Ndim

Número de dimensiones (ejes) de un array.

```python
arr_1d = np.array([1, 2, 3])
arr_2d = np.array([[1, 2], [3, 4]])
arr_1d.ndim  # 1
arr_2d.ndim  # 2
```

---

## R

### Reshape

Cambiar la forma de un array sin modificar sus datos.

```python
arr = np.arange(12)
arr.reshape((3, 4))  # 3 filas, 4 columnas
arr.reshape((3, -1)) # -1 calcula automáticamente
```

### Ravel

Similar a flatten pero retorna una vista cuando es posible.

```python
matrix = np.array([[1, 2], [3, 4]])
matrix.ravel()  # Vista 1D de los datos
```

---

## S

### Shape

Tupla que indica el tamaño de cada dimensión del array.

```python
matrix = np.zeros((3, 4, 5))
matrix.shape  # (3, 4, 5)
```

### Slicing

Extraer secciones de un array con sintaxis `start:stop:step`.

```python
arr = np.arange(10)
arr[2:7]    # [2, 3, 4, 5, 6]
arr[::2]    # [0, 2, 4, 6, 8]
arr[::-1]   # Array invertido
```

### Stride

Número de bytes a saltar en memoria para moverse al siguiente elemento en cada dimensión.

---

## U

### Ufunc (Universal Function)

Funciones NumPy optimizadas que operan elemento a elemento sobre arrays.

```python
np.sqrt(arr)   # Raíz cuadrada
np.exp(arr)    # Exponencial
np.sin(arr)    # Seno
```

---

## V

### Vectorization (Vectorización)

Técnica de reemplazar bucles explícitos con operaciones de array para mejor rendimiento.

```python
# ❌ Lento con bucle
result = []
for x in data:
    result.append(x * 2)

# ✅ Vectorizado (rápido)
result = data * 2
```

### View (Vista)

Array que comparte datos con otro array. Modificar uno afecta al otro.

```python
arr = np.array([1, 2, 3, 4])
view = arr[1:3]  # Vista, no copia
view[0] = 99  # Modifica arr también!
```

---

## Símbolos y Operadores

### @ (Operador de producto matricial)

```python
A @ B  # Equivale a np.matmul(A, B)
```

### : (Slicing)

```python
arr[start:stop:step]
arr[:]    # Todos los elementos
arr[::2]  # Cada 2 elementos
```

### ... (Ellipsis)

```python
arr_4d[..., 0]  # Equivale a arr_4d[:, :, :, 0]
```

---

## 📊 Resumen Visual

```
Array NumPy
├── Atributos
│   ├── shape    → Dimensiones
│   ├── dtype    → Tipo de datos
│   ├── ndim     → Número de ejes
│   └── size     → Total elementos
│
├── Creación
│   ├── array()   → Desde lista
│   ├── zeros()   → Array de ceros
│   ├── ones()    → Array de unos
│   ├── arange()  → Rango con paso
│   └── linspace()→ Puntos equiespaciados
│
├── Acceso
│   ├── Indexing  → arr[i, j]
│   ├── Slicing   → arr[start:stop]
│   ├── Fancy     → arr[[0, 2, 4]]
│   └── Boolean   → arr[arr > 0]
│
└── Operaciones
    ├── Element-wise → +, -, *, /
    ├── Broadcasting → Auto-expansión
    ├── Ufuncs       → sqrt, sin, exp
    └── Agregaciones → sum, mean, max
```
