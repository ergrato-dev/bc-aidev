"""
Ejercicio 01: Creación de Arrays
================================
Aprende a crear arrays NumPy de diferentes formas.
"""

# ============================================
# PASO 1: Importar NumPy y crear arrays básicos
# ============================================
print('--- Paso 1: Arrays básicos ---')

# NumPy se importa con el alias estándar 'np'
# Descomenta las siguientes líneas:

# import numpy as np

# # Crear array desde lista Python
# arr_1d = np.array([1, 2, 3, 4, 5])
# print('Array 1D:', arr_1d)

# # Crear array 2D desde lista de listas
# arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
# print('Array 2D:')
# print(arr_2d)

# # Crear array 3D
# arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# print('Array 3D shape:', arr_3d.shape)

print()

# ============================================
# PASO 2: Explorar atributos del array
# ============================================
print('--- Paso 2: Atributos del array ---')

# Los atributos nos dan información sobre el array
# Descomenta las siguientes líneas:

# import numpy as np

# matrix = np.array([
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12]
# ])

# print('Array:')
# print(matrix)
# print()
# print(f'shape: {matrix.shape}')      # Dimensiones (filas, columnas)
# print(f'dtype: {matrix.dtype}')      # Tipo de datos
# print(f'ndim: {matrix.ndim}')        # Número de dimensiones
# print(f'size: {matrix.size}')        # Total de elementos
# print(f'itemsize: {matrix.itemsize} bytes')  # Bytes por elemento
# print(f'nbytes: {matrix.nbytes} bytes')      # Bytes totales

print()

# ============================================
# PASO 3: Funciones de inicialización
# ============================================
print('--- Paso 3: Funciones de inicialización ---')

# NumPy tiene funciones para crear arrays con valores específicos
# Descomenta las siguientes líneas:

# import numpy as np

# # Array de ceros
# zeros = np.zeros((3, 4))
# print('zeros(3, 4):')
# print(zeros)
# print()

# # Array de unos
# ones = np.ones((2, 3))
# print('ones(2, 3):')
# print(ones)
# print()

# # Array con valor específico
# full = np.full((2, 2), 7)
# print('full(2, 2) con valor 7:')
# print(full)
# print()

# # Matriz identidad
# eye = np.eye(4)
# print('eye(4) - Matriz identidad:')
# print(eye)
# print()

# # Matriz diagonal
# diag = np.diag([1, 2, 3, 4])
# print('diag([1, 2, 3, 4]):')
# print(diag)

print()

# ============================================
# PASO 4: Rangos numéricos
# ============================================
print('--- Paso 4: Rangos numéricos ---')

# arange y linspace generan secuencias de números
# Descomenta las siguientes líneas:

# import numpy as np

# # arange: similar a range() pero retorna array
# arr1 = np.arange(10)
# print('arange(10):', arr1)

# arr2 = np.arange(5, 15)
# print('arange(5, 15):', arr2)

# arr3 = np.arange(0, 20, 2)
# print('arange(0, 20, 2):', arr3)

# arr4 = np.arange(0, 1, 0.1)
# print('arange(0, 1, 0.1):', arr4)
# print()

# # linspace: puntos equiespaciados
# lin1 = np.linspace(0, 10, 5)
# print('linspace(0, 10, 5):', lin1)

# lin2 = np.linspace(0, 1, 11)
# print('linspace(0, 1, 11):', lin2)

# # Con retstep para ver el paso
# lin3, step = np.linspace(0, 100, 5, retstep=True)
# print(f'linspace(0, 100, 5): {lin3}, step={step}')

print()

# ============================================
# PASO 5: Reshape y manipulación de forma
# ============================================
print('--- Paso 5: Reshape ---')

# Cambiar la forma de un array sin modificar sus datos
# Descomenta las siguientes líneas:

# import numpy as np

# # Crear array 1D
# arr = np.arange(12)
# print('Array original:', arr)
# print('Shape:', arr.shape)
# print()

# # Reshape a 2D (3 filas, 4 columnas)
# matrix_3x4 = arr.reshape((3, 4))
# print('reshape(3, 4):')
# print(matrix_3x4)
# print()

# # Reshape a 2D (4 filas, 3 columnas)
# matrix_4x3 = arr.reshape((4, 3))
# print('reshape(4, 3):')
# print(matrix_4x3)
# print()

# # Usar -1 para calcular automáticamente
# auto_cols = arr.reshape((3, -1))  # 12/3 = 4 columnas
# print('reshape(3, -1):')
# print(auto_cols)
# print()

# # Reshape a 3D
# tensor = arr.reshape((2, 2, 3))
# print('reshape(2, 2, 3):')
# print(tensor)
# print('Shape:', tensor.shape)
# print()

# # Aplanar con flatten y ravel
# flat = matrix_3x4.flatten()
# print('flatten():', flat)

print()

# ============================================
# PASO 6: Especificar dtype
# ============================================
print('--- Paso 6: Tipos de datos (dtype) ---')

# Controlar el tipo de datos del array
# Descomenta las siguientes líneas:

# import numpy as np

# # Crear con dtype específico
# arr_float32 = np.array([1, 2, 3], dtype=np.float32)
# print(f'float32: {arr_float32}, dtype={arr_float32.dtype}')

# arr_int8 = np.array([1, 2, 3], dtype=np.int8)
# print(f'int8: {arr_int8}, dtype={arr_int8.dtype}')

# arr_bool = np.array([1, 0, 1, 0], dtype=bool)
# print(f'bool: {arr_bool}, dtype={arr_bool.dtype}')
# print()

# # zeros y ones con dtype
# zeros_int = np.zeros((2, 2), dtype=int)
# print('zeros con dtype=int:')
# print(zeros_int)
# print()

# # Convertir dtype con astype
# floats = np.array([1.7, 2.3, 3.9, 4.1])
# print(f'Original (float): {floats}')

# ints = floats.astype(int)  # Trunca, no redondea
# print(f'astype(int): {ints}')

# # Comparar uso de memoria
# big = np.zeros((1000, 1000), dtype=np.float64)
# small = np.zeros((1000, 1000), dtype=np.float32)
# print()
# print(f'float64: {big.nbytes / 1024 / 1024:.1f} MB')
# print(f'float32: {small.nbytes / 1024 / 1024:.1f} MB')

print()
print('=' * 50)
print('✅ Ejercicio 01 completado!')
print('=' * 50)
