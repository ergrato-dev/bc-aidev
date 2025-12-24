"""
Ejercicio 02: Indexing y Slicing
================================
Aprende a acceder y manipular elementos en arrays NumPy.
"""

import numpy as np

# ============================================
# PASO 1: Indexing básico 1D
# ============================================
print('--- Paso 1: Indexing básico 1D ---')

# Acceder a elementos con índices positivos y negativos
# Descomenta las siguientes líneas:

# arr = np.array([10, 20, 30, 40, 50])
# print('Array:', arr)
# print()

# # Índices positivos (desde el inicio)
# print(f'arr[0] = {arr[0]}')   # Primer elemento
# print(f'arr[2] = {arr[2]}')   # Tercer elemento
# print(f'arr[4] = {arr[4]}')   # Quinto elemento
# print()

# # Índices negativos (desde el final)
# print(f'arr[-1] = {arr[-1]}')  # Último elemento
# print(f'arr[-2] = {arr[-2]}')  # Penúltimo
# print(f'arr[-5] = {arr[-5]}')  # Primero (desde atrás)

print()

# ============================================
# PASO 2: Indexing 2D y 3D
# ============================================
print('--- Paso 2: Indexing 2D y 3D ---')

# Arrays multidimensionales usan múltiples índices
# Descomenta las siguientes líneas:

# # Crear matriz 3x3
# matrix = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])
# print('Matrix:')
# print(matrix)
# print()

# # Acceder a elementos [fila, columna]
# print(f'matrix[0, 0] = {matrix[0, 0]}')  # Esquina superior izquierda
# print(f'matrix[0, 2] = {matrix[0, 2]}')  # Esquina superior derecha
# print(f'matrix[1, 1] = {matrix[1, 1]}')  # Centro
# print(f'matrix[2, 0] = {matrix[2, 0]}')  # Esquina inferior izquierda
# print(f'matrix[-1, -1] = {matrix[-1, -1]}')  # Última posición
# print()

# # Array 3D
# tensor = np.arange(24).reshape((2, 3, 4))
# print('Tensor shape:', tensor.shape)
# print(f'tensor[0, 0, 0] = {tensor[0, 0, 0]}')
# print(f'tensor[1, 2, 3] = {tensor[1, 2, 3]}')

print()

# ============================================
# PASO 3: Slicing básico
# ============================================
print('--- Paso 3: Slicing básico ---')

# Extraer secciones con start:stop:step
# Descomenta las siguientes líneas:

# arr = np.arange(10)
# print('Array:', arr)
# print()

# # Slicing básico [start:stop] - stop no incluido
# print(f'arr[2:7] = {arr[2:7]}')    # Elementos 2 al 6
# print(f'arr[:5] = {arr[:5]}')      # Primeros 5
# print(f'arr[5:] = {arr[5:]}')      # Desde el 5 hasta el final
# print()

# # Con step [start:stop:step]
# print(f'arr[::2] = {arr[::2]}')    # Cada 2 elementos
# print(f'arr[1::2] = {arr[1::2]}')  # Impares
# print(f'arr[::3] = {arr[::3]}')    # Cada 3 elementos
# print()

# # Step negativo (invertir)
# print(f'arr[::-1] = {arr[::-1]}')  # Invertir array
# print(f'arr[::-2] = {arr[::-2]}')  # Invertir cada 2

print()

# ============================================
# PASO 4: Slicing 2D
# ============================================
print('--- Paso 4: Slicing 2D ---')

# Combinar slicing en filas y columnas
# Descomenta las siguientes líneas:

# matrix = np.arange(20).reshape((4, 5))
# print('Matrix:')
# print(matrix)
# print()

# # Seleccionar filas
# print('matrix[0] - Primera fila:')
# print(matrix[0])

# print('matrix[1:3] - Filas 1 y 2:')
# print(matrix[1:3])
# print()

# # Seleccionar columnas
# print('matrix[:, 0] - Primera columna:', matrix[:, 0])
# print('matrix[:, -1] - Última columna:', matrix[:, -1])
# print('matrix[:, 1:4] - Columnas 1, 2, 3:')
# print(matrix[:, 1:4])
# print()

# # Submatrices
# print('matrix[1:3, 2:4] - Submatriz:')
# print(matrix[1:3, 2:4])

# print('matrix[::2, ::2] - Filas y columnas alternas:')
# print(matrix[::2, ::2])

print()

# ============================================
# PASO 5: Fancy Indexing
# ============================================
print('--- Paso 5: Fancy Indexing ---')

# Usar arrays de índices para selección múltiple
# Descomenta las siguientes líneas:

# # 1D Fancy Indexing
# arr = np.array([10, 20, 30, 40, 50, 60, 70])
# print('Array:', arr)

# indices = [0, 2, 5]
# print(f'arr[[0, 2, 5]] = {arr[indices]}')

# indices = [6, 4, 2, 0]
# print(f'arr[[6, 4, 2, 0]] = {arr[indices]}')  # Cualquier orden
# print()

# # 2D Fancy Indexing
# matrix = np.arange(12).reshape((3, 4))
# print('Matrix:')
# print(matrix)
# print()

# # Seleccionar filas específicas
# print('matrix[[0, 2]] - Filas 0 y 2:')
# print(matrix[[0, 2]])

# # Seleccionar elementos específicos
# rows = [0, 1, 2]
# cols = [0, 2, 3]
# print(f'matrix[{rows}, {cols}] = {matrix[rows, cols]}')  # Elementos (0,0), (1,2), (2,3)

print()

# ============================================
# PASO 6: Boolean Indexing
# ============================================
print('--- Paso 6: Boolean Indexing ---')

# Filtrar elementos con máscaras booleanas
# Descomenta las siguientes líneas:

# arr = np.array([1, 5, 2, 8, 3, 9, 4, 7, 6])
# print('Array:', arr)
# print()

# # Crear máscara booleana
# mask = arr > 5
# print(f'Máscara (arr > 5): {mask}')
# print(f'Elementos > 5: {arr[mask]}')
# print()

# # Condiciones directas (más común)
# print(f'arr[arr < 4] = {arr[arr < 4]}')   # Menores que 4
# print(f'arr[arr >= 5] = {arr[arr >= 5]}') # Mayores o iguales a 5
# print()

# # Condiciones múltiples
# print(f'arr[(arr > 2) & (arr < 7)] = {arr[(arr > 2) & (arr < 7)]}')  # AND
# print(f'arr[(arr < 3) | (arr > 7)] = {arr[(arr < 3) | (arr > 7)]}')  # OR
# print(f'arr[~(arr > 5)] = {arr[~(arr > 5)]}')  # NOT
# print()

# # Aplicar a matrices
# matrix = np.arange(12).reshape((3, 4))
# print('Matrix:')
# print(matrix)
# print(f'Elementos > 5: {matrix[matrix > 5]}')  # Retorna 1D

print()

# ============================================
# PASO 7: Modificar con indexing
# ============================================
print('--- Paso 7: Modificar con indexing ---')

# Asignar valores usando indexing y slicing
# Descomenta las siguientes líneas:

# # Modificar elemento individual
# arr = np.array([1, 2, 3, 4, 5])
# print('Original:', arr)

# arr[0] = 100
# print('arr[0] = 100:', arr)

# arr[-1] = 500
# print('arr[-1] = 500:', arr)
# print()

# # Modificar con slicing
# arr = np.arange(10)
# print('Original:', arr)

# arr[2:5] = [20, 30, 40]
# print('arr[2:5] = [20, 30, 40]:', arr)

# arr[::2] = 0
# print('arr[::2] = 0:', arr)
# print()

# # Modificar con boolean indexing
# arr = np.array([1, -2, 3, -4, 5, -6])
# print('Original:', arr)

# arr[arr < 0] = 0  # Reemplazar negativos
# print('arr[arr < 0] = 0:', arr)
# print()

# # Modificar matriz
# matrix = np.arange(12).reshape((3, 4))
# print('Matrix original:')
# print(matrix)

# matrix[1, :] = 99  # Toda la fila 1
# print('matrix[1, :] = 99:')
# print(matrix)

# matrix[:, 0] = -1  # Toda la columna 0
# print('matrix[:, 0] = -1:')
# print(matrix)

print()
print('=' * 50)
print('✅ Ejercicio 02 completado!')
print('=' * 50)
