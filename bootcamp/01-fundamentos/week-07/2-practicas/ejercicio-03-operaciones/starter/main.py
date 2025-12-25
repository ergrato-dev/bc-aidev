"""
Ejercicio 03: Operaciones Vectorizadas
======================================
Aprende operaciones elemento a elemento y funciones universales.
"""

import numpy as np

# ============================================
# PASO 1: Operaciones aritméticas básicas
# ============================================
print('--- Paso 1: Operaciones aritméticas básicas ---')

# Las operaciones se aplican elemento a elemento
# Descomenta las siguientes líneas:

# a = np.array([1, 2, 3, 4])
# b = np.array([10, 20, 30, 40])
# print(f'a = {a}')
# print(f'b = {b}')
# print()

# # Operaciones básicas
# print(f'a + b = {a + b}')   # Suma
# print(f'a - b = {a - b}')   # Resta
# print(f'a * b = {a * b}')   # Multiplicación
# print(f'b / a = {b / a}')   # División
# print(f'b // a = {b // a}') # División entera
# print(f'b % a = {b % a}')   # Módulo
# print()

# # Potencias
# print(f'a ** 2 = {a ** 2}')     # Cuadrado
# print(f'a ** 0.5 = {a ** 0.5}') # Raíz cuadrada
# print(f'2 ** a = {2 ** a}')     # 2 elevado a cada elemento

print()

# ============================================
# PASO 2: Operaciones con escalares
# ============================================
print('--- Paso 2: Operaciones con escalares ---')

# Los escalares se aplican a todos los elementos
# Descomenta las siguientes líneas:

# arr = np.array([1, 2, 3, 4, 5])
# print(f'arr = {arr}')
# print()

# # Escalar se "expande" a todos los elementos
# print(f'arr + 10 = {arr + 10}')
# print(f'arr - 5 = {arr - 5}')
# print(f'arr * 2 = {arr * 2}')
# print(f'arr / 10 = {arr / 10}')
# print(f'arr ** 3 = {arr ** 3}')
# print()

# # Caso práctico: normalización
# data = np.array([10, 20, 30, 40, 50])
# mean = data.mean()
# std = data.std()
# normalized = (data - mean) / std
# print(f'Original: {data}')
# print(f'Media: {mean}, Std: {std:.2f}')
# print(f'Normalizado: {normalized}')

print()

# ============================================
# PASO 3: Broadcasting con diferentes shapes
# ============================================
print('--- Paso 3: Broadcasting ---')

# NumPy expande arrays automáticamente para operaciones
# Descomenta las siguientes líneas:

# # Broadcasting: escalar con array
# matrix = np.ones((3, 4))
# print('Matrix (3x4):')
# print(matrix)
# print()

# result = matrix * 5
# print('matrix * 5:')
# print(result)
# print()

# # Broadcasting: 1D con 2D
# row = np.array([1, 2, 3, 4])  # Shape (4,)
# print(f'Row: {row}')
# result = matrix + row
# print('matrix + row (suma a cada fila):')
# print(result)
# print()

# # Broadcasting: columna con matriz
# col = np.array([[10], [20], [30]])  # Shape (3, 1)
# print('Columna:')
# print(col)
# result = matrix + col
# print('matrix + col (suma a cada columna):')
# print(result)
# print()

# # Caso práctico: distancia de cada punto al origen
# points = np.array([[1, 2], [3, 4], [5, 6]])  # 3 puntos 2D
# origin = np.array([0, 0])
# distances = np.sqrt(np.sum((points - origin) ** 2, axis=1))
# print(f'Puntos:\n{points}')
# print(f'Distancias al origen: {distances}')

print()

# ============================================
# PASO 4: Funciones matemáticas (ufuncs)
# ============================================
print('--- Paso 4: Funciones matemáticas ---')

# Funciones universales optimizadas
# Descomenta las siguientes líneas:

# arr = np.array([1, 4, 9, 16, 25])
# print(f'arr = {arr}')
# print()

# # Raíces y potencias
# print(f'np.sqrt(arr) = {np.sqrt(arr)}')
# print(f'np.square(arr) = {np.square(arr)}')
# print(f'np.power(arr, 3) = {np.power(arr, 3)}')
# print()

# # Exponenciales y logaritmos
# arr2 = np.array([1, 2, 3, 4])
# print(f'np.exp(arr2) = {np.exp(arr2)}')      # e^x
# print(f'np.log(arr2) = {np.log(arr2)}')      # ln(x)
# print(f'np.log10(arr2) = {np.log10(arr2)}')  # log10(x)
# print(f'np.log2(arr2) = {np.log2(arr2)}')    # log2(x)
# print()

# # Valor absoluto y signo
# arr3 = np.array([-3, -1, 0, 2, 5])
# print(f'arr3 = {arr3}')
# print(f'np.abs(arr3) = {np.abs(arr3)}')
# print(f'np.sign(arr3) = {np.sign(arr3)}')
# print()

# # Redondeo
# floats = np.array([1.2, 2.5, 3.7, 4.1, 5.9])
# print(f'floats = {floats}')
# print(f'np.floor(floats) = {np.floor(floats)}')  # Hacia abajo
# print(f'np.ceil(floats) = {np.ceil(floats)}')    # Hacia arriba
# print(f'np.round(floats) = {np.round(floats)}')  # Al más cercano

print()

# ============================================
# PASO 5: Funciones trigonométricas
# ============================================
print('--- Paso 5: Funciones trigonométricas ---')

# Trabajan con ángulos en radianes
# Descomenta las siguientes líneas:

# # Ángulos comunes
# angles_deg = np.array([0, 30, 45, 60, 90, 180])
# angles_rad = np.deg2rad(angles_deg)  # Convertir a radianes
# print(f'Grados: {angles_deg}')
# print(f'Radianes: {np.round(angles_rad, 4)}')
# print()

# # Funciones trigonométricas
# print(f'sin: {np.round(np.sin(angles_rad), 4)}')
# print(f'cos: {np.round(np.cos(angles_rad), 4)}')
# print()

# # Funciones inversas
# values = np.array([0, 0.5, 1])
# print(f'arcsin({values}) en grados: {np.rad2deg(np.arcsin(values))}')
# print()

# # Caso práctico: generar onda senoidal
# t = np.linspace(0, 2 * np.pi, 9)  # Un ciclo completo
# wave = np.sin(t)
# print('Onda senoidal:')
# for time, value in zip(t, wave):
#     bar = '█' * int((value + 1) * 10)
#     print(f't={time:.2f}: {value:+.2f} {bar}')

print()

# ============================================
# PASO 6: Comparaciones y operaciones lógicas
# ============================================
print('--- Paso 6: Comparaciones y lógicas ---')

# Comparaciones retornan arrays booleanos
# Descomenta las siguientes líneas:

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# print(f'arr = {arr}')
# print()

# # Comparaciones elemento a elemento
# print(f'arr > 4: {arr > 4}')
# print(f'arr <= 3: {arr <= 3}')
# print(f'arr == 5: {arr == 5}')
# print(f'arr != 5: {arr != 5}')
# print()

# # Combinaciones lógicas
# print(f'(arr > 2) & (arr < 6): {(arr > 2) & (arr < 6)}')  # AND
# print(f'(arr < 3) | (arr > 6): {(arr < 3) | (arr > 6)}')  # OR
# print(f'~(arr > 5): {~(arr > 5)}')  # NOT
# print()

# # Funciones de agregación booleana
# print(f'np.all(arr > 0): {np.all(arr > 0)}')  # ¿Todos cumplen?
# print(f'np.any(arr > 7): {np.any(arr > 7)}')  # ¿Alguno cumple?
# print(f'np.sum(arr > 3): {np.sum(arr > 3)}')  # Contar True
# print()

# # Caso práctico: encontrar valores
# data = np.array([23, 45, 12, 67, 34, 89, 11])
# print(f'data = {data}')
# print(f'Índices donde data > 30: {np.where(data > 30)[0]}')
# print(f'Valores donde data > 30: {data[data > 30]}')

print()

# ============================================
# PASO 7: Operaciones in-place vs copias
# ============================================
print('--- Paso 7: In-place vs copias ---')

# Entender cuándo se modifica el original
# Descomenta las siguientes líneas:

# # Operación normal: crea copia
# arr = np.array([1, 2, 3, 4, 5])
# print(f'Original: {arr}')

# result = arr + 10
# print(f'arr + 10: {result}')
# print(f'arr después: {arr}')  # No cambia
# print()

# # Operación in-place: modifica original
# arr2 = np.array([1, 2, 3, 4, 5])
# print(f'Original: {arr2}')

# arr2 += 10  # Equivale a arr2 = arr2 + 10, pero in-place
# print(f'arr2 += 10: {arr2}')
# print()

# # Otras operaciones in-place
# arr3 = np.array([2, 4, 6, 8])
# print(f'Original: {arr3}')
# arr3 *= 2
# print(f'arr3 *= 2: {arr3}')
# arr3 //= 2
# print(f'arr3 //= 2: {arr3}')
# print()

# # np.add con out parameter (avanzado)
# a = np.array([1, 2, 3])
# b = np.array([10, 20, 30])
# out = np.empty(3)
# np.add(a, b, out=out)
# print(f'np.add(a, b, out=out): {out}')

print()
print('=' * 50)
print('✅ Ejercicio 03 completado!')
print('=' * 50)
