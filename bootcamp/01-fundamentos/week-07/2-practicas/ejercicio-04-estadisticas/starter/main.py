"""
Ejercicio 04: Estadísticas y Álgebra Lineal
===========================================
Aprende agregaciones estadísticas y operaciones matriciales.
"""

import numpy as np

# ============================================
# PASO 1: Agregaciones básicas
# ============================================
print('--- Paso 1: Agregaciones básicas ---')

# Funciones que reducen un array a un escalar
# Descomenta las siguientes líneas:

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# print(f'arr = {arr}')
# print()

# # Agregaciones comunes
# print(f'np.sum(arr) = {np.sum(arr)}')     # Suma total
# print(f'np.prod(arr) = {np.prod(arr)}')   # Producto total
# print(f'np.mean(arr) = {np.mean(arr)}')   # Media (promedio)
# print(f'np.min(arr) = {np.min(arr)}')     # Mínimo
# print(f'np.max(arr) = {np.max(arr)}')     # Máximo
# print()

# # También como métodos del array
# print(f'arr.sum() = {arr.sum()}')
# print(f'arr.mean() = {arr.mean()}')
# print(f'arr.min() = {arr.min()}')
# print(f'arr.max() = {arr.max()}')
# print()

# # Sumas acumulativas
# print(f'np.cumsum(arr) = {np.cumsum(arr)}')  # Suma acumulada
# print(f'np.cumprod([1,2,3,4]) = {np.cumprod([1,2,3,4])}')  # Producto acumulado

print()

# ============================================
# PASO 2: Agregaciones con axis
# ============================================
print('--- Paso 2: Agregaciones con axis ---')

# Agregar a lo largo de un eje específico
# Descomenta las siguientes líneas:

# matrix = np.array([
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12]
# ])
# print('Matrix (3x4):')
# print(matrix)
# print()

# # Sin axis: agrega todo el array
# print(f'sum() total: {matrix.sum()}')
# print()

# # axis=0: agregar columnas (colapsa filas)
# print(f'sum(axis=0) por columnas: {matrix.sum(axis=0)}')
# print(f'mean(axis=0) por columnas: {matrix.mean(axis=0)}')
# print()

# # axis=1: agregar filas (colapsa columnas)
# print(f'sum(axis=1) por filas: {matrix.sum(axis=1)}')
# print(f'mean(axis=1) por filas: {matrix.mean(axis=1)}')
# print()

# # Múltiples funciones
# print(f'min(axis=0): {matrix.min(axis=0)}')
# print(f'max(axis=1): {matrix.max(axis=1)}')

print()

# ============================================
# PASO 3: Estadísticas descriptivas
# ============================================
print('--- Paso 3: Estadísticas descriptivas ---')

# Varianza, desviación estándar, percentiles
# Descomenta las siguientes líneas:

# # Dataset de ejemplo
# scores = np.array([65, 72, 68, 90, 85, 78, 82, 95, 70, 88])
# print(f'Scores: {scores}')
# print()

# # Medidas de tendencia central
# print(f'Media: {np.mean(scores):.2f}')
# print(f'Mediana: {np.median(scores):.2f}')
# print()

# # Medidas de dispersión
# print(f'Varianza: {np.var(scores):.2f}')
# print(f'Desv. Estándar: {np.std(scores):.2f}')
# print(f'Rango: {np.ptp(scores)}')  # peak-to-peak (max - min)
# print()

# # Percentiles y cuartiles
# print(f'Mínimo (P0): {np.percentile(scores, 0)}')
# print(f'Q1 (P25): {np.percentile(scores, 25)}')
# print(f'Mediana (P50): {np.percentile(scores, 50)}')
# print(f'Q3 (P75): {np.percentile(scores, 75)}')
# print(f'Máximo (P100): {np.percentile(scores, 100)}')
# print()

# # Múltiples percentiles a la vez
# percentiles = np.percentile(scores, [10, 25, 50, 75, 90])
# print(f'Percentiles [10,25,50,75,90]: {percentiles}')

print()

# ============================================
# PASO 4: Encontrar valores
# ============================================
print('--- Paso 4: Encontrar valores ---')

# Localizar mínimos, máximos y valores específicos
# Descomenta las siguientes líneas:

# arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
# print(f'arr = {arr}')
# print()

# # Índices de min y max
# print(f'Mínimo: {arr.min()} en índice {arr.argmin()}')
# print(f'Máximo: {arr.max()} en índice {arr.argmax()}')
# print()

# # En matrices
# matrix = np.array([
#     [5, 2, 8],
#     [1, 9, 3],
#     [7, 4, 6]
# ])
# print('Matrix:')
# print(matrix)
# print()

# # argmax/argmin retorna índice plano
# flat_idx = matrix.argmax()
# print(f'Máximo valor: {matrix.max()}')
# print(f'Índice plano: {flat_idx}')
# print(f'Índice 2D: {np.unravel_index(flat_idx, matrix.shape)}')
# print()

# # np.where para encontrar índices
# arr = np.array([1, 5, 2, 8, 3, 9, 4])
# indices = np.where(arr > 4)[0]
# print(f'arr = {arr}')
# print(f'Índices donde arr > 4: {indices}')
# print(f'Valores: {arr[indices]}')
# print()

# # np.where con condición ternaria
# result = np.where(arr > 4, 'alto', 'bajo')
# print(f'Clasificación: {result}')

print()

# ============================================
# PASO 5: Producto de matrices
# ============================================
print('--- Paso 5: Producto de matrices ---')

# Multiplicación matricial
# Descomenta las siguientes líneas:

# # Producto punto de vectores
# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# print(f'a = {a}')
# print(f'b = {b}')
# print(f'Producto punto a·b: {np.dot(a, b)}')  # 1*4 + 2*5 + 3*6 = 32
# print(f'También: a @ b = {a @ b}')
# print()

# # Producto de matrices
# A = np.array([[1, 2],
#               [3, 4]])
# B = np.array([[5, 6],
#               [7, 8]])
# print('A:')
# print(A)
# print('B:')
# print(B)
# print()

# print('A @ B (producto matricial):')
# print(A @ B)
# print()

# # Comparar con multiplicación elemento a elemento
# print('A * B (elemento a elemento):')
# print(A * B)
# print()

# # Matriz × vector
# v = np.array([1, 2])
# print(f'v = {v}')
# print(f'A @ v = {A @ v}')
# print()

# # Matrices no cuadradas
# C = np.array([[1, 2, 3],
#               [4, 5, 6]])  # 2x3
# D = np.array([[1, 2],
#               [3, 4],
#               [5, 6]])     # 3x2
# print('C (2x3) @ D (3x2) = (2x2):')
# print(C @ D)

print()

# ============================================
# PASO 6: Operaciones de álgebra lineal
# ============================================
print('--- Paso 6: Álgebra lineal ---')

# Transpuesta, determinante, inversa
# Descomenta las siguientes líneas:

# A = np.array([[1, 2],
#               [3, 4]])
# print('Matriz A:')
# print(A)
# print()

# # Transpuesta
# print('Transpuesta (A.T):')
# print(A.T)
# print()

# # Determinante
# det = np.linalg.det(A)
# print(f'Determinante: {det:.2f}')
# print()

# # Inversa (si det != 0)
# A_inv = np.linalg.inv(A)
# print('Inversa (A⁻¹):')
# print(A_inv)
# print()

# # Verificar: A @ A⁻¹ = I
# identity = A @ A_inv
# print('A @ A⁻¹ ≈ I:')
# print(np.round(identity, 10))
# print()

# # Traza (suma de diagonal)
# print(f'Traza: {np.trace(A)}')
# print()

# # Eigenvalores y eigenvectores
# eigenvalues, eigenvectors = np.linalg.eig(A)
# print(f'Eigenvalores: {eigenvalues}')
# print('Eigenvectores:')
# print(eigenvectors)
# print()

# # Resolver sistema de ecuaciones: Ax = b
# # x + 2y = 5
# # 3x + 4y = 11
# A = np.array([[1, 2], [3, 4]])
# b = np.array([5, 11])
# x = np.linalg.solve(A, b)
# print(f'Sistema Ax = b, donde b = {b}')
# print(f'Solución x = {x}')

print()

# ============================================
# PASO 7: Caso práctico - Análisis de datos
# ============================================
print('--- Paso 7: Caso práctico ---')

# Análisis de un dataset de estudiantes
# Descomenta las siguientes líneas:

# # Simular notas de 5 estudiantes en 4 materias
# np.random.seed(42)  # Para reproducibilidad
# grades = np.random.randint(50, 100, size=(5, 4))
# subjects = ['Matemáticas', 'Física', 'Química', 'Biología']
# students = ['Ana', 'Bob', 'Carlos', 'Diana', 'Eva']

# print('Notas de estudiantes:')
# print(f'{"":10}', end='')
# for subj in subjects:
#     print(f'{subj:12}', end='')
# print()

# for i, student in enumerate(students):
#     print(f'{student:10}', end='')
#     for grade in grades[i]:
#         print(f'{grade:12}', end='')
#     print()
# print()

# # Estadísticas por estudiante (axis=1)
# print('Promedio por estudiante:')
# for i, student in enumerate(students):
#     avg = grades[i].mean()
#     print(f'  {student}: {avg:.1f}')
# print()

# # Estadísticas por materia (axis=0)
# print('Estadísticas por materia:')
# for j, subj in enumerate(subjects):
#     col = grades[:, j]
#     print(f'  {subj}: media={col.mean():.1f}, std={col.std():.1f}')
# print()

# # Mejor y peor estudiante
# promedios = grades.mean(axis=1)
# mejor_idx = promedios.argmax()
# peor_idx = promedios.argmin()
# print(f'Mejor estudiante: {students[mejor_idx]} ({promedios[mejor_idx]:.1f})')
# print(f'Peor estudiante: {students[peor_idx]} ({promedios[peor_idx]:.1f})')
# print()

# # Estudiantes con promedio > 70
# aprobados = np.where(promedios > 70)[0]
# print(f'Estudiantes con promedio > 70: {[students[i] for i in aprobados]}')

print()
print('=' * 50)
print('✅ Ejercicio 04 completado!')
print('=' * 50)
