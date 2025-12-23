# Ejercicio 02: Listas y Tuplas
# Semana 02 - Bootcamp IA

# ============================================
# PASO 1: Crear y Acceder
# ============================================
print('--- Paso 1: Crear y Acceder ---')

# Crea una lista y accede a elementos
# Descomenta las siguientes líneas:

# fruits = ["apple", "banana", "cherry", "date"]
# print(f"Lista: {fruits}")
# print(f"Primero: {fruits[0]}")
# print(f"Último: {fruits[-1]}")
# print(f"Índice 2: {fruits[2]}")

print()

# ============================================
# PASO 2: Slicing
# ============================================
print('--- Paso 2: Slicing ---')

# Extrae partes de la lista con slicing
# Descomenta las siguientes líneas:

# numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# print(f"Original: {numbers}")
# print(f"[2:5]: {numbers[2:5]}")
# print(f"[::2]: {numbers[::2]}")
# print(f"[::-1]: {numbers[::-1]}")

print()

# ============================================
# PASO 3: Métodos de Listas
# ============================================
print('--- Paso 3: Métodos de Listas ---')

# Usa métodos para modificar la lista
# Descomenta las siguientes líneas:

# colors = ["red", "green"]
# print("append, insert, remove...")
#
# colors.append("blue")
# colors.insert(0, "yellow")
# colors.remove("green")
#
# print(f"Lista final: {colors}")

print()

# ============================================
# PASO 4: Comprensiones de Lista
# ============================================
print('--- Paso 4: Comprensiones de Lista ---')

# Crea listas con comprehensions
# Descomenta las siguientes líneas:

# squares = [x ** 2 for x in range(10)]
# print(f"Cuadrados: {squares}")
#
# evens = [x for x in range(20) if x % 2 == 0]
# print(f"Pares: {evens}")

print()

# ============================================
# PASO 5: Tuplas
# ============================================
print('--- Paso 5: Tuplas ---')

# Trabaja con tuplas inmutables
# Descomenta las siguientes líneas:

# point = (10, 20, 30)
# print(f"Punto: {point}")
#
# x, y, z = point
# print(f"Desempaquetado: x={x}, y={y}, z={z}")

print()

# ============================================
# PASO 6: Listas en ML
# ============================================
print('--- Paso 6: Listas en ML ---')

# Normalización min-max
# Descomenta las siguientes líneas:

# values = [10, 20, 30, 40, 50]
# min_val = min(values)
# max_val = max(values)
# normalized = [(v - min_val) / (max_val - min_val) for v in values]
#
# print(f"Original: {values}")
# print(f"Normalizado: {normalized}")

print()

# ============================================
# FIN DEL EJERCICIO
# ============================================
print("=" * 50)
print("¡Ejercicio 02 completado!")
print("Siguiente: ejercicio-03-diccionarios")
print("=" * 50)
