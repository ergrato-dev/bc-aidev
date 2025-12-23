"""
Ejercicio 01: Funciones
Semana 02 - Bootcamp IA

Instrucciones:
- Descomenta cada sección según avanzas en el README
- Ejecuta después de cada paso para ver los resultados
"""

# ============================================
# PASO 1: Función Básica
# ============================================
print('--- Paso 1: Función Básica ---')

# Define una función que retorna un saludo
# Descomenta las siguientes líneas:

# def greet(name: str) -> str:
#     """Return a greeting message."""
#     return f"Hello, {name}!"
#
# message = greet("Ana")
# print(message)
#
# # Llamar con otro argumento
# print(greet("World"))

print()

# ============================================
# PASO 2: Parámetros por Defecto
# ============================================
print('--- Paso 2: Parámetros por Defecto ---')

# Función con valor por defecto para exponent
# Descomenta las siguientes líneas:

# def power(base: int, exponent: int = 2) -> int:
#     """Calculate base raised to exponent."""
#     return base ** exponent
#
# # Sin segundo argumento - usa default (2)
# print(f"5^2 = {power(5)}")
#
# # Con segundo argumento
# print(f"2^10 = {power(2, 10)}")
#
# # Con keyword argument
# print(f"3^3 = {power(3, exponent=3)}")

print()

# ============================================
# PASO 3: Retorno Múltiple
# ============================================
print('--- Paso 3: Retorno Múltiple ---')

# Función que retorna múltiples valores
# Descomenta las siguientes líneas:

# def get_stats(numbers: list) -> tuple:
#     """Return min, max, and average of a list."""
#     minimum = min(numbers)
#     maximum = max(numbers)
#     average = sum(numbers) / len(numbers)
#     return minimum, maximum, average
#
# # Desempaquetado de tupla
# data = [10, 20, 30, 40, 50]
# minimum, maximum, average = get_stats(data)
#
# print(f"Números: {data}")
# print(f"Min: {minimum}, Max: {maximum}, Avg: {average}")

print()

# ============================================
# PASO 4: *args y **kwargs
# ============================================
print('--- Paso 4: *args y **kwargs ---')

# *args: acepta cualquier número de argumentos posicionales
# **kwargs: acepta cualquier número de argumentos keyword
# Descomenta las siguientes líneas:

# def sum_all(*args) -> int:
#     """Sum any number of arguments."""
#     return sum(args)
#
# def print_info(**kwargs) -> None:
#     """Print key-value pairs."""
#     for key, value in kwargs.items():
#         print(f"{key}: {value}")
#
# # Probar *args
# print(f"sum_all(1, 2, 3) = {sum_all(1, 2, 3)}")
# print(f"sum_all(1, 2, 3, 4, 5) = {sum_all(1, 2, 3, 4, 5)}")
#
# # Probar **kwargs
# print_info(name="Ana", age=25, city="Madrid")

print()

# ============================================
# PASO 5: Funciones Lambda
# ============================================
print('--- Paso 5: Funciones Lambda ---')

# Funciones anónimas de una línea
# Descomenta las siguientes líneas:

# # Lambda simple
# square = lambda x: x ** 2
# print(f"square(5) = {square(5)}")
#
# # Lambda con sorted
# words = ["Python", "is", "awesome"]
# by_length = sorted(words, key=lambda w: len(w))
# print(f"Ordenadas por longitud: {by_length}")

print()

# ============================================
# PASO 6: Función para ML
# ============================================
print('--- Paso 6: Función para ML ---')

# Implementa accuracy para clasificación
# Descomenta las siguientes líneas:

# def calculate_accuracy(y_true: list, y_pred: list) -> float:
#     """
#     Calculate classification accuracy.
#
#     Args:
#         y_true: True labels
#         y_pred: Predicted labels
#
#     Returns:
#         Accuracy score between 0 and 1
#     """
#     if len(y_true) != len(y_pred):
#         raise ValueError("Lists must have same length")
#
#     correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
#     return correct / len(y_true)
#
# # Datos de ejemplo
# true_labels = [1, 0, 1, 1, 0, 1]
# predictions = [1, 0, 0, 1, 0, 1]
#
# accuracy = calculate_accuracy(true_labels, predictions)
# print(f"Accuracy: {accuracy:.2%}")

print()

# ============================================
# FIN DEL EJERCICIO
# ============================================
print("=" * 50)
print("¡Ejercicio 01 completado!")
print("Siguiente: ejercicio-02-listas")
print("=" * 50)
