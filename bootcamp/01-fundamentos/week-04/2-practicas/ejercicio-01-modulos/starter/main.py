# ============================================
# EJERCICIO 01: Crear Módulos Propios
# ============================================
# Aprenderás a crear módulos reutilizables
# con funciones, constantes y documentación.
# ============================================

print('=== Ejercicio 01: Módulos Propios ===')
print()

# ============================================
# PASO 1: Crear el Módulo
# ============================================
# Crea un archivo llamado "math_utils.py" en
# esta misma carpeta (starter/) con el siguiente
# contenido. Descomenta y copia al nuevo archivo:

# --- INICIO: Copiar a math_utils.py ---
# """
# Módulo de utilidades matemáticas.
#
# Proporciona funciones para cálculos geométricos
# y constantes matemáticas comunes.
#
# Example:
#     >>> from math_utils import circle_area
#     >>> circle_area(5)
#     78.53975
# """
#
# # Constantes
# PI = 3.14159
# E = 2.71828
#
#
# def circle_area(radius: float) -> float:
#     """
#     Calcula el área de un círculo.
#
#     Args:
#         radius: Radio del círculo.
#
#     Returns:
#         Área del círculo.
#     """
#     return PI * radius ** 2
#
#
# def circle_perimeter(radius: float) -> float:
#     """Calcula el perímetro de un círculo."""
#     return 2 * PI * radius
#
#
# def rectangle_area(width: float, height: float) -> float:
#     """Calcula el área de un rectángulo."""
#     return width * height
#
#
# def factorial(n: int) -> int:
#     """
#     Calcula el factorial de un número.
#
#     Args:
#         n: Número entero no negativo.
#
#     Returns:
#         Factorial de n.
#
#     Raises:
#         ValueError: Si n es negativo.
#     """
#     if n < 0:
#         raise ValueError("n must be non-negative")
#     if n <= 1:
#         return 1
#     return n * factorial(n - 1)
# --- FIN: Copiar a math_utils.py ---

print('--- Paso 1: Crear math_utils.py ---')
print('Crea el archivo math_utils.py con el código de arriba')
print('(descomenta el código y cópialo al nuevo archivo)')
print()

# ============================================
# PASO 2: Agregar if __name__ == "__main__"
# ============================================
# Al final de math_utils.py, agrega:

# --- AGREGAR al final de math_utils.py ---
# if __name__ == "__main__":
#     # Este código SOLO se ejecuta si corres:
#     # python math_utils.py
#     print("=== Math Utils Demo ===")
#     print(f"PI = {PI}")
#     print(f"E = {E}")
#     print()
#     print(f"Área círculo (r=5): {circle_area(5)}")
#     print(f"Perímetro círculo (r=5): {circle_perimeter(5)}")
#     print(f"Área rectángulo (3x4): {rectangle_area(3, 4)}")
#     print(f"5! = {factorial(5)}")
# --- FIN ---

print('--- Paso 2: Agregar __name__ guard ---')
print('Agrega el bloque if __name__ == "__main__" al final')
print('Prueba ejecutando: python math_utils.py')
print()

# ============================================
# PASO 3: Import Completo
# ============================================
print('--- Paso 3: Import Completo ---')

# Descomenta las siguientes líneas después de crear math_utils.py:
# import math_utils
#
# print(f"PI desde módulo: {math_utils.PI}")
# area = math_utils.circle_area(10)
# print(f"Área círculo (r=10): {area}")

print()

# ============================================
# PASO 4: Import Selectivo
# ============================================
print('--- Paso 4: Import Selectivo ---')

# Descomenta las siguientes líneas:
# from math_utils import circle_area, rectangle_area, PI
#
# print(f"PI (import directo): {PI}")
# print(f"Área círculo (r=7): {circle_area(7)}")
# print(f"Área rectángulo (5x8): {rectangle_area(5, 8)}")

print()

# ============================================
# PASO 5: Import con Alias
# ============================================
print('--- Paso 5: Import con Alias ---')

# Descomenta las siguientes líneas:
# import math_utils as mu
#
# print(f"Usando alias 'mu':")
# print(f"E = {mu.E}")
# print(f"6! = {mu.factorial(6)}")

print()

# ============================================
# PASO 6: Explorar el Módulo
# ============================================
print('--- Paso 6: Explorar el Módulo ---')

# Descomenta las siguientes líneas:
# import math_utils
#
# # Ver la ubicación del módulo
# print(f"Archivo: {math_utils.__file__}")
# print(f"Nombre: {math_utils.__name__}")
#
# # Ver la documentación
# print()
# print("Docstring del módulo:")
# print(math_utils.__doc__)
#
# # Listar contenido del módulo
# print()
# print("Contenido del módulo:")
# for item in dir(math_utils):
#     if not item.startswith('_'):
#         print(f"  - {item}")

print()

# ============================================
# PASO 7: Usar help()
# ============================================
print('--- Paso 7: Documentación con help() ---')

# Descomenta para ver la documentación completa:
# import math_utils
# help(math_utils)

# O documentación de una función específica:
# help(math_utils.factorial)

print('Descomenta help(math_utils) para ver la documentación')
print()

# ============================================
# RESUMEN
# ============================================
print('=== Resumen ===')
print('''
Un módulo Python es:
- Un archivo .py con código reutilizable
- Puede contener funciones, clases, constantes
- Se documenta con docstrings
- Usa if __name__ == "__main__" para código de prueba

Formas de importar:
- import module
- from module import item
- import module as alias
- from module import item as alias
''')
