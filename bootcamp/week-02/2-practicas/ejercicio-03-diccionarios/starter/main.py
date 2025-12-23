# Ejercicio 03: Diccionarios y Sets
# Semana 02 - Bootcamp IA

# ============================================
# PASO 1: Crear y Acceder Diccionarios
# ============================================
print('--- Paso 1: Crear y Acceder ---')

# Crea un diccionario y accede a valores
# Descomenta las siguientes líneas:

# person = {"name": "Ana", "age": 25, "city": "Madrid"}
# print(f"Persona: {person}")
# print(f"Nombre: {person['name']}")
# print(f"Email: {person.get('email', 'N/A')}")

print()

# ============================================
# PASO 2: Modificar Diccionarios
# ============================================
print('--- Paso 2: Modificar ---')

# Agrega, modifica y elimina
# Descomenta las siguientes líneas:

# person = {"name": "Ana", "age": 25, "city": "Madrid"}
#
# person["email"] = "ana@mail.com"
# person["age"] = 26
# del person["city"]
#
# print("Agregado email, modificado age, eliminado city")
# print(f"Resultado: {person}")

print()

# ============================================
# PASO 3: Iterar Diccionarios
# ============================================
print('--- Paso 3: Iterar ---')

# Recorre el diccionario
# Descomenta las siguientes líneas:

# person = {"name": "Ana", "age": 26, "email": "ana@mail.com"}
#
# for key, value in person.items():
#     print(f"{key}: {value}")

print()

# ============================================
# PASO 4: Comprensiones de Diccionario
# ============================================
print('--- Paso 4: Comprensiones ---')

# Crea diccionarios con comprehensions
# Descomenta las siguientes líneas:

# squares = {x: x**2 for x in range(6)}
# print(f"Cuadrados: {squares}")

print()

# ============================================
# PASO 5: Sets Básicos
# ============================================
print('--- Paso 5: Sets Básicos ---')

# Trabaja con conjuntos
# Descomenta las siguientes líneas:

# numbers = {1, 2, 3, 4, 5}
# print(f"Set inicial: {numbers}")
#
# numbers.add(6)
# print(f"Después de add(6): {numbers}")
#
# numbers.discard(1)
# print(f"Después de discard(1): {numbers}")

print()

# ============================================
# PASO 6: Operaciones de Sets
# ============================================
print('--- Paso 6: Operaciones de Sets ---')

# Unión, intersección, diferencia
# Descomenta las siguientes líneas:

# A = {1, 2, 3, 4}
# B = {3, 4, 5, 6}
#
# print(f"A = {A}")
# print(f"B = {B}")
# print(f"Unión: {A | B}")
# print(f"Intersección: {A & B}")
# print(f"Diferencia A-B: {A - B}")

print()

# ============================================
# FIN DEL EJERCICIO
# ============================================
print("=" * 50)
print("¡Ejercicio 03 completado!")
print("Siguiente: ejercicio-04-integrador")
print("=" * 50)
