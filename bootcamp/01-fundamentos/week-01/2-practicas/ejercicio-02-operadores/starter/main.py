# ============================================
# EJERCICIO 02: Operadores
# ============================================
# Descomenta las líneas indicadas en cada paso
# Ejecuta después de cada paso para ver resultados
# ============================================

# ============================================
# PASO 1: Operadores Aritméticos Básicos
# ============================================
print('--- Paso 1: Operadores Aritméticos Básicos ---')

# Descomenta las siguientes líneas:
# a = 10
# b = 3

# suma = a + b
# resta = a - b
# multiplicacion = a * b
# division = a / b  # Siempre retorna float

# print(f"{a} + {b} = {suma}")
# print(f"{a} - {b} = {resta}")
# print(f"{a} * {b} = {multiplicacion}")
# print(f"{a} / {b} = {division}")

print()

# ============================================
# PASO 2: División Entera, Módulo y Potencia
# ============================================
print('--- Paso 2: División Entera, Módulo y Potencia ---')

# Descomenta las siguientes líneas:
# a = 10
# b = 3

# division_entera = a // b  # Trunca decimales
# modulo = a % b            # Resto de la división
# potencia = a ** b         # a elevado a b

# print(f"{a} // {b} = {division_entera} (división entera)")
# print(f"{a} % {b} = {modulo} (módulo/resto)")
# print(f"{a} ** {b} = {potencia} (potencia)")

# # Uso práctico del módulo: verificar par/impar
# numero = 7
# es_par = numero % 2 == 0
# print(f"¿{numero} es par? {es_par}")

print()

# ============================================
# PASO 3: Precedencia de Operadores
# ============================================
print('--- Paso 3: Precedencia de Operadores ---')

# Descomenta las siguientes líneas:
# # Sin paréntesis: sigue orden matemático
# resultado1 = 2 + 3 * 4
# print(f"2 + 3 * 4 = {resultado1} (multiplicación primero)")

# # Con paréntesis: cambia el orden
# resultado2 = (2 + 3) * 4
# print(f"(2 + 3) * 4 = {resultado2} (suma primero)")

# # Ejemplo más complejo
# resultado3 = 2 ** 3 * 4 + 5
# print(f"2 ** 3 * 4 + 5 = {resultado3}")
# print("  → 8 * 4 + 5 = 32 + 5 = 37")

print()

# ============================================
# PASO 4: Operadores de Comparación
# ============================================
print('--- Paso 4: Operadores de Comparación ---')

# Descomenta las siguientes líneas:
# x = 10
# y = 5

# print(f"{x} == {y}: {x == y}")   # Igual a
# print(f"{x} != {y}: {x != y}")   # Diferente de
# print(f"{x} > {y}: {x > y}")     # Mayor que
# print(f"{x} < {y}: {x < y}")     # Menor que
# print(f"{x} >= {y}: {x >= y}")   # Mayor o igual
# print(f"{x} <= {y}: {x <= y}")   # Menor o igual

# # Comparaciones encadenadas (Pythonic)
# edad = 25
# en_rango = 18 <= edad <= 65
# print(f"¿{edad} está entre 18 y 65? {en_rango}")

print()

# ============================================
# PASO 5: Operadores Lógicos
# ============================================
print('--- Paso 5: Operadores Lógicos ---')

# Descomenta las siguientes líneas:
# # AND: ambos deben ser True
# print(f"True and True = {True and True}")
# print(f"True and False = {True and False}")

# # OR: al menos uno True
# print(f"True or False = {True or False}")
# print(f"False or False = {False or False}")

# # NOT: invierte el valor
# print(f"not True = {not True}")
# print(f"not False = {not False}")

# # Ejemplo práctico
# age = 25
# has_license = True
# can_drive = age >= 18 and has_license
# print(f"¿Puede conducir? (edad={age}, licencia={has_license}): {can_drive}")

print()

# ============================================
# PASO 6: Operadores de Asignación Compuesta
# ============================================
print('--- Paso 6: Operadores de Asignación Compuesta ---')

# Descomenta las siguientes líneas:
# x = 10
# print(f"Valor inicial: x = {x}")

# x += 5  # x = x + 5
# print(f"x += 5 → x = {x}")

# x -= 3  # x = x - 3
# print(f"x -= 3 → x = {x}")

# x *= 2  # x = x * 2
# print(f"x *= 2 → x = {x}")

# x //= 4  # x = x // 4
# print(f"x //= 4 → x = {x}")

# x **= 2  # x = x ** 2
# print(f"x **= 2 → x = {x}")

print()

# ============================================
# PASO 7: Operadores de Identidad y Membresía
# ============================================
print('--- Paso 7: Identidad y Membresía ---')

# Descomenta las siguientes líneas:
# # Identidad: is / is not
# valor = None
# print(f"valor is None: {valor is None}")
# print(f"valor is not None: {valor is not None}")

# # Membresía: in / not in
# numeros = [1, 2, 3, 4, 5]
# print(f"3 in {numeros}: {3 in numeros}")
# print(f"10 in {numeros}: {10 in numeros}")
# print(f"10 not in {numeros}: {10 not in numeros}")

# # En strings
# texto = "Hola Mundo"
# print(f"'Mundo' in '{texto}': {'Mundo' in texto}")
# print(f"'Python' in '{texto}': {'Python' in texto}")

print()

# ============================================
# ¡FELICIDADES! Has completado el ejercicio 02
# ============================================
print("=" * 50)
print("✅ Ejercicio 02 completado")
print("=" * 50)
