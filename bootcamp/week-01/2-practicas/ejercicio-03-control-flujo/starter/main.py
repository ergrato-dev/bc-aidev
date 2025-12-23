# ============================================
# EJERCICIO 03: Control de Flujo
# ============================================
# Descomenta las líneas indicadas en cada paso
# Ejecuta después de cada paso para ver resultados
# ============================================

# ============================================
# PASO 1: Condicionales if/elif/else
# ============================================
print('--- Paso 1: Condicionales if/elif/else ---')

# Descomenta las siguientes líneas:
# nota = 85

# if nota >= 90:
#     calificacion = "Excelente"
# elif nota >= 80:
#     calificacion = "Bueno"
# elif nota >= 70:
#     calificacion = "Aprobado"
# else:
#     calificacion = "Reprobado"

# print(f"Nota: {nota} → Calificación: {calificacion}")

# # Prueba con diferentes valores
# accuracy = 0.75
# if accuracy >= 0.9:
#     status = "Modelo excelente"
# elif accuracy >= 0.7:
#     status = "Modelo aceptable"
# else:
#     status = "Modelo necesita mejora"

# print(f"Accuracy: {accuracy} → Status: {status}")

print()

# ============================================
# PASO 2: Operador Ternario
# ============================================
print('--- Paso 2: Operador Ternario ---')

# Descomenta las siguientes líneas:
# # Forma tradicional
# edad = 20
# if edad >= 18:
#     estado = "Mayor de edad"
# else:
#     estado = "Menor de edad"
# print(f"Tradicional: {estado}")

# # Forma ternaria (una línea)
# estado_ternario = "Mayor de edad" if edad >= 18 else "Menor de edad"
# print(f"Ternario: {estado_ternario}")

# # Útil para asignaciones condicionales
# valor = None
# resultado = valor if valor is not None else "Valor por defecto"
# print(f"Resultado con valor None: {resultado}")

print()

# ============================================
# PASO 3: Bucle for con range()
# ============================================
print('--- Paso 3: Bucle for con range() ---')

# Descomenta las siguientes líneas:
# # range(n): 0 hasta n-1
# print("range(5):", end=" ")
# for i in range(5):
#     print(i, end=" ")
# print()

# # range(inicio, fin): inicio hasta fin-1
# print("range(1, 6):", end=" ")
# for i in range(1, 6):
#     print(i, end=" ")
# print()

# # range(inicio, fin, paso): con incremento
# print("range(0, 10, 2):", end=" ")
# for i in range(0, 10, 2):
#     print(i, end=" ")
# print()

# # Cuenta regresiva
# print("Cuenta regresiva:", end=" ")
# for i in range(5, 0, -1):
#     print(i, end=" ")
# print("¡Despegue!")

print()

# ============================================
# PASO 4: for con Listas y enumerate()
# ============================================
print('--- Paso 4: for con Listas y enumerate() ---')

# Descomenta las siguientes líneas:
# # Iterar sobre lista
# frutas = ["manzana", "banana", "cereza"]
# print("Frutas:")
# for fruta in frutas:
#     print(f"  - {fruta}")

# # Con enumerate para obtener índice
# print("\nFrutas con índice:")
# for i, fruta in enumerate(frutas):
#     print(f"  {i}: {fruta}")

# # enumerate con inicio personalizado
# print("\nFrutas numeradas desde 1:")
# for num, fruta in enumerate(frutas, start=1):
#     print(f"  {num}. {fruta}")

print()

# ============================================
# PASO 5: Bucle while
# ============================================
print('--- Paso 5: Bucle while ---')

# Descomenta las siguientes líneas:
# # While básico
# contador = 0
# print("Contando hasta 5:")
# while contador < 5:
#     print(f"  contador = {contador}")
#     contador += 1

# # Simular entrenamiento de modelo
# print("\nSimulación de entrenamiento:")
# loss = 1.0
# epoch = 0
# while loss > 0.1 and epoch < 10:
#     loss *= 0.7  # Reducir loss
#     epoch += 1
#     print(f"  Epoch {epoch}: loss = {loss:.4f}")

# print(f"Entrenamiento completado en {epoch} epochs")

print()

# ============================================
# PASO 6: break y continue
# ============================================
print('--- Paso 6: break y continue ---')

# Descomenta las siguientes líneas:
# # break: terminar el bucle
# print("Buscar primer número par:")
# numeros = [1, 3, 5, 4, 7, 9]
# for num in numeros:
#     if num % 2 == 0:
#         print(f"  Encontrado: {num}")
#         break
#     print(f"  {num} no es par, continuando...")

# # continue: saltar iteración
# print("\nSolo números impares (skip pares):")
# for i in range(10):
#     if i % 2 == 0:
#         continue
#     print(f"  {i}", end=" ")
# print()

# # else en bucle: se ejecuta si NO hubo break
# print("\nBuscar 10 en la lista:")
# numeros = [1, 2, 3, 4, 5]
# for num in numeros:
#     if num == 10:
#         print("  ¡Encontrado!")
#         break
# else:
#     print("  No encontrado (else ejecutado)")

print()

# ============================================
# PASO 7: Comprensiones de Lista
# ============================================
print('--- Paso 7: Comprensiones de Lista ---')

# Descomenta las siguientes líneas:
# # Crear lista de cuadrados
# cuadrados = [x ** 2 for x in range(6)]
# print(f"Cuadrados: {cuadrados}")

# # Filtrar con condición
# pares = [x for x in range(10) if x % 2 == 0]
# print(f"Pares: {pares}")

# # Transformar elementos
# nombres = ["ana", "bob", "carlos"]
# mayusculas = [nombre.upper() for nombre in nombres]
# print(f"Mayúsculas: {mayusculas}")

# # Combinado: filtrar y transformar
# numeros = [1, -2, 3, -4, 5, -6]
# positivos_dobles = [x * 2 for x in numeros if x > 0]
# print(f"Positivos duplicados: {positivos_dobles}")

print()

# ============================================
# ¡FELICIDADES! Has completado el ejercicio 03
# ============================================
print("=" * 50)
print("✅ Ejercicio 03 completado")
print("=" * 50)
