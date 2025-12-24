"""
Ejercicio 02: Formatos de Datos (CSV y JSON)
============================================
Sigue las instrucciones en README.md
Descomenta cada paso y ejecuta el script.
"""

import csv
import json
from pathlib import Path

print("=" * 50)
print("EJERCICIO 02: Formatos de Datos")
print("=" * 50)


# ============================================
# PASO 1: Escribir CSV con DictWriter
# ============================================
print('\n--- Paso 1: Escribir CSV ---')

# Descomenta las siguientes líneas:
# users = [
#     {'name': 'Alice', 'age': 30, 'email': 'alice@email.com'},
#     {'name': 'Bob', 'age': 25, 'email': 'bob@email.com'},
#     {'name': 'Carlos', 'age': 35, 'email': 'carlos@email.com'},
# ]
#
# output_path = Path('output/users.csv')
# output_path.parent.mkdir(exist_ok=True)
#
# with open(output_path, 'w', encoding='utf-8', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=['name', 'age', 'email'])
#     writer.writeheader()
#     writer.writerows(users)
#
# print(f"CSV creado: {output_path}")
# print(f"Contenido:\n{output_path.read_text(encoding='utf-8')}")


# ============================================
# PASO 2: Leer CSV con DictReader
# ============================================
print('\n--- Paso 2: Leer CSV ---')

# Descomenta las siguientes líneas:
# csv_path = Path('output/users.csv')
#
# if csv_path.exists():
#     print("Leyendo usuarios:")
#     with open(csv_path, 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             print(f"  - {row['name']} ({row['age']} años): {row['email']}")
# else:
#     print("Ejecuta Paso 1 primero")


# ============================================
# PASO 3: Escribir JSON
# ============================================
print('\n--- Paso 3: Escribir JSON ---')

# Descomenta las siguientes líneas:
# config = {
#     'app': 'Mi Aplicación',
#     'version': '1.0.0',
#     'settings': {
#         'debug': True,
#         'max_users': 100,
#         'theme': 'dark'
#     },
#     'features': ['auth', 'api', 'logging'],
#     'description': 'Aplicación con caracteres especiales: áéíóú ñ'
# }
#
# json_path = Path('output/config.json')
#
# with open(json_path, 'w', encoding='utf-8') as f:
#     json.dump(config, f, indent=2, ensure_ascii=False)
#
# print(f"JSON creado: {json_path}")
# print(f"Contenido:\n{json_path.read_text(encoding='utf-8')}")


# ============================================
# PASO 4: Leer JSON
# ============================================
print('\n--- Paso 4: Leer JSON ---')

# Descomenta las siguientes líneas:
# json_path = Path('output/config.json')
#
# if json_path.exists():
#     with open(json_path, 'r', encoding='utf-8') as f:
#         config = json.load(f)
#
#     print(f"App: {config['app']}")
#     print(f"Version: {config['version']}")
#     print(f"Debug: {config['settings']['debug']}")
#     print(f"Features: {', '.join(config['features'])}")
# else:
#     print("Ejecuta Paso 3 primero")


# ============================================
# PASO 5: Convertir CSV a JSON
# ============================================
print('\n--- Paso 5: CSV a JSON ---')

# Descomenta las siguientes líneas:
# csv_path = Path('output/users.csv')
# json_path = Path('output/users.json')
#
# if csv_path.exists():
#     # Leer CSV
#     with open(csv_path, 'r', encoding='utf-8') as f:
#         users = list(csv.DictReader(f))
#
#     # Convertir age a int
#     for user in users:
#         user['age'] = int(user['age'])
#
#     # Escribir JSON
#     with open(json_path, 'w', encoding='utf-8') as f:
#         json.dump(users, f, indent=2, ensure_ascii=False)
#
#     print(f"Convertido a: {json_path}")
#     print(f"Contenido:\n{json_path.read_text(encoding='utf-8')}")
# else:
#     print("Ejecuta Paso 1 primero")


# ============================================
# PASO 6: JSON Strings
# ============================================
print('\n--- Paso 6: JSON Strings ---')

# Descomenta las siguientes líneas:
# # Dict a string JSON (dumps)
# data = {'name': 'Test', 'value': 42, 'active': True}
# json_string = json.dumps(data, indent=2)
# print(f"Dict → JSON string:\n{json_string}")
#
# # String JSON a dict (loads)
# json_input = '{"status": "ok", "count": 10, "items": ["a", "b", "c"]}'
# parsed = json.loads(json_input)
# print(f"\nJSON string → Dict:")
# print(f"  Status: {parsed['status']}")
# print(f"  Count: {parsed['count']}")
# print(f"  Items: {parsed['items']}")


# ============================================
# PASO 7: Manejo de Errores
# ============================================
print('\n--- Paso 7: Manejo de Errores ---')

# Descomenta las siguientes líneas:
# # JSON inválido
# invalid_json = '{"name": "Test", invalid_key}'
#
# print("Intentando parsear JSON inválido...")
# try:
#     data = json.loads(invalid_json)
#     print(f"Datos: {data}")
# except json.JSONDecodeError as e:
#     print(f"❌ Error de JSON: {e.msg}")
#     print(f"   Línea: {e.lineno}, Columna: {e.colno}")
#
# # CSV con campos faltantes
# print("\nCSV con datos incompletos:")
# csv_data = "name,age,email\nAlice,30,alice@test.com\nBob,,bob@test.com"
# import io
# reader = csv.DictReader(io.StringIO(csv_data))
# for row in reader:
#     age = row['age'] if row['age'] else 'N/A'
#     print(f"  {row['name']}: edad={age}")


# ============================================
# FIN DEL EJERCICIO
# ============================================
print('\n' + '=' * 50)
print('Ejercicio completado!')
print('=' * 50)
