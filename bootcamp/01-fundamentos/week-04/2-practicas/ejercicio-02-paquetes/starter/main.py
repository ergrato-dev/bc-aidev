# ============================================
# EJERCICIO 02: Estructurar Paquetes
# ============================================
# Aprenderás a crear paquetes Python con
# __init__.py, módulos y subpaquetes.
# ============================================

print('=== Ejercicio 02: Paquetes ===')
print()

# ============================================
# PASO 1: Crear Estructura de Carpetas
# ============================================
print('--- Paso 1: Crear Estructura ---')
print('''
Crea la siguiente estructura de carpetas:

starter/
├── main.py  (este archivo)
└── data_tools/
    ├── __init__.py
    ├── readers.py
    └── writers.py

En Linux/Mac:
  mkdir -p data_tools
  touch data_tools/__init__.py
  touch data_tools/readers.py
  touch data_tools/writers.py
''')
print()

# ============================================
# PASO 2: Crear readers.py
# ============================================
print('--- Paso 2: Crear readers.py ---')

# Copia este contenido a data_tools/readers.py:
# --- INICIO: data_tools/readers.py ---
# """Módulo para lectura de datos."""
#
# import json
# from pathlib import Path
#
#
# def read_json(filepath: str) -> dict:
#     """Lee un archivo JSON y retorna un diccionario."""
#     with open(filepath, 'r', encoding='utf-8') as f:
#         return json.load(f)
#
#
# def read_text(filepath: str) -> str:
#     """Lee un archivo de texto completo."""
#     return Path(filepath).read_text(encoding='utf-8')
#
#
# def read_lines(filepath: str) -> list[str]:
#     """Lee un archivo y retorna lista de líneas."""
#     with open(filepath, 'r', encoding='utf-8') as f:
#         return [line.strip() for line in f]
# --- FIN ---

print('Crea data_tools/readers.py con las funciones de lectura')
print()

# ============================================
# PASO 3: Crear writers.py
# ============================================
print('--- Paso 3: Crear writers.py ---')

# Copia este contenido a data_tools/writers.py:
# --- INICIO: data_tools/writers.py ---
# """Módulo para escritura de datos."""
#
# import json
# from pathlib import Path
#
#
# def write_json(data: dict, filepath: str, indent: int = 2) -> None:
#     """Escribe un diccionario a un archivo JSON."""
#     with open(filepath, 'w', encoding='utf-8') as f:
#         json.dump(data, f, indent=indent, ensure_ascii=False)
#
#
# def write_text(content: str, filepath: str) -> None:
#     """Escribe texto a un archivo."""
#     Path(filepath).write_text(content, encoding='utf-8')
#
#
# def write_lines(lines: list[str], filepath: str) -> None:
#     """Escribe lista de líneas a un archivo."""
#     with open(filepath, 'w', encoding='utf-8') as f:
#         f.writelines(line + '\n' for line in lines)
# --- FIN ---

print('Crea data_tools/writers.py con las funciones de escritura')
print()

# ============================================
# PASO 4: Crear __init__.py (vacío primero)
# ============================================
print('--- Paso 4: __init__.py vacío ---')

# Crea data_tools/__init__.py vacío:
# --- INICIO: data_tools/__init__.py ---
# """Paquete data_tools - vacío por ahora."""
# --- FIN ---

# Prueba importar desde los módulos:
# from data_tools.readers import read_json
# from data_tools.writers import write_json
#
# print("Import desde módulos funciona!")
# print(f"read_json: {read_json}")
# print(f"write_json: {write_json}")

print('Crea __init__.py vacío y prueba importar desde módulos')
print()

# ============================================
# PASO 5: Configurar __init__.py con Exports
# ============================================
print('--- Paso 5: Configurar __init__.py ---')

# Actualiza data_tools/__init__.py:
# --- INICIO: data_tools/__init__.py ---
# """
# Data Tools - Paquete para manejo de datos.
#
# Proporciona utilidades para:
# - Leer archivos (JSON, texto)
# - Escribir archivos
#
# Example:
#     >>> from data_tools import read_json, write_json
#     >>> data = read_json("config.json")
# """
#
# __version__ = "1.0.0"
# __author__ = "AI Bootcamp"
#
# # Exponer funciones principales
# from .readers import read_json, read_text, read_lines
# from .writers import write_json, write_text, write_lines
#
# # Controlar "from data_tools import *"
# __all__ = [
#     "read_json",
#     "read_text",
#     "read_lines",
#     "write_json",
#     "write_text",
#     "write_lines",
# ]
# --- FIN ---

# Ahora puedes importar directamente del paquete:
# from data_tools import read_json, write_json
#
# # O todo el paquete
# import data_tools
# print(f"Versión: {data_tools.__version__}")

print('Actualiza __init__.py para exponer la API pública')
print()

# ============================================
# PASO 6: Crear Subpaquete transformers/
# ============================================
print('--- Paso 6: Crear Subpaquete ---')

# Crea la estructura:
# data_tools/
# └── transformers/
#     ├── __init__.py
#     └── text.py

# --- INICIO: data_tools/transformers/text.py ---
# """Transformadores de texto."""
#
# import re
#
#
# def clean_text(text: str) -> str:
#     """Elimina caracteres especiales."""
#     return re.sub(r'[^\w\s]', '', text)
#
#
# def normalize(text: str) -> str:
#     """Normaliza a minúsculas sin espacios extra."""
#     return ' '.join(text.lower().split())
#
#
# def word_count(text: str) -> int:
#     """Cuenta palabras en el texto."""
#     return len(text.split())
# --- FIN ---

# --- INICIO: data_tools/transformers/__init__.py ---
# """Subpaquete de transformadores."""
#
# from .text import clean_text, normalize, word_count
#
# __all__ = ["clean_text", "normalize", "word_count"]
# --- FIN ---

print('Crea el subpaquete transformers/ con text.py')
print()

# ============================================
# PASO 7: Probar Todo el Paquete
# ============================================
print('--- Paso 7: Probar el Paquete ---')

# Descomenta después de crear toda la estructura:
# # Import desde el paquete principal
# from data_tools import read_json, write_json, __version__
#
# print(f"data_tools v{__version__}")
#
# # Import desde subpaquete
# from data_tools.transformers import clean_text, normalize
#
# # Probar transformadores
# texto = "  ¡Hola,   Mundo!  "
# print(f"Original: '{texto}'")
# print(f"Limpio: '{clean_text(texto)}'")
# print(f"Normalizado: '{normalize(texto)}'")
#
# # Probar lectura/escritura
# data = {"nombre": "Test", "valor": 42}
# write_json(data, "test_output.json")
# loaded = read_json("test_output.json")
# print(f"Datos guardados y cargados: {loaded}")
#
# # Limpiar
# import os
# os.remove("test_output.json")
# print("Archivo de prueba eliminado")

print()

# ============================================
# PASO 8: Explorar el Paquete
# ============================================
print('--- Paso 8: Explorar el Paquete ---')

# Descomenta para explorar:
# import data_tools
#
# print(f"Archivo __init__.py: {data_tools.__file__}")
# print(f"Documentación:")
# print(data_tools.__doc__)
#
# print(f"\nContenido exportado (__all__):")
# print(data_tools.__all__)
#
# print(f"\nTodo el contenido:")
# for item in dir(data_tools):
#     if not item.startswith('_'):
#         print(f"  - {item}")

print()

# ============================================
# RESUMEN
# ============================================
print('=== Resumen ===')
print('''
Estructura de un paquete:
  package/
  ├── __init__.py    # Hace que sea un paquete
  ├── module_a.py    # Módulos
  └── subpackage/    # Subpaquetes
      ├── __init__.py
      └── module_b.py

__init__.py:
  - Puede estar vacío
  - Expone API pública con imports
  - Define __all__ para "import *"
  - Contiene metadatos (__version__, __author__)

Imports:
  from package import func           # Desde __init__.py
  from package.module import func    # Desde módulo
  from package.subpkg import func    # Desde subpaquete
''')
