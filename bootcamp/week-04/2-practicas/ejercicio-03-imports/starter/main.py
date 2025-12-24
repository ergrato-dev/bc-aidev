# ============================================
# EJERCICIO 03: Imports Absolutos y Relativos
# ============================================
# Aprenderás las diferentes formas de importar
# y cómo organizar imports correctamente.
# ============================================

print('=== Ejercicio 03: Imports ===')
print()

# ============================================
# PASO 1: Crear Estructura de Paquete
# ============================================
print('--- Paso 1: Crear Estructura ---')
print('Crea la siguiente estructura:')
print()
print('starter/')
print('├── main.py  (este archivo)')
print('└── myapp/')
print('    ├── __init__.py')
print('    ├── config.py')
print('    ├── core.py')
print('    ├── utils.py')
print('    └── services/')
print('        ├── __init__.py')
print('        └── processor.py')
print()

# ============================================
# PASO 2: Crear myapp/config.py
# ============================================
print('--- Paso 2: Crear config.py ---')

# Copia a myapp/config.py:
# --- INICIO: myapp/config.py ---
# """Configuración de la aplicación."""
#
# APP_NAME = "MyApp"
# VERSION = "1.0.0"
# DEBUG = True
# MAX_ITEMS = 100
# --- FIN ---

print('Crea myapp/config.py con constantes')
print()

# ============================================
# PASO 3: Crear myapp/utils.py
# ============================================
print('--- Paso 3: Crear utils.py ---')

# Copia a myapp/utils.py:
# --- INICIO: myapp/utils.py ---
# """Utilidades generales."""
#
# from .config import DEBUG  # Import relativo
#
#
# def log(message: str) -> None:
#     """Imprime mensaje si DEBUG está activo."""
#     if DEBUG:
#         print(f"[LOG] {message}")
#
#
# def format_output(data: dict) -> str:
#     """Formatea un diccionario para mostrar."""
#     lines = [f"  {k}: {v}" for k, v in data.items()]
#     return "\n".join(lines)
# --- FIN ---

print('Crea myapp/utils.py con import relativo de config')
print()

# ============================================
# PASO 4: Crear myapp/core.py
# ============================================
print('--- Paso 4: Crear core.py ---')

# Copia a myapp/core.py:
# --- INICIO: myapp/core.py ---
# """Funcionalidad principal de la aplicación."""
#
# # Imports relativos (dentro del mismo paquete)
# from .config import APP_NAME, VERSION, MAX_ITEMS
# from .utils import log, format_output
#
#
# class App:
#     """Clase principal de la aplicación."""
#
#     def __init__(self):
#         self.name = APP_NAME
#         self.version = VERSION
#         self.items: list = []
#         log(f"Inicializando {self.name} v{self.version}")
#
#     def add_item(self, item: str) -> bool:
#         """Agrega un item a la lista."""
#         if len(self.items) >= MAX_ITEMS:
#             log(f"Límite alcanzado: {MAX_ITEMS}")
#             return False
#         self.items.append(item)
#         log(f"Item agregado: {item}")
#         return True
#
#     def get_info(self) -> dict:
#         """Retorna información de la app."""
#         return {
#             "name": self.name,
#             "version": self.version,
#             "item_count": len(self.items),
#         }
#
#     def __str__(self) -> str:
#         return format_output(self.get_info())
# --- FIN ---

print('Crea myapp/core.py con la clase App')
print()

# ============================================
# PASO 5: Crear myapp/services/processor.py
# ============================================
print('--- Paso 5: Crear services/processor.py ---')

# Copia a myapp/services/processor.py:
# --- INICIO: myapp/services/processor.py ---
# """Servicio de procesamiento."""
#
# # Import relativo al paquete padre (..)
# from ..core import App
# from ..utils import log
# from ..config import DEBUG
#
#
# class Processor:
#     """Procesa items usando la App."""
#
#     def __init__(self, app: App):
#         self.app = app
#         log("Processor inicializado")
#
#     def process_batch(self, items: list[str]) -> int:
#         """Procesa un lote de items."""
#         added = 0
#         for item in items:
#             if self.app.add_item(item):
#                 added += 1
#         log(f"Procesados {added} de {len(items)} items")
#         return added
#
#
# if __name__ == "__main__":
#     # Solo se ejecuta con: python -m myapp.services.processor
#     print("=== Processor Demo ===")
#     app = App()
#     proc = Processor(app)
#     proc.process_batch(["item1", "item2", "item3"])
#     print(app)
# --- FIN ---

print('Crea myapp/services/processor.py')
print()

# ============================================
# PASO 6: Crear los __init__.py
# ============================================
print('--- Paso 6: Crear __init__.py ---')

# Copia a myapp/__init__.py:
# --- INICIO: myapp/__init__.py ---
# """
# MyApp - Aplicación de ejemplo para imports.
# """
#
# from .config import APP_NAME, VERSION
# from .core import App
# from .utils import log
#
# __all__ = ["App", "log", "APP_NAME", "VERSION"]
# --- FIN ---

# Copia a myapp/services/__init__.py:
# --- INICIO: myapp/services/__init__.py ---
# """Servicios de MyApp."""
#
# from .processor import Processor
#
# __all__ = ["Processor"]
# --- FIN ---

print('Crea los archivos __init__.py')
print()

# ============================================
# PASO 7: Probar Imports Absolutos
# ============================================
print('--- Paso 7: Imports Absolutos ---')

# Descomenta para probar:
# # Import absoluto desde el paquete
# from myapp import App, APP_NAME, VERSION
#
# print(f"App: {APP_NAME} v{VERSION}")
# app = App()
# app.add_item("test")
# print(app)
#
# # Import absoluto desde submódulos
# from myapp.core import App
# from myapp.services.processor import Processor

print('Prueba los imports absolutos')
print()

# ============================================
# PASO 8: Organizar Imports (PEP 8)
# ============================================
print('--- Paso 8: Organización PEP 8 ---')

# Los imports deben estar organizados así:
# --- EJEMPLO ---
# # 1. Standard library
# import os
# import sys
# from pathlib import Path
#
# # 2. Third party (paquetes pip)
# import requests
# import pandas as pd
#
# # 3. Local (tu código)
# from myapp import App
# from myapp.services import Processor
# --- FIN ---

print('Organiza imports: stdlib > third-party > local')
print()

# ============================================
# PASO 9: Ejecutar como Módulo
# ============================================
print('--- Paso 9: Ejecutar como Módulo ---')
print()
print('Para ejecutar un módulo dentro de un paquete:')
print('  python -m myapp.services.processor')
print()
print('NO funciona:')
print('  python myapp/services/processor.py  # Error!')
print()
print('La diferencia es que -m establece el contexto del paquete')
print()

# ============================================
# PASO 10: Prueba Final
# ============================================
print('--- Paso 10: Prueba Final ---')

# Descomenta para la prueba final:
# from myapp import App
# from myapp.services import Processor
#
# # Crear app y processor
# app = App()
# proc = Processor(app)
#
# # Procesar items
# items = ["Python", "JavaScript", "Rust", "Go"]
# proc.process_batch(items)
#
# print()
# print("Estado final:")
# print(app)

print()

# ============================================
# RESUMEN
# ============================================
print('=== Resumen ===')
print()
print('IMPORTS ABSOLUTOS (desde raíz):')
print('  from myapp.core import App')
print('  from myapp.services.processor import Processor')
print()
print('IMPORTS RELATIVOS (dentro del paquete):')
print('  from .config import DEBUG        # mismo nivel')
print('  from ..core import App           # nivel padre')
print('  from ...other import func        # dos niveles arriba')
print()
print('ORGANIZACIÓN PEP 8:')
print('  1. stdlib (import os)')
print('  2. third-party (import pandas)')
print('  3. local (from myapp import App)')
print()
print('EJECUTAR MÓDULOS:')
print('  python -m myapp.module  # Correcto')
print('  python myapp/module.py  # Imports relativos fallan')
