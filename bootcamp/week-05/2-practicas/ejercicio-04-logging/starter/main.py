"""
Ejercicio 04: Logging
=====================
Sigue las instrucciones en README.md
Descomenta cada paso y ejecuta el script.

NOTA: Algunos pasos reconfiguran logging.
Puedes comentar pasos anteriores para ver mejor los resultados.
"""

print("=" * 50)
print("EJERCICIO 04: Logging")
print("=" * 50)


# ============================================
# PASO 1: Logging Básico vs print
# ============================================
print('\n--- Paso 1: Logging Básico ---')

# Descomenta las siguientes líneas:
# import logging
#
# # Configuración mínima
# logging.basicConfig(level=logging.DEBUG)
#
# print("Usando print(): Hola mundo")
#
# print("\nUsando logging (diferentes niveles):")
# logging.debug("DEBUG: Información detallada para debugging")
# logging.info("INFO: Confirmación de que las cosas funcionan")
# logging.warning("WARNING: Algo inesperado, pero no crítico")
# logging.error("ERROR: Problema serio, algo falló")
# logging.critical("CRITICAL: Error grave, el programa puede fallar")


# ============================================
# PASO 2: Formato Personalizado
# ============================================
print('\n--- Paso 2: Formato Personalizado ---')

# Descomenta las siguientes líneas:
# import logging
#
# # Reconfigurar con formato
# # force=True permite reconfigurar
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s | %(levelname)-8s | %(message)s',
#     datefmt='%H:%M:%S',
#     force=True
# )
#
# logging.info("Mensaje con timestamp")
# logging.warning("Advertencia formateada")
# logging.error("Error con formato personalizado")


# ============================================
# PASO 3: Logger por Módulo
# ============================================
print('\n--- Paso 3: Logger por Módulo ---')

# Descomenta las siguientes líneas:
# import logging
#
# # Crear logger con nombre
# logger = logging.getLogger('mi_modulo')
# logger.setLevel(logging.DEBUG)
#
# # Evitar duplicados si ya hay handlers
# if not logger.handlers:
#     # Handler de consola
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.DEBUG)
#
#     # Formatter con nombre del logger
#     formatter = logging.Formatter(
#         '%(name)s | %(levelname)-8s | %(message)s'
#     )
#     console_handler.setFormatter(formatter)
#
#     logger.addHandler(console_handler)
#
# # Usar el logger
# logger.debug("Debug desde mi_modulo")
# logger.info("Info desde mi_modulo")
# logger.warning("Warning desde mi_modulo")


# ============================================
# PASO 4: Logging a Archivo
# ============================================
print('\n--- Paso 4: Logging a Archivo ---')

# Descomenta las siguientes líneas:
# import logging
# from pathlib import Path
#
# # Crear directorio de logs
# log_dir = Path('logs')
# log_dir.mkdir(exist_ok=True)
#
# # Logger para archivo
# file_logger = logging.getLogger('file_app')
# file_logger.setLevel(logging.DEBUG)
#
# if not file_logger.handlers:
#     # Handler de archivo
#     file_handler = logging.FileHandler(
#         log_dir / 'app.log',
#         encoding='utf-8',
#         mode='w'  # 'w' sobrescribe, 'a' añade
#     )
#     file_handler.setLevel(logging.DEBUG)
#
#     formatter = logging.Formatter(
#         '%(asctime)s | %(levelname)-8s | %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
#     file_handler.setFormatter(formatter)
#
#     file_logger.addHandler(file_handler)
#
# # Escribir logs
# file_logger.info("Aplicación iniciada")
# file_logger.debug("Cargando configuración")
# file_logger.warning("Configuración por defecto usada")
# file_logger.info("Aplicación lista")
#
# print(f"Logs guardados en: {log_dir / 'app.log'}")
# print(f"Contenido:\n{(log_dir / 'app.log').read_text()}")


# ============================================
# PASO 5: Múltiples Handlers
# ============================================
print('\n--- Paso 5: Múltiples Handlers ---')

# Descomenta las siguientes líneas:
# import logging
# from pathlib import Path
#
# log_dir = Path('logs')
# log_dir.mkdir(exist_ok=True)
#
# # Logger con múltiples handlers
# multi_logger = logging.getLogger('multi_app')
# multi_logger.setLevel(logging.DEBUG)
#
# if not multi_logger.handlers:
#     formatter = logging.Formatter(
#         '%(asctime)s | %(levelname)-8s | %(message)s',
#         datefmt='%H:%M:%S'
#     )
#
#     # Handler consola: solo INFO y superior
#     console = logging.StreamHandler()
#     console.setLevel(logging.INFO)
#     console.setFormatter(formatter)
#     multi_logger.addHandler(console)
#
#     # Handler archivo: todo (DEBUG+)
#     file_handler = logging.FileHandler(
#         log_dir / 'debug.log',
#         encoding='utf-8',
#         mode='w'
#     )
#     file_handler.setLevel(logging.DEBUG)
#     file_handler.setFormatter(formatter)
#     multi_logger.addHandler(file_handler)
#
# print("Generando logs (DEBUG solo va al archivo):")
# multi_logger.debug("Este DEBUG solo aparece en archivo")
# multi_logger.info("INFO aparece en consola y archivo")
# multi_logger.warning("WARNING en ambos")
# multi_logger.error("ERROR en ambos")
#
# print(f"\nContenido de debug.log:")
# print((log_dir / 'debug.log').read_text())


# ============================================
# PASO 6: Logging de Excepciones
# ============================================
print('\n--- Paso 6: Logging de Excepciones ---')

# Descomenta las siguientes líneas:
# import logging
#
# exc_logger = logging.getLogger('exceptions')
# exc_logger.setLevel(logging.DEBUG)
#
# if not exc_logger.handlers:
#     handler = logging.StreamHandler()
#     handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
#     exc_logger.addHandler(handler)
#
#
# def risky_division(a, b):
#     """División con logging de errores."""
#     exc_logger.debug(f"Dividiendo {a} / {b}")
#     try:
#         result = a / b
#         exc_logger.info(f"Resultado: {result}")
#         return result
#     except ZeroDivisionError:
#         # exception() incluye el traceback automáticamente
#         exc_logger.exception("Error de división por cero")
#         return None
#     except TypeError:
#         exc_logger.exception("Error de tipos")
#         return None
#
#
# print("División exitosa:")
# risky_division(10, 2)
#
# print("\nDivisión por cero:")
# risky_division(10, 0)
#
# print("\nTipos incorrectos:")
# risky_division("10", 2)


# ============================================
# PASO 7: Configuración Completa
# ============================================
print('\n--- Paso 7: Configuración Completa ---')

# Descomenta las siguientes líneas:
# import logging
# from logging.handlers import RotatingFileHandler
# from pathlib import Path
#
#
# def setup_logging(name='myapp', log_dir='logs', console_level=logging.INFO):
#     """Configura logging profesional."""
#     log_path = Path(log_dir)
#     log_path.mkdir(exist_ok=True)
#
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.DEBUG)
#
#     # Limpiar handlers existentes
#     logger.handlers.clear()
#
#     # Formatter
#     formatter = logging.Formatter(
#         '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
#
#     # Handler consola
#     console = logging.StreamHandler()
#     console.setLevel(console_level)
#     console.setFormatter(formatter)
#     logger.addHandler(console)
#
#     # Handler archivo con rotación
#     file_handler = RotatingFileHandler(
#         log_path / f'{name}.log',
#         maxBytes=1024 * 1024,  # 1MB
#         backupCount=3,
#         encoding='utf-8'
#     )
#     file_handler.setLevel(logging.DEBUG)
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
#
#     # Handler solo para errores
#     error_handler = logging.FileHandler(
#         log_path / f'{name}_errors.log',
#         encoding='utf-8',
#         mode='w'
#     )
#     error_handler.setLevel(logging.ERROR)
#     error_handler.setFormatter(formatter)
#     logger.addHandler(error_handler)
#
#     return logger
#
#
# # Usar la configuración
# app_logger = setup_logging('produccion', console_level=logging.INFO)
#
# app_logger.debug("Debug: iniciando sistema")
# app_logger.info("Sistema iniciado correctamente")
# app_logger.warning("Memoria al 80%")
# app_logger.error("No se pudo conectar a servicio externo")
# app_logger.critical("Base de datos no responde")
#
# print("\nArchivos de log creados:")
# for log_file in Path('logs').glob('produccion*.log'):
#     print(f"  📄 {log_file.name}")
#     print(f"     {log_file.read_text()[:200]}...")


# ============================================
# FIN DEL EJERCICIO
# ============================================
print('\n' + '=' * 50)
print('Ejercicio completado!')
print('=' * 50)
