"""
Ejercicio 03: Excepciones
=========================
Sigue las instrucciones en README.md
Descomenta cada paso y ejecuta el script.
"""

print("=" * 50)
print("EJERCICIO 03: Excepciones")
print("=" * 50)


# ============================================
# PASO 1: try/except Básico
# ============================================
print('\n--- Paso 1: try/except Básico ---')

# Descomenta las siguientes líneas:
# # Sin manejo de errores (¡crashearía!)
# # value = int("no es número")
#
# # Con manejo de errores
# print("Intentando convertir 'hello' a int...")
# try:
#     value = int("hello")
#     print(f"Valor: {value}")
# except ValueError:
#     print("❌ No es un número válido")
#     value = 0
#
# print(f"Valor final: {value}")
#
# # Otro ejemplo
# print("\nIntentando convertir '42' a int...")
# try:
#     value = int("42")
#     print(f"✅ Valor: {value}")
# except ValueError:
#     print("❌ No es un número válido")


# ============================================
# PASO 2: Acceder al Objeto Excepción
# ============================================
print('\n--- Paso 2: Objeto Excepción ---')

# Descomenta las siguientes líneas:
# print("Intentando dividir por cero...")
# try:
#     result = 10 / 0
# except ZeroDivisionError as e:
#     print(f"Error capturado:")
#     print(f"  Mensaje: {e}")
#     print(f"  Tipo: {type(e).__name__}")
#     print(f"  Args: {e.args}")
#
# print("\nIntentando acceder a índice inválido...")
# try:
#     lista = [1, 2, 3]
#     valor = lista[10]
# except IndexError as e:
#     print(f"Error capturado:")
#     print(f"  Mensaje: {e}")
#     print(f"  Tipo: {type(e).__name__}")


# ============================================
# PASO 3: Múltiples Excepciones
# ============================================
print('\n--- Paso 3: Múltiples Excepciones ---')

# Descomenta las siguientes líneas:
# def divide(a, b):
#     """Divide a entre b con manejo de errores."""
#     try:
#         return a / b
#     except ZeroDivisionError:
#         print("  ❌ No se puede dividir por cero")
#         return None
#     except TypeError:
#         print("  ❌ Los valores deben ser números")
#         return None
#
# print("divide(10, 2):", divide(10, 2))
# print("divide(10, 0):", divide(10, 0))
# print("divide(10, 'a'):", divide(10, 'a'))
#
# # Capturar múltiples en un bloque
# print("\nCapturando múltiples excepciones:")
# try:
#     # Cambia esta línea para probar diferentes errores
#     result = int("abc")
# except (ValueError, TypeError) as e:
#     print(f"  Error de tipo o valor: {e}")


# ============================================
# PASO 4: else y finally
# ============================================
print('\n--- Paso 4: else y finally ---')

# Descomenta las siguientes líneas:
# from pathlib import Path
#
# def read_file_safe(filepath):
#     """Lee archivo con manejo completo de errores."""
#     print(f"Intentando leer: {filepath}")
#     try:
#         with open(filepath, 'r', encoding='utf-8') as f:
#             content = f.read()
#     except FileNotFoundError:
#         print("  ❌ Archivo no encontrado")
#         content = None
#     except PermissionError:
#         print("  ❌ Sin permisos de lectura")
#         content = None
#     else:
#         # Solo se ejecuta si NO hubo excepciones
#         print(f"  ✅ Leídos {len(content)} caracteres")
#     finally:
#         # SIEMPRE se ejecuta
#         print("  📋 Operación completada")
#     return content
#
# # Probar con archivo que no existe
# result = read_file_safe('no_existe.txt')
# print(f"Resultado: {result}")
#
# # Crear un archivo temporal y leerlo
# print()
# test_file = Path('output/test_read.txt')
# test_file.parent.mkdir(exist_ok=True)
# test_file.write_text('Contenido de prueba', encoding='utf-8')
# result = read_file_safe(test_file)
# print(f"Resultado: {result}")


# ============================================
# PASO 5: Lanzar Excepciones (raise)
# ============================================
print('\n--- Paso 5: raise ---')

# Descomenta las siguientes líneas:
# def validate_age(age):
#     """Valida que la edad sea válida."""
#     if not isinstance(age, int):
#         raise TypeError("La edad debe ser un entero")
#     if age < 0:
#         raise ValueError("La edad no puede ser negativa")
#     if age > 150:
#         raise ValueError("Edad inválida (>150)")
#     return True
#
# # Probar validaciones
# test_ages = [25, -5, 200, "treinta", 0]
#
# for age in test_ages:
#     try:
#         validate_age(age)
#         print(f"  ✅ Edad {age} es válida")
#     except (ValueError, TypeError) as e:
#         print(f"  ❌ Edad {age!r}: {e}")


# ============================================
# PASO 6: Excepciones Personalizadas
# ============================================
print('\n--- Paso 6: Excepciones Personalizadas ---')

# Descomenta las siguientes líneas:
# class ValidationError(Exception):
#     """Error base de validación."""
#     pass
#
#
# class EmailError(ValidationError):
#     """Error de validación de email."""
#     def __init__(self, email, message="Email inválido"):
#         self.email = email
#         self.message = message
#         super().__init__(f"{message}: {email}")
#
#
# class PasswordError(ValidationError):
#     """Error de validación de contraseña."""
#     def __init__(self, reason):
#         self.reason = reason
#         super().__init__(f"Contraseña inválida: {reason}")
#
#
# def validate_email(email):
#     if not email:
#         raise EmailError(email, "Email vacío")
#     if '@' not in email:
#         raise EmailError(email, "Falta @")
#     if '.' not in email.split('@')[1]:
#         raise EmailError(email, "Dominio inválido")
#     return True
#
#
# def validate_password(password):
#     if len(password) < 8:
#         raise PasswordError("Mínimo 8 caracteres")
#     if not any(c.isupper() for c in password):
#         raise PasswordError("Requiere mayúscula")
#     if not any(c.isdigit() for c in password):
#         raise PasswordError("Requiere número")
#     return True
#
#
# # Probar validaciones
# test_data = [
#     ('user@example.com', 'Password123'),
#     ('invalid-email', 'Password123'),
#     ('user@example.com', 'weak'),
#     ('', 'Password123'),
# ]
#
# for email, password in test_data:
#     print(f"\nValidando: {email!r}, {password!r}")
#     try:
#         validate_email(email)
#         validate_password(password)
#         print("  ✅ Datos válidos")
#     except EmailError as e:
#         print(f"  ❌ Email: {e}")
#     except PasswordError as e:
#         print(f"  ❌ Password: {e}")


# ============================================
# PASO 7: Re-lanzar Excepciones
# ============================================
print('\n--- Paso 7: Re-lanzar Excepciones ---')

# Descomenta las siguientes líneas:
# import logging
#
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# logger = logging.getLogger(__name__)
#
#
# def process_data(data):
#     """Procesa datos con logging y re-raise."""
#     logger.info(f"Procesando: {data!r}")
#     try:
#         result = int(data) * 2
#         logger.info(f"Resultado: {result}")
#         return result
#     except ValueError:
#         logger.error(f"Dato inválido: {data!r}")
#         raise  # Re-lanza la misma excepción
#
#
# def main():
#     """Función principal que maneja errores de process_data."""
#     datos = ['10', '20', 'invalid', '30']
#
#     for dato in datos:
#         try:
#             result = process_data(dato)
#             print(f"  Procesado: {dato} → {result}")
#         except ValueError:
#             print(f"  ⚠️ Saltando dato inválido: {dato}")
#             continue
#
#
# main()


# ============================================
# FIN DEL EJERCICIO
# ============================================
print('\n' + '=' * 50)
print('Ejercicio completado!')
print('=' * 50)
