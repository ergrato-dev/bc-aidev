"""
Ejercicio 01: Operaciones con Archivos
======================================
Sigue las instrucciones en README.md
Descomenta cada paso y ejecuta el script.
"""

from pathlib import Path

print("=" * 50)
print("EJERCICIO 01: Operaciones con Archivos")
print("=" * 50)


# ============================================
# PASO 1: Crear y Escribir Archivos
# ============================================
print('\n--- Paso 1: Crear y Escribir Archivos ---')

# Descomenta las siguientes líneas:
# output_path = Path('output/mensaje.txt')
# output_path.parent.mkdir(exist_ok=True)
#
# with open(output_path, 'w', encoding='utf-8') as f:
#     f.write('¡Hola Mundo!\n')
#     f.write('Segunda línea\n')
#
# print(f"Archivo creado: {output_path}")


# ============================================
# PASO 2: Leer Archivos
# ============================================
print('\n--- Paso 2: Leer Archivos ---')

# Descomenta las siguientes líneas:
# output_path = Path('output/mensaje.txt')
#
# # Método 1: read() - todo el contenido
# print("Con read():")
# with open(output_path, 'r', encoding='utf-8') as f:
#     content = f.read()
#     print(repr(content))  # repr muestra \n
#
# # Método 2: readlines() - lista de líneas
# print("\nCon readlines():")
# with open(output_path, 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     print(lines)


# ============================================
# PASO 3: Iterar Líneas (Archivos Grandes)
# ============================================
print('\n--- Paso 3: Iterar Líneas ---')

# Descomenta las siguientes líneas:
# output_path = Path('output/mensaje.txt')
#
# print("Iterando línea por línea:")
# with open(output_path, 'r', encoding='utf-8') as f:
#     for i, line in enumerate(f, 1):
#         print(f"  Línea {i}: {line.strip()}")


# ============================================
# PASO 4: Append (Añadir al Final)
# ============================================
print('\n--- Paso 4: Append ---')

# Descomenta las siguientes líneas:
# output_path = Path('output/mensaje.txt')
#
# with open(output_path, 'a', encoding='utf-8') as f:
#     f.write('Tercera línea (append)\n')
#     f.write('Cuarta línea (append)\n')
#
# print("Líneas añadidas. Contenido actual:")
# with open(output_path, 'r', encoding='utf-8') as f:
#     print(f.read())


# ============================================
# PASO 5: Pathlib - Lectura/Escritura Directa
# ============================================
print('\n--- Paso 5: Pathlib Directo ---')

# Descomenta las siguientes líneas:
# quick_path = Path('output/quick.txt')
#
# # Escribir con pathlib
# quick_path.write_text('Línea 1 con pathlib\nLínea 2\n', encoding='utf-8')
# print(f"Archivo creado: {quick_path}")
#
# # Leer con pathlib
# content = quick_path.read_text(encoding='utf-8')
# print(f"Contenido:\n{content}")


# ============================================
# PASO 6: Información de Archivos
# ============================================
print('\n--- Paso 6: Información de Archivos ---')

# Descomenta las siguientes líneas:
# path = Path('output/mensaje.txt')
#
# if path.exists():
#     print(f"Nombre:     {path.name}")
#     print(f"Stem:       {path.stem}")
#     print(f"Extensión:  {path.suffix}")
#     print(f"Padre:      {path.parent}")
#     print(f"Absoluta:   {path.absolute()}")
#     print(f"Es archivo: {path.is_file()}")
#     print(f"Es dir:     {path.is_dir()}")
#     print(f"Tamaño:     {path.stat().st_size} bytes")
# else:
#     print("Ejecuta Paso 1 primero")


# ============================================
# PASO 7: Listar Archivos
# ============================================
print('\n--- Paso 7: Listar Archivos ---')

# Descomenta las siguientes líneas:
# output_dir = Path('output')
#
# if output_dir.exists():
#     print("Todos los elementos en output/:")
#     for item in output_dir.iterdir():
#         tipo = "📁" if item.is_dir() else "📄"
#         print(f"  {tipo} {item.name}")
#
#     print("\nSolo archivos .txt:")
#     for txt_file in output_dir.glob('*.txt'):
#         size = txt_file.stat().st_size
#         print(f"  📄 {txt_file.name} ({size} bytes)")
# else:
#     print("Ejecuta los pasos anteriores primero")


# ============================================
# FIN DEL EJERCICIO
# ============================================
print('\n' + '=' * 50)
print('Ejercicio completado!')
print('=' * 50)
