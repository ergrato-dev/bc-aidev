"""
Ejercicio 01: Creación y Exploración de DataFrames
==================================================
Aprende a crear y explorar DataFrames en Pandas.

Instrucciones:
- Lee cada sección y descomenta el código correspondiente
- Ejecuta el script después de cada paso para ver los resultados
- Experimenta modificando los valores
"""

# ============================================
# PASO 1: Importar Pandas
# ============================================
print('=== Paso 1: Importar Pandas ===')

# Descomenta las siguientes líneas:
# import pandas as pd
# import numpy as np
# print(f'Pandas version: {pd.__version__}')

print()

# ============================================
# PASO 2: Crear una Series
# ============================================
print('=== Paso 2: Crear una Series ===')

# Una Series es un array 1D con índice
# Descomenta las siguientes líneas:
# notas = pd.Series([85, 92, 78, 95, 88])
# print('Series desde lista:')
# print(notas)
# print()

# Series con índice personalizado
# notas_estudiantes = pd.Series(
#     [85, 92, 78, 95, 88],
#     index=['Ana', 'Bob', 'Carlos', 'Diana', 'Eva']
# )
# print('Series con índice personalizado:')
# print(notas_estudiantes)
# print()

# Atributos de la Series
# print(f'Valores: {notas_estudiantes.values}')
# print(f'Índice: {notas_estudiantes.index.tolist()}')
# print(f'Tipo de datos: {notas_estudiantes.dtype}')

print()

# ============================================
# PASO 3: Operaciones con Series
# ============================================
print('=== Paso 3: Operaciones con Series ===')

# Operaciones vectorizadas
# Descomenta las siguientes líneas:
# notas_ajustadas = notas_estudiantes + 5
# print('Notas ajustadas (+5):')
# print(notas_ajustadas)
# print()

# Estadísticas básicas
# print(f'Media: {notas_estudiantes.mean():.2f}')
# print(f'Mediana: {notas_estudiantes.median():.2f}')
# print(f'Desviación estándar: {notas_estudiantes.std():.2f}')
# print(f'Mínimo: {notas_estudiantes.min()}')
# print(f'Máximo: {notas_estudiantes.max()}')
# print()

# Acceso por índice
# print(f"Nota de Ana: {notas_estudiantes['Ana']}")
# print(f"Nota de Bob: {notas_estudiantes['Bob']}")

print()

# ============================================
# PASO 4: Crear un DataFrame desde Diccionario
# ============================================
print('=== Paso 4: Crear DataFrame desde Diccionario ===')

# Descomenta las siguientes líneas:
# datos = {
#     'nombre': ['Ana', 'Bob', 'Carlos', 'Diana', 'Eva'],
#     'edad': [25, 30, 28, 35, 22],
#     'ciudad': ['Madrid', 'Barcelona', 'Madrid', 'Valencia', 'Barcelona'],
#     'salario': [45000, 52000, 48000, 55000, 42000]
# }
# df = pd.DataFrame(datos)
# print('DataFrame creado:')
# print(df)

print()

# ============================================
# PASO 5: Atributos del DataFrame
# ============================================
print('=== Paso 5: Atributos del DataFrame ===')

# Descomenta las siguientes líneas:
# print(f'Forma (filas, columnas): {df.shape}')
# print(f'Número de filas: {len(df)}')
# print(f'Columnas: {df.columns.tolist()}')
# print(f'Índice: {df.index.tolist()}')
# print()
# print('Tipos de datos:')
# print(df.dtypes)

print()

# ============================================
# PASO 6: Leer CSV
# ============================================
print('=== Paso 6: Leer CSV ===')

# Primero creamos un CSV de ejemplo
# Descomenta las siguientes líneas:
# csv_content = '''producto,categoria,precio,stock
# Laptop,Electrónica,1200,15
# Mouse,Electrónica,25,150
# Teclado,Electrónica,75,80
# Silla,Oficina,200,30
# Escritorio,Oficina,350,12
# Monitor,Electrónica,400,25
# Lámpara,Oficina,45,60
# Auriculares,Electrónica,150,40'''

# Guardar CSV temporal
# with open('productos.csv', 'w', encoding='utf-8') as f:
#     f.write(csv_content)
# print('Archivo productos.csv creado')

# Leer el CSV
# df_productos = pd.read_csv('productos.csv')
# print('\nDataFrame desde CSV:')
# print(df_productos)

print()

# ============================================
# PASO 7: Exploración Básica
# ============================================
print('=== Paso 7: Exploración Básica ===')

# Descomenta las siguientes líneas:
# print('Primeras 3 filas:')
# print(df_productos.head(3))
# print()

# print('Últimas 2 filas:')
# print(df_productos.tail(2))
# print()

# print('Información del DataFrame:')
# df_productos.info()
# print()

# print('Estadísticas descriptivas:')
# print(df_productos.describe())

print()

# ============================================
# PASO 8: Acceso a Columnas
# ============================================
print('=== Paso 8: Acceso a Columnas ===')

# Descomenta las siguientes líneas:
# Una columna (retorna Series)
# print('Columna "producto":')
# print(df_productos['producto'])
# print()

# Múltiples columnas (retorna DataFrame)
# print('Columnas "producto" y "precio":')
# print(df_productos[['producto', 'precio']])
# print()

# Acceso como atributo (si no tiene espacios ni caracteres especiales)
# print('Usando df.categoria:')
# print(df_productos.categoria)

print()

# ============================================
# PASO 9: Value Counts
# ============================================
print('=== Paso 9: Value Counts ===')

# Descomenta las siguientes líneas:
# print('Conteo por categoría:')
# print(df_productos['categoria'].value_counts())
# print()

# print(f"Categorías únicas: {df_productos['categoria'].nunique()}")
# print(f"Lista de categorías: {df_productos['categoria'].unique().tolist()}")

print()

# ============================================
# LIMPIEZA
# ============================================
# Descomenta para eliminar el archivo CSV temporal:
# import os
# if os.path.exists('productos.csv'):
#     os.remove('productos.csv')
#     print('Archivo temporal eliminado')

print('=== Ejercicio completado ===')
