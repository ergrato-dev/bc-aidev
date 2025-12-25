"""
Ejercicio 03: Limpieza de Datos
===============================
Aprende a limpiar y transformar datos en Pandas.

Instrucciones:
- Lee cada sección y descomenta el código correspondiente
- Ejecuta el script después de cada paso para ver los resultados
"""

import pandas as pd
import numpy as np

# ============================================
# PASO 1: Dataset con Datos Sucios
# ============================================
print('=== Paso 1: Dataset con Datos Sucios ===')

# Descomenta las siguientes líneas:
# df = pd.DataFrame({
#     'nombre': ['  Ana  ', 'BOB', None, 'diana', 'Ana', 'Carlos'],
#     'edad': ['25', '30', '35', 'treinta', '25', '28'],
#     'ciudad': ['Madrid', 'barcelona', 'VALENCIA', None, 'Madrid', 'Sevilla'],
#     'salario': [50000, 60000, np.nan, 52000, 50000, 48000],
#     'fecha_ingreso': ['2020-01-15', '2019/06/20', '2021-03-10', '2020-11-05', '2020-01-15', '2022-02-28']
# })
# print('Dataset original (con problemas):')
# print(df)
# print()
# print('Tipos de datos:')
# print(df.dtypes)

print()

# ============================================
# PASO 2: Detectar Missing Values
# ============================================
print('=== Paso 2: Detectar Missing Values ===')

# Descomenta las siguientes líneas:
# print('Matriz de valores nulos:')
# print(df.isna())
# print()

# print('Conteo de NaN por columna:')
# print(df.isna().sum())
# print()

# print('Porcentaje de NaN por columna:')
# print((df.isna().mean() * 100).round(2))
# print()

# print('Filas con algún NaN:')
# print(df[df.isna().any(axis=1)])

print()

# ============================================
# PASO 3: Eliminar Missing Values (dropna)
# ============================================
print('=== Paso 3: Eliminar Missing Values ===')

# Descomenta las siguientes líneas:
# print('Original:')
# print(df)
# print()

# # Eliminar filas con cualquier NaN
# df_sin_nan = df.dropna()
# print('Después de dropna():')
# print(df_sin_nan)
# print()

# # Eliminar solo si NaN en columnas específicas
# df_parcial = df.dropna(subset=['salario'])
# print('dropna solo en salario:')
# print(df_parcial)

print()

# ============================================
# PASO 4: Rellenar Missing Values (fillna)
# ============================================
print('=== Paso 4: Rellenar Missing Values ===')

# Descomenta las siguientes líneas:
# df_filled = df.copy()

# # Rellenar nombre con string
# df_filled['nombre'] = df_filled['nombre'].fillna('Desconocido')
# print('Nombre rellenado:')
# print(df_filled['nombre'])
# print()

# # Rellenar ciudad con valor específico
# df_filled['ciudad'] = df_filled['ciudad'].fillna('No especificada')
# print('Ciudad rellenada:')
# print(df_filled['ciudad'])
# print()

# # Rellenar salario con la media
# media_salario = df_filled['salario'].mean()
# df_filled['salario'] = df_filled['salario'].fillna(media_salario)
# print(f'Salario rellenado con media ({media_salario:.0f}):')
# print(df_filled['salario'])

print()

# ============================================
# PASO 5: Detectar Duplicados
# ============================================
print('=== Paso 5: Detectar Duplicados ===')

# Descomenta las siguientes líneas:
# print('¿Filas duplicadas?')
# print(df.duplicated())
# print()

# print('Ver filas duplicadas:')
# print(df[df.duplicated()])
# print()

# print('Duplicados por nombre (incluyendo primero):')
# print(df[df.duplicated(subset=['nombre'], keep=False)])
# print()

# print(f'Total duplicados: {df.duplicated().sum()}')

print()

# ============================================
# PASO 6: Eliminar Duplicados
# ============================================
print('=== Paso 6: Eliminar Duplicados ===')

# Descomenta las siguientes líneas:
# print('Original:')
# print(df[['nombre', 'salario']])
# print()

# # Eliminar duplicados exactos
# df_unique = df.drop_duplicates()
# print('Sin duplicados exactos:')
# print(df_unique[['nombre', 'salario']])
# print()

# # Eliminar por columna específica (mantiene primero)
# df_unique_nombre = df.drop_duplicates(subset=['nombre'])
# print('Sin duplicados por nombre:')
# print(df_unique_nombre[['nombre', 'salario']])

print()

# ============================================
# PASO 7: Conversión de Tipos
# ============================================
print('=== Paso 7: Conversión de Tipos ===')

# Descomenta las siguientes líneas:
# df_tipos = df.copy()

# # Convertir edad a numérico (errores a NaN)
# df_tipos['edad'] = pd.to_numeric(df_tipos['edad'], errors='coerce')
# print('Edad convertida a numérico:')
# print(df_tipos['edad'])
# print(f'Tipo: {df_tipos["edad"].dtype}')
# print()

# # Convertir fecha
# df_tipos['fecha_ingreso'] = pd.to_datetime(df_tipos['fecha_ingreso'], errors='coerce')
# print('Fecha convertida:')
# print(df_tipos['fecha_ingreso'])
# print(f'Tipo: {df_tipos["fecha_ingreso"].dtype}')
# print()

# # Extraer componentes de fecha
# df_tipos['año'] = df_tipos['fecha_ingreso'].dt.year
# df_tipos['mes'] = df_tipos['fecha_ingreso'].dt.month
# print('Año y mes extraídos:')
# print(df_tipos[['fecha_ingreso', 'año', 'mes']])

print()

# ============================================
# PASO 8: Limpieza de Strings
# ============================================
print('=== Paso 8: Limpieza de Strings ===')

# Descomenta las siguientes líneas:
# df_str = df.copy()
# df_str['nombre'] = df_str['nombre'].fillna('Desconocido')

# # Quitar espacios
# df_str['nombre'] = df_str['nombre'].str.strip()
# print('Sin espacios extra:')
# print(df_str['nombre'])
# print()

# # Capitalizar (Title Case)
# df_str['nombre'] = df_str['nombre'].str.title()
# print('Capitalizado:')
# print(df_str['nombre'])
# print()

# # Estandarizar ciudad
# df_str['ciudad'] = df_str['ciudad'].fillna('No especificada')
# df_str['ciudad'] = df_str['ciudad'].str.title()
# print('Ciudad estandarizada:')
# print(df_str['ciudad'])

print()

# ============================================
# PASO 9: Transformaciones con apply()
# ============================================
print('=== Paso 9: Transformaciones con apply() ===')

# Descomenta las siguientes líneas:
# df_apply = df.copy()
# df_apply['salario'] = df_apply['salario'].fillna(0)

# # Lambda simple
# df_apply['salario_anual'] = df_apply['salario'].apply(lambda x: x * 12)
# print('Salario anual (lambda):')
# print(df_apply[['salario', 'salario_anual']])
# print()

# # Función personalizada
# def categorizar_salario(salario):
#     if salario >= 55000:
#         return 'Alto'
#     elif salario >= 45000:
#         return 'Medio'
#     else:
#         return 'Bajo'

# df_apply['categoria'] = df_apply['salario'].apply(categorizar_salario)
# print('Categoría de salario:')
# print(df_apply[['salario', 'categoria']])
# print()

# # Apply por fila (axis=1)
# def resumen(row):
#     nombre = str(row['nombre']).strip() if pd.notna(row['nombre']) else 'N/A'
#     salario = row['salario']
#     return f'{nombre}: ${salario:,.0f}'

# df_apply['resumen'] = df_apply.apply(resumen, axis=1)
# print('Resumen por fila:')
# print(df_apply['resumen'])

print()

print('=== Ejercicio completado ===')
