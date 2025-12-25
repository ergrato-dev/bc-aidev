"""
Ejercicio 02: Selección y Filtrado de Datos
===========================================
Aprende a seleccionar y filtrar datos en Pandas.

Instrucciones:
- Lee cada sección y descomenta el código correspondiente
- Ejecuta el script después de cada paso para ver los resultados
- Experimenta modificando los valores
"""

import pandas as pd
import numpy as np

# ============================================
# PASO 1: Preparar el Dataset
# ============================================
print('=== Paso 1: Preparar el Dataset ===')

# Descomenta las siguientes líneas:
# df = pd.DataFrame({
#     'nombre': ['Ana', 'Bob', 'Carlos', 'Diana', 'Eva', 'Frank'],
#     'departamento': ['IT', 'Ventas', 'IT', 'RRHH', 'Ventas', 'IT'],
#     'salario': [55000, 48000, 62000, 45000, 51000, 58000],
#     'años_exp': [5, 3, 8, 2, 4, 6],
#     'activo': [True, True, True, False, True, True]
# })
# print('Dataset de empleados:')
# print(df)

print()

# ============================================
# PASO 2: Selección con loc (etiquetas)
# ============================================
print('=== Paso 2: Selección con loc ===')

# loc selecciona por etiquetas (índice y nombres de columna)
# Descomenta las siguientes líneas:

# Una fila completa
# print('Fila con índice 0:')
# print(df.loc[0])
# print()

# Rango de filas (incluye ambos extremos)
# print('Filas 0 a 2 (inclusive):')
# print(df.loc[0:2])
# print()

# Valor específico
# print(f"Nombre en fila 0: {df.loc[0, 'nombre']}")
# print()

# Filas y columnas específicas
# print('Filas 0-2, columnas nombre y salario:')
# print(df.loc[0:2, ['nombre', 'salario']])
# print()

# Todas las filas, columnas específicas
# print('Todas las filas, solo nombre y departamento:')
# print(df.loc[:, ['nombre', 'departamento']])

print()

# ============================================
# PASO 3: Selección con iloc (posición)
# ============================================
print('=== Paso 3: Selección con iloc ===')

# iloc selecciona por posición numérica (como arrays)
# Descomenta las siguientes líneas:

# Primera fila
# print('Primera fila (iloc[0]):')
# print(df.iloc[0])
# print()

# Rango de filas (NO incluye el final, como slicing Python)
# print('Filas 0 y 1 (iloc[0:2]):')
# print(df.iloc[0:2])
# print()

# Valor específico por posición
# print(f'Valor en fila 0, columna 1: {df.iloc[0, 1]}')
# print()

# Subconjunto por posiciones
# print('Primeras 3 filas, primeras 2 columnas:')
# print(df.iloc[0:3, 0:2])
# print()

# Filas específicas no contiguas
# print('Filas 0, 2 y 4:')
# print(df.iloc[[0, 2, 4]])

print()

# ============================================
# PASO 4: loc vs iloc con Índice Personalizado
# ============================================
print('=== Paso 4: loc vs iloc con Índice Personalizado ===')

# Descomenta las siguientes líneas:
# df_indexed = df.set_index('nombre')
# print('DataFrame con índice personalizado:')
# print(df_indexed)
# print()

# loc usa las etiquetas del índice
# print("loc['Ana'] - busca por etiqueta:")
# print(df_indexed.loc['Ana'])
# print()

# iloc sigue usando posición numérica
# print('iloc[0] - busca por posición:')
# print(df_indexed.iloc[0])
# print()

# Rango con loc (usa etiquetas)
# print("loc['Ana':'Carlos'] - incluye Carlos:")
# print(df_indexed.loc['Ana':'Carlos'])
# print()

# Rango con iloc (excluye final)
# print('iloc[0:3] - no incluye posición 3:')
# print(df_indexed.iloc[0:3])

print()

# ============================================
# PASO 5: Filtros Booleanos
# ============================================
print('=== Paso 5: Filtros Booleanos ===')

# Descomenta las siguientes líneas:
# Crear máscara booleana
# mask = df['salario'] > 50000
# print('Máscara booleana (salario > 50000):')
# print(mask)
# print()

# Aplicar máscara
# print('Empleados con salario > 50000:')
# print(df[mask])
# print()

# Forma directa (más común)
# print('Empleados del departamento IT:')
# print(df[df['departamento'] == 'IT'])
# print()

# Comparación con texto
# print('Empleados activos:')
# print(df[df['activo'] == True])

print()

# ============================================
# PASO 6: Condiciones Múltiples
# ============================================
print('=== Paso 6: Condiciones Múltiples ===')

# IMPORTANTE: Usar paréntesis y operadores &, |, ~
# Descomenta las siguientes líneas:

# AND: ambas condiciones deben cumplirse
# print('IT con salario > 55000 (AND):')
# print(df[(df['departamento'] == 'IT') & (df['salario'] > 55000)])
# print()

# OR: al menos una condición debe cumplirse
# print('IT o Ventas (OR):')
# print(df[(df['departamento'] == 'IT') | (df['departamento'] == 'Ventas')])
# print()

# NOT: negación
# print('No son de RRHH (NOT):')
# print(df[~(df['departamento'] == 'RRHH')])
# print()

# Combinación compleja
# print('IT activos con más de 5 años de experiencia:')
# filtro = (df['departamento'] == 'IT') & (df['activo'] == True) & (df['años_exp'] > 5)
# print(df[filtro])

print()

# ============================================
# PASO 7: Método isin()
# ============================================
print('=== Paso 7: Método isin() ===')

# isin() es ideal para múltiples valores OR
# Descomenta las siguientes líneas:
# departamentos_interes = ['IT', 'Ventas']
# print(f'Empleados en {departamentos_interes}:')
# print(df[df['departamento'].isin(departamentos_interes)])
# print()

# Negación con ~
# print('Empleados NO en IT ni Ventas:')
# print(df[~df['departamento'].isin(departamentos_interes)])
# print()

# Con valores numéricos
# print('Empleados con 3, 5 o 8 años de experiencia:')
# print(df[df['años_exp'].isin([3, 5, 8])])

print()

# ============================================
# PASO 8: Método query()
# ============================================
print('=== Paso 8: Método query() ===')

# query() tiene sintaxis más legible, similar a SQL
# Descomenta las siguientes líneas:

# Condición simple
# print('query: salario > 50000')
# print(df.query('salario > 50000'))
# print()

# Múltiples condiciones (and, or, not)
# print('query: salario > 50000 and departamento == "IT"')
# print(df.query('salario > 50000 and departamento == "IT"'))
# print()

# Usar variables con @
# min_salario = 50000
# dept = 'IT'
# print(f'query con variables: salario > {min_salario} and departamento == "{dept}"')
# print(df.query('salario > @min_salario and departamento == @dept'))
# print()

# Nombres de columnas con espacios se pueden usar con backticks
# (nuestras columnas no tienen espacios, pero es útil saberlo)
# print('query: años_exp >= 5')
# print(df.query('años_exp >= 5'))

print()

# ============================================
# PASO 9: between() y str.contains()
# ============================================
print('=== Paso 9: between() y str.contains() ===')

# Descomenta las siguientes líneas:

# between() para rangos numéricos (incluye ambos extremos)
# print('Salario entre 48000 y 55000:')
# print(df[df['salario'].between(48000, 55000)])
# print()

# str.contains() para búsqueda en texto
# print('Nombres que contienen "a" (case insensitive):')
# print(df[df['nombre'].str.contains('a', case=False)])
# print()

# str.startswith() y str.endswith()
# print('Nombres que empiezan con "A" o "E":')
# print(df[df['nombre'].str.startswith(('A', 'E'))])
# print()

# Combinar todo
# print('Empleados IT con experiencia entre 4 y 8 años:')
# filtro = (df['departamento'] == 'IT') & df['años_exp'].between(4, 8)
# print(df[filtro])

print()

print('=== Ejercicio completado ===')
