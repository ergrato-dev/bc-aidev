"""
Ejercicio 04: Agrupación y Combinación
======================================
Aprende a agrupar y combinar datos en Pandas.

Instrucciones:
- Lee cada sección y descomenta el código correspondiente
- Ejecuta el script después de cada paso para ver los resultados
"""

import pandas as pd
import numpy as np

# ============================================
# PASO 1: Dataset de Ventas
# ============================================
print('=== Paso 1: Dataset de Ventas ===')

# Descomenta las siguientes líneas:
# ventas = pd.DataFrame({
#     'fecha': pd.date_range('2024-01-01', periods=12, freq='D'),
#     'producto': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A'],
#     'region': ['Norte', 'Sur', 'Norte', 'Sur', 'Norte', 'Sur', 'Norte', 'Sur', 'Norte', 'Sur', 'Norte', 'Sur'],
#     'vendedor': ['Ana', 'Bob', 'Ana', 'Carlos', 'Bob', 'Ana', 'Carlos', 'Bob', 'Ana', 'Carlos', 'Bob', 'Ana'],
#     'cantidad': [10, 15, 8, 20, 12, 9, 18, 14, 11, 22, 16, 7],
#     'precio': [100, 80, 100, 50, 80, 100, 50, 80, 100, 50, 80, 100]
# })
# ventas['total'] = ventas['cantidad'] * ventas['precio']
# print('Dataset de ventas:')
# print(ventas)

print()

# ============================================
# PASO 2: Agrupación Básica con groupby
# ============================================
print('=== Paso 2: Agrupación Básica ===')

# Descomenta las siguientes líneas:
# # Suma por producto
# print('Total vendido por producto:')
# print(ventas.groupby('producto')['total'].sum())
# print()

# # Media por región
# print('Cantidad promedio por región:')
# print(ventas.groupby('region')['cantidad'].mean())
# print()

# # Conteo por vendedor
# print('Número de ventas por vendedor:')
# print(ventas.groupby('vendedor').size())

print()

# ============================================
# PASO 3: Múltiples Agregaciones con agg()
# ============================================
print('=== Paso 3: Múltiples Agregaciones ===')

# Descomenta las siguientes líneas:
# # Varias funciones a una columna
# print('Estadísticas de cantidad por producto:')
# print(ventas.groupby('producto')['cantidad'].agg(['sum', 'mean', 'min', 'max']))
# print()

# # Diferentes funciones por columna
# print('Agregaciones mixtas por región:')
# agg_result = ventas.groupby('region').agg({
#     'cantidad': ['sum', 'mean'],
#     'total': ['sum', 'max'],
#     'producto': 'nunique'  # Productos únicos
# })
# print(agg_result)

print()

# ============================================
# PASO 4: Agrupación por Múltiples Columnas
# ============================================
print('=== Paso 4: Agrupación por Múltiples Columnas ===')

# Descomenta las siguientes líneas:
# # Agrupar por región y producto
# print('Total por región y producto:')
# print(ventas.groupby(['region', 'producto'])['total'].sum())
# print()

# # Desapilar para ver como tabla
# print('Como tabla (unstack):')
# tabla = ventas.groupby(['region', 'producto'])['total'].sum().unstack(fill_value=0)
# print(tabla)

print()

# ============================================
# PASO 5: Agregaciones con Nombres
# ============================================
print('=== Paso 5: Named Aggregation ===')

# Descomenta las siguientes líneas:
# # Sintaxis más clara con nombres personalizados
# resumen = ventas.groupby('producto').agg(
#     ventas_totales=('total', 'sum'),
#     cantidad_total=('cantidad', 'sum'),
#     precio_promedio=('precio', 'mean'),
#     num_transacciones=('total', 'count')
# )
# print('Resumen por producto:')
# print(resumen)
# print()

# # Por vendedor
# resumen_vendedor = ventas.groupby('vendedor').agg(
#     total_vendido=('total', 'sum'),
#     venta_promedio=('total', 'mean'),
#     mejor_venta=('total', 'max')
# )
# print('Resumen por vendedor:')
# print(resumen_vendedor)

print()

# ============================================
# PASO 6: Transform para Mantener Tamaño
# ============================================
print('=== Paso 6: Transform ===')

# Descomenta las siguientes líneas:
# # Agregar media del grupo a cada fila
# ventas['media_producto'] = ventas.groupby('producto')['total'].transform('mean')
# print('Con media del producto:')
# print(ventas[['producto', 'total', 'media_producto']].head(6))
# print()

# # Calcular porcentaje respecto al grupo
# ventas['pct_region'] = ventas['total'] / ventas.groupby('region')['total'].transform('sum') * 100
# print('Porcentaje de ventas en su región:')
# print(ventas[['region', 'total', 'pct_region']].round(2).head(6))

print()

# ============================================
# PASO 7: Merge (Join) de DataFrames
# ============================================
print('=== Paso 7: Merge ===')

# Descomenta las siguientes líneas:
# # Crear DataFrame de productos con info adicional
# productos = pd.DataFrame({
#     'producto': ['A', 'B', 'C', 'D'],
#     'categoria': ['Electrónica', 'Hogar', 'Alimentos', 'Ropa'],
#     'proveedor': ['Prov1', 'Prov2', 'Prov1', 'Prov3']
# })
# print('Tabla de productos:')
# print(productos)
# print()

# # Inner merge (solo coincidencias)
# merged = pd.merge(ventas[['producto', 'cantidad', 'total']], productos, on='producto')
# print('Ventas con categoría (inner):')
# print(merged.head())
# print()

# # Left merge (todas las ventas)
# merged_left = pd.merge(ventas[['producto', 'total']], productos, on='producto', how='left')
# print(f'Filas en merge left: {len(merged_left)}')

print()

# ============================================
# PASO 8: Concat para Apilar DataFrames
# ============================================
print('=== Paso 8: Concat ===')

# Descomenta las siguientes líneas:
# # Crear dos DataFrames
# enero = pd.DataFrame({
#     'producto': ['A', 'B'],
#     'ventas': [100, 150],
#     'mes': 'Enero'
# })
# febrero = pd.DataFrame({
#     'producto': ['A', 'B'],
#     'ventas': [120, 140],
#     'mes': 'Febrero'
# })
# print('Enero:')
# print(enero)
# print()
# print('Febrero:')
# print(febrero)
# print()

# # Concatenar verticalmente
# combinado = pd.concat([enero, febrero], ignore_index=True)
# print('Combinado (vertical):')
# print(combinado)
# print()

# # Concatenar horizontalmente
# h_concat = pd.concat([enero[['producto', 'ventas']], febrero[['ventas']]], axis=1)
# h_concat.columns = ['producto', 'ventas_ene', 'ventas_feb']
# print('Combinado (horizontal):')
# print(h_concat)

print()

# ============================================
# PASO 9: Pivot Table
# ============================================
print('=== Paso 9: Pivot Table ===')

# Descomenta las siguientes líneas:
# # Pivot simple
# pivot = pd.pivot_table(
#     ventas,
#     values='total',
#     index='region',
#     columns='producto',
#     aggfunc='sum',
#     fill_value=0
# )
# print('Pivot: Total por región y producto:')
# print(pivot)
# print()

# # Pivot con márgenes (totales)
# pivot_margins = pd.pivot_table(
#     ventas,
#     values='total',
#     index='vendedor',
#     columns='region',
#     aggfunc='sum',
#     fill_value=0,
#     margins=True,
#     margins_name='Total'
# )
# print('Pivot con totales:')
# print(pivot_margins)

print()

print('=== Ejercicio completado ===')
