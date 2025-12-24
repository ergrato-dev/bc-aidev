"""
Proyecto: Análisis de Ventas
============================
Analiza un dataset de ventas aplicando técnicas de Pandas.

Instrucciones:
- Completa cada sección marcada con TODO
- Ejecuta el script para verificar tu progreso
- Consulta la teoría si necesitas ayuda
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================
# CONFIGURACIÓN
# ============================================

# Crear directorio de datos si no existe
DATA_DIR = Path(__file__).parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)
DATA_FILE = DATA_DIR / 'ventas.csv'


def generar_dataset():
    """Genera el dataset de ventas si no existe."""
    if DATA_FILE.exists():
        return
    
    np.random.seed(42)
    n = 200
    
    fechas = pd.date_range('2024-01-01', periods=90, freq='D')
    productos = ['A', 'B', 'C', 'D']
    categorias = {'A': 'Electrónica', 'B': 'Hogar', 'C': 'Alimentos', 'D': 'Ropa'}
    regiones = ['Norte', 'Sur', 'Este', 'Oeste']
    vendedores = ['Ana', 'Bob', 'Carlos', 'Diana', 'Eva']
    precios = {'A': 150, 'B': 80, 'C': 25, 'D': 60}
    
    data = {
        'fecha': np.random.choice(fechas, n),
        'producto': np.random.choice(productos, n),
        'region': np.random.choice(regiones, n),
        'vendedor': np.random.choice(vendedores, n),
        'cantidad': np.random.randint(1, 20, n),
        'descuento': np.random.choice([0, 0.05, 0.10, 0.15, 0.20], n)
    }
    
    df = pd.DataFrame(data)
    df['categoria'] = df['producto'].map(categorias)
    df['precio_unitario'] = df['producto'].map(precios)
    
    # Introducir algunos problemas para limpiar
    # Missing values
    df.loc[np.random.choice(df.index, 5), 'cantidad'] = np.nan
    df.loc[np.random.choice(df.index, 3), 'region'] = None
    
    # Duplicados
    duplicados = df.sample(10)
    df = pd.concat([df, duplicados], ignore_index=True)
    
    # Reordenar columnas
    cols = ['fecha', 'producto', 'categoria', 'region', 'vendedor', 
            'cantidad', 'precio_unitario', 'descuento']
    df = df[cols]
    
    df.to_csv(DATA_FILE, index=False)
    print(f'Dataset generado: {DATA_FILE}')


# Generar dataset
generar_dataset()


# ============================================
# 1. CARGA Y EXPLORACIÓN
# ============================================
print('=' * 60)
print('1. CARGA Y EXPLORACIÓN')
print('=' * 60)

# TODO: Cargar el dataset desde DATA_FILE
# df = ...

# TODO: Mostrar las primeras 5 filas
# print(...)

# TODO: Mostrar información del DataFrame (info())
# ...

# TODO: Mostrar dimensiones (shape)
# print(f'Dimensiones: ...')

# TODO: Mostrar estadísticas descriptivas
# print(...)

# TODO: Contar valores nulos por columna
# print('Valores nulos:')
# print(...)

print()


# ============================================
# 2. LIMPIEZA DE DATOS
# ============================================
print('=' * 60)
print('2. LIMPIEZA DE DATOS')
print('=' * 60)

# TODO: Eliminar filas duplicadas
# df = df.drop_duplicates()
# print(f'Filas después de eliminar duplicados: ...')

# TODO: Manejar valores nulos en 'cantidad' (rellenar con mediana)
# df['cantidad'] = ...

# TODO: Manejar valores nulos en 'region' (rellenar con 'Desconocida')
# df['region'] = ...

# TODO: Convertir 'fecha' a datetime si no lo es
# df['fecha'] = ...

# TODO: Crear columna 'total' = cantidad * precio_unitario * (1 - descuento)
# df['total'] = ...

# TODO: Verificar que no hay más nulos
# print('Valores nulos después de limpieza:')
# print(...)

# TODO: Mostrar tipos de datos finales
# print('Tipos de datos:')
# print(...)

print()


# ============================================
# 3. ANÁLISIS POR DIMENSIONES
# ============================================
print('=' * 60)
print('3. ANÁLISIS POR DIMENSIONES')
print('=' * 60)

# --- 3.1 Ventas por Producto ---
print('\n--- 3.1 Ventas por Producto ---')

# TODO: Calcular total de ventas por producto
# ventas_producto = df.groupby('producto')...
# print(ventas_producto)

# TODO: Calcular cantidad total y número de transacciones por producto
# resumen_producto = df.groupby('producto').agg(...)
# print(resumen_producto)


# --- 3.2 Ventas por Región ---
print('\n--- 3.2 Ventas por Región ---')

# TODO: Calcular total de ventas por región
# ventas_region = ...
# print(ventas_region)

# TODO: Calcular porcentaje de ventas por región
# total_ventas = df['total'].sum()
# pct_region = (ventas_region / total_ventas * 100).round(2)
# print('Porcentaje por región:')
# print(pct_region)


# --- 3.3 Rendimiento por Vendedor ---
print('\n--- 3.3 Rendimiento por Vendedor ---')

# TODO: Calcular métricas por vendedor
# rendimiento = df.groupby('vendedor').agg(
#     total_vendido=('total', 'sum'),
#     num_ventas=('total', 'count'),
#     venta_promedio=('total', 'mean')
# ).round(2)
# print(rendimiento)


# --- 3.4 Análisis Temporal ---
print('\n--- 3.4 Análisis Temporal ---')

# TODO: Extraer mes de la fecha
# df['mes'] = df['fecha'].dt.month

# TODO: Calcular ventas por mes
# ventas_mes = df.groupby('mes')['total'].sum()
# print('Ventas por mes:')
# print(ventas_mes)

print()


# ============================================
# 4. REPORTES Y PIVOTS
# ============================================
print('=' * 60)
print('4. REPORTES Y PIVOTS')
print('=' * 60)

# --- 4.1 Pivot: Ventas por Región y Producto ---
print('\n--- 4.1 Pivot: Región x Producto ---')

# TODO: Crear pivot table de ventas por región y producto
# pivot_region_producto = pd.pivot_table(
#     df,
#     values='total',
#     index='region',
#     columns='producto',
#     aggfunc='sum',
#     fill_value=0,
#     margins=True
# ).round(2)
# print(pivot_region_producto)


# --- 4.2 Top 5 Productos más Vendidos ---
print('\n--- 4.2 Top 5 por Cantidad Vendida ---')

# TODO: Encontrar los 5 productos con más unidades vendidas
# top_5 = df.groupby('producto')['cantidad'].sum().nlargest(5)
# print(top_5)


# --- 4.3 Mejor Vendedor ---
print('\n--- 4.3 Mejor Vendedor ---')

# TODO: Encontrar el vendedor con más ventas totales
# mejor_vendedor = df.groupby('vendedor')['total'].sum().idxmax()
# ventas_mejor = df.groupby('vendedor')['total'].sum().max()
# print(f'Mejor vendedor: {mejor_vendedor} con ${ventas_mejor:,.2f}')


# --- 4.4 Pivot: Vendedor x Mes ---
print('\n--- 4.4 Pivot: Vendedor x Mes ---')

# TODO: Crear pivot de ventas por vendedor y mes
# pivot_vendedor_mes = pd.pivot_table(
#     df,
#     values='total',
#     index='vendedor',
#     columns='mes',
#     aggfunc='sum',
#     fill_value=0
# ).round(2)
# print(pivot_vendedor_mes)

print()


# ============================================
# 5. INSIGHTS Y CONCLUSIONES
# ============================================
print('=' * 60)
print('5. INSIGHTS Y CONCLUSIONES')
print('=' * 60)

# TODO: Basándote en tu análisis, completa las conclusiones

print('''
Insights del análisis:

1. Producto más vendido: [TODO: completar]
   
2. Región con más ventas: [TODO: completar]

3. Mejor vendedor: [TODO: completar]

4. Mes con más ventas: [TODO: completar]

5. Observaciones adicionales:
   - [TODO: agregar observaciones]
''')

print()
print('=' * 60)
print('PROYECTO COMPLETADO')
print('=' * 60)
