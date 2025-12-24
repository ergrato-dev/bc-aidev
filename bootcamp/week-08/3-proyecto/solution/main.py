"""
Proyecto: Análisis de Ventas - SOLUCIÓN
=======================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================
# CONFIGURACIÓN
# ============================================

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
    
    df.loc[np.random.choice(df.index, 5), 'cantidad'] = np.nan
    df.loc[np.random.choice(df.index, 3), 'region'] = None
    
    duplicados = df.sample(10)
    df = pd.concat([df, duplicados], ignore_index=True)
    
    cols = ['fecha', 'producto', 'categoria', 'region', 'vendedor', 
            'cantidad', 'precio_unitario', 'descuento']
    df = df[cols]
    
    df.to_csv(DATA_FILE, index=False)
    print(f'Dataset generado: {DATA_FILE}')


generar_dataset()


# ============================================
# 1. CARGA Y EXPLORACIÓN
# ============================================
print('=' * 60)
print('1. CARGA Y EXPLORACIÓN')
print('=' * 60)

df = pd.read_csv(DATA_FILE)

print('\nPrimeras 5 filas:')
print(df.head())

print('\nInformación del DataFrame:')
df.info()

print(f'\nDimensiones: {df.shape[0]} filas x {df.shape[1]} columnas')

print('\nEstadísticas descriptivas:')
print(df.describe())

print('\nValores nulos:')
print(df.isna().sum())

print()


# ============================================
# 2. LIMPIEZA DE DATOS
# ============================================
print('=' * 60)
print('2. LIMPIEZA DE DATOS')
print('=' * 60)

# Eliminar duplicados
filas_antes = len(df)
df = df.drop_duplicates()
print(f'Filas eliminadas (duplicados): {filas_antes - len(df)}')
print(f'Filas después de eliminar duplicados: {len(df)}')

# Manejar nulos en cantidad
mediana_cantidad = df['cantidad'].median()
df['cantidad'] = df['cantidad'].fillna(mediana_cantidad)
print(f'Cantidad: nulos rellenados con mediana ({mediana_cantidad})')

# Manejar nulos en region
df['region'] = df['region'].fillna('Desconocida')
print('Region: nulos rellenados con "Desconocida"')

# Convertir fecha
df['fecha'] = pd.to_datetime(df['fecha'])
print(f'Fecha convertida a: {df["fecha"].dtype}')

# Crear columna total
df['total'] = df['cantidad'] * df['precio_unitario'] * (1 - df['descuento'])
print('Columna "total" creada')

print('\nValores nulos después de limpieza:')
print(df.isna().sum())

print('\nTipos de datos:')
print(df.dtypes)

print()


# ============================================
# 3. ANÁLISIS POR DIMENSIONES
# ============================================
print('=' * 60)
print('3. ANÁLISIS POR DIMENSIONES')
print('=' * 60)

# --- 3.1 Ventas por Producto ---
print('\n--- 3.1 Ventas por Producto ---')

ventas_producto = df.groupby('producto')['total'].sum().sort_values(ascending=False)
print('Total de ventas por producto:')
print(ventas_producto.round(2))

resumen_producto = df.groupby('producto').agg(
    total_ventas=('total', 'sum'),
    cantidad_total=('cantidad', 'sum'),
    num_transacciones=('total', 'count'),
    venta_promedio=('total', 'mean')
).round(2)
print('\nResumen por producto:')
print(resumen_producto)


# --- 3.2 Ventas por Región ---
print('\n--- 3.2 Ventas por Región ---')

ventas_region = df.groupby('region')['total'].sum().sort_values(ascending=False)
print('Total de ventas por región:')
print(ventas_region.round(2))

total_ventas = df['total'].sum()
pct_region = (ventas_region / total_ventas * 100).round(2)
print('\nPorcentaje por región:')
print(pct_region)


# --- 3.3 Rendimiento por Vendedor ---
print('\n--- 3.3 Rendimiento por Vendedor ---')

rendimiento = df.groupby('vendedor').agg(
    total_vendido=('total', 'sum'),
    num_ventas=('total', 'count'),
    venta_promedio=('total', 'mean')
).sort_values('total_vendido', ascending=False).round(2)
print(rendimiento)


# --- 3.4 Análisis Temporal ---
print('\n--- 3.4 Análisis Temporal ---')

df['mes'] = df['fecha'].dt.month
df['mes_nombre'] = df['fecha'].dt.month_name()

ventas_mes = df.groupby('mes')['total'].sum()
print('Ventas por mes:')
print(ventas_mes.round(2))

print()


# ============================================
# 4. REPORTES Y PIVOTS
# ============================================
print('=' * 60)
print('4. REPORTES Y PIVOTS')
print('=' * 60)

# --- 4.1 Pivot: Ventas por Región y Producto ---
print('\n--- 4.1 Pivot: Región x Producto ---')

pivot_region_producto = pd.pivot_table(
    df,
    values='total',
    index='region',
    columns='producto',
    aggfunc='sum',
    fill_value=0,
    margins=True,
    margins_name='Total'
).round(2)
print(pivot_region_producto)


# --- 4.2 Top Productos ---
print('\n--- 4.2 Top por Cantidad Vendida ---')

top_cantidad = df.groupby('producto')['cantidad'].sum().sort_values(ascending=False)
print(top_cantidad)


# --- 4.3 Mejor Vendedor ---
print('\n--- 4.3 Mejor Vendedor ---')

mejor_vendedor = df.groupby('vendedor')['total'].sum().idxmax()
ventas_mejor = df.groupby('vendedor')['total'].sum().max()
print(f'Mejor vendedor: {mejor_vendedor} con ${ventas_mejor:,.2f}')


# --- 4.4 Pivot: Vendedor x Mes ---
print('\n--- 4.4 Pivot: Vendedor x Mes ---')

pivot_vendedor_mes = pd.pivot_table(
    df,
    values='total',
    index='vendedor',
    columns='mes',
    aggfunc='sum',
    fill_value=0
).round(2)
print(pivot_vendedor_mes)

print()


# ============================================
# 5. INSIGHTS Y CONCLUSIONES
# ============================================
print('=' * 60)
print('5. INSIGHTS Y CONCLUSIONES')
print('=' * 60)

# Calcular insights
producto_top = ventas_producto.idxmax()
region_top = ventas_region.idxmax()
mes_top = ventas_mes.idxmax()

print(f'''
Insights del análisis:

1. Producto más vendido: {producto_top}
   - Total: ${ventas_producto[producto_top]:,.2f}
   
2. Región con más ventas: {region_top}
   - Total: ${ventas_region[region_top]:,.2f}
   - Porcentaje: {pct_region[region_top]}%

3. Mejor vendedor: {mejor_vendedor}
   - Total vendido: ${ventas_mejor:,.2f}

4. Mes con más ventas: Mes {mes_top}
   - Total: ${ventas_mes[mes_top]:,.2f}

5. Observaciones adicionales:
   - El producto A (Electrónica) tiene el precio más alto
   - Hay variación significativa entre vendedores
   - Las ventas están distribuidas entre las regiones
''')

print()
print('=' * 60)
print('PROYECTO COMPLETADO')
print('=' * 60)
