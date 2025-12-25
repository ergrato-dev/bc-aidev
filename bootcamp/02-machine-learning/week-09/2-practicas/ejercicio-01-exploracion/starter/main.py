"""
Ejercicio 01: Exploración de Datos para ML
==========================================

Objetivo: Explorar el dataset Iris para prepararlo para ML.

Instrucciones:
1. Lee cada sección
2. Descomenta el código indicado
3. Ejecuta y observa los resultados
"""

# ============================================
# PASO 1: Cargar el Dataset
# ============================================
print('--- Paso 1: Cargar el Dataset ---')

# Scikit-learn incluye datasets de ejemplo
# Descomenta las siguientes líneas:

# from sklearn.datasets import load_iris
# 
# iris = load_iris()
# print(f'Tipo de objeto: {type(iris)}')
# print(f'Claves disponibles: {iris.keys()}')

print()

# ============================================
# PASO 2: Explorar la Estructura del Dataset
# ============================================
print('--- Paso 2: Explorar la Estructura ---')

# El dataset tiene varios atributos importantes
# Descomenta las siguientes líneas:

# print(f'Shape de los datos: {iris.data.shape}')
# print(f'Shape del target: {iris.target.shape}')
# print(f'\nNombres de features: {iris.feature_names}')
# print(f'Nombres de clases: {iris.target_names}')
# print(f'\nPrimeras 5 filas de datos:')
# print(iris.data[:5])
# print(f'\nPrimeros 5 targets: {iris.target[:5]}')

print()

# ============================================
# PASO 3: Convertir a DataFrame de Pandas
# ============================================
print('--- Paso 3: Convertir a DataFrame ---')

# Para mejor manipulación, usamos pandas
# Descomenta las siguientes líneas:

# import pandas as pd
# 
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# df['species'] = iris.target
# 
# # Mapear números a nombres de especies
# species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
# df['species_name'] = df['species'].map(species_map)
# 
# print('DataFrame creado:')
# print(df.head(10))
# print(f'\nShape: {df.shape}')
# print(f'Columnas: {df.columns.tolist()}')

print()

# ============================================
# PASO 4: Análisis Estadístico Descriptivo
# ============================================
print('--- Paso 4: Análisis Estadístico ---')

# describe() muestra estadísticas de cada columna numérica
# Descomenta las siguientes líneas:

# print('Estadísticas descriptivas:')
# print(df.describe())
# 
# print('\nInformación del DataFrame:')
# print(df.info())
# 
# print('\nValores nulos por columna:')
# print(df.isnull().sum())

print()

# ============================================
# PASO 5: Distribución del Target (Clases)
# ============================================
print('--- Paso 5: Distribución del Target ---')

# Es importante verificar si las clases están balanceadas
# Descomenta las siguientes líneas:

# print('Distribución de clases:')
# print(df['species_name'].value_counts())
# 
# print('\nDistribución porcentual:')
# print(df['species_name'].value_counts(normalize=True).round(3) * 100)
# 
# # Verificar balance
# counts = df['species'].value_counts()
# if counts.max() / counts.min() < 1.5:
#     print('\n✅ Las clases están balanceadas')
# else:
#     print('\n⚠️  Las clases están desbalanceadas')

print()

# ============================================
# PASO 6: Correlaciones entre Features
# ============================================
print('--- Paso 6: Correlaciones ---')

# La correlación ayuda a entender relaciones entre variables
# Descomenta las siguientes líneas:

# # Seleccionar solo columnas numéricas de features
# numeric_cols = iris.feature_names
# correlacion = df[numeric_cols].corr()
# 
# print('Matriz de correlación:')
# print(correlacion.round(2))
# 
# # Encontrar features más correlacionadas
# print('\nCorrelaciones más altas (excluyendo diagonal):')
# import numpy as np
# mask = np.triu(np.ones_like(correlacion, dtype=bool), k=1)
# correlaciones_upper = correlacion.where(mask)
# 
# for col in correlaciones_upper.columns:
#     for idx in correlaciones_upper.index:
#         val = correlaciones_upper.loc[idx, col]
#         if pd.notna(val) and abs(val) > 0.8:
#             print(f'  {idx} <-> {col}: {val:.2f}')

print()

# ============================================
# PASO 7: Visualización (Opcional)
# ============================================
print('--- Paso 7: Visualización ---')

# Crear visualizaciones para entender mejor los datos
# Descomenta las siguientes líneas:

# import matplotlib.pyplot as plt
# 
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# 
# # Histogramas de cada feature
# for i, col in enumerate(iris.feature_names):
#     ax = axes[i // 2, i % 2]
#     for species in df['species_name'].unique():
#         subset = df[df['species_name'] == species]
#         ax.hist(subset[col], alpha=0.5, label=species, bins=15)
#     ax.set_xlabel(col)
#     ax.set_ylabel('Frecuencia')
#     ax.legend()
#     ax.set_title(f'Distribución de {col}')
# 
# plt.tight_layout()
# plt.savefig('exploracion_iris.png', dpi=150)
# print('Gráfico guardado como: exploracion_iris.png')
# plt.show()

print()

# ============================================
# RESUMEN
# ============================================
print('--- Resumen de la Exploración ---')

# Descomenta para ver el resumen:

# print(f'''
# Dataset Iris - Resumen:
# =======================
# - Samples: {len(df)}
# - Features: {len(iris.feature_names)}
# - Clases: {len(iris.target_names)} ({", ".join(iris.target_names)})
# - Balance: Sí (50 samples por clase)
# - Valores nulos: Ninguno
# - Features más correlacionadas: petal length y petal width (0.96)
# 
# ✅ Dataset listo para modelado de clasificación
# ''')

print('Ejercicio completado!')
