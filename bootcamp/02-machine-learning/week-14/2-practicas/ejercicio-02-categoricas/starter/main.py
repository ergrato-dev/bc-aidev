# ============================================
# EJERCICIO 02: CODIFICACIÓN DE CATEGÓRICAS
# ============================================
# Objetivo: Practicar OneHotEncoder, OrdinalEncoder
# y TargetEncoder
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================
# PASO 1: Crear Dataset de Ejemplo
# ============================================
print('--- Paso 1: Crear Dataset de Ejemplo ---')

# Descomenta las siguientes líneas:
# df = pd.DataFrame({
#     # Variable nominal (sin orden)
#     'color': np.random.choice(['rojo', 'verde', 'azul'], 100),
#     
#     # Variable ordinal (con orden)
#     'talla': np.random.choice(['S', 'M', 'L', 'XL'], 100),
#     
#     # Variable de alta cardinalidad
#     'ciudad': np.random.choice([
#         'Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Bilbao',
#         'Málaga', 'Zaragoza', 'Murcia', 'Palma', 'Las Palmas',
#         'Alicante', 'Córdoba', 'Valladolid', 'Vigo', 'Gijón'
#     ], 100),
#     
#     # Variable numérica (target para regresión)
#     'precio': np.random.uniform(100, 500, 100)
# })

# print("Dataset creado:")
# print(df.head(10))
# print(f"\nCategorías únicas por columna:")
# print(f"color: {df['color'].nunique()}")
# print(f"talla: {df['talla'].nunique()}")
# print(f"ciudad: {df['ciudad'].nunique()}")

print()

# ============================================
# PASO 2: OneHotEncoder Básico
# ============================================
print('--- Paso 2: OneHotEncoder Básico ---')

# OneHotEncoder crea una columna binaria por cada categoría
# Ideal para variables nominales (sin orden)

# Descomenta las siguientes líneas:
# from sklearn.preprocessing import OneHotEncoder

# # Crear encoder
# ohe = OneHotEncoder(sparse_output=False)

# # Aplicar a la columna 'color'
# color_encoded = ohe.fit_transform(df[['color']])

# # Ver nombres de las nuevas columnas
# feature_names = ohe.get_feature_names_out(['color'])
# print(f"Columnas creadas: {feature_names}")

# # Crear DataFrame con resultado
# df_color_ohe = pd.DataFrame(color_encoded, columns=feature_names)
# print("\nPrimeras filas codificadas:")
# print(pd.concat([df['color'], df_color_ohe], axis=1).head(10))

print()

# ============================================
# PASO 3: OneHotEncoder con drop='first'
# ============================================
print('--- Paso 3: OneHotEncoder con drop="first" ---')

# drop='first' elimina una columna para evitar multicolinealidad
# Esto es importante para modelos lineales

# Descomenta las siguientes líneas:
# ohe_drop = OneHotEncoder(sparse_output=False, drop='first')
# color_encoded_drop = ohe_drop.fit_transform(df[['color']])

# feature_names_drop = ohe_drop.get_feature_names_out(['color'])
# print(f"Columnas con drop='first': {feature_names_drop}")
# print(f"(Se eliminó una columna de referencia)")

# # Si color_azul=0 y color_rojo=0, entonces es 'verde'
# df_drop = pd.DataFrame(color_encoded_drop, columns=feature_names_drop)
# print("\nInterpretación:")
# print("Si ambas columnas son 0, la categoría es la eliminada (referencia)")

print()

# ============================================
# PASO 4: OrdinalEncoder para Variables Ordinales
# ============================================
print('--- Paso 4: OrdinalEncoder ---')

# OrdinalEncoder asigna enteros preservando un orden definido
# Ideal para variables ordinales: tallas, niveles de educación, etc.

# Descomenta las siguientes líneas:
# from sklearn.preprocessing import OrdinalEncoder

# # Definir el orden correcto
# orden_tallas = [['S', 'M', 'L', 'XL']]  # De menor a mayor

# oe = OrdinalEncoder(categories=orden_tallas)
# talla_encoded = oe.fit_transform(df[['talla']])

# df['talla_ordinal'] = talla_encoded
# print("Tallas codificadas con orden:")
# print(df[['talla', 'talla_ordinal']].drop_duplicates().sort_values('talla_ordinal'))

# # Verificar el mapeo
# print("\nMapeo: S=0, M=1, L=2, XL=3")
# print("Esto preserva el orden natural de las tallas")

print()

# ============================================
# PASO 5: Manejar Categorías Desconocidas
# ============================================
print('--- Paso 5: Manejar Categorías Desconocidas ---')

# En producción, pueden aparecer categorías que no estaban en train
# Debemos configurar los encoders para manejar esto

# Descomenta las siguientes líneas:
# from sklearn.model_selection import train_test_split

# # Split
# df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# # Simular categoría nueva en test
# df_test_new = df_test.copy()
# df_test_new.iloc[0, df_test_new.columns.get_loc('color')] = 'amarillo'  # Nueva categoría

# # ❌ Sin handle_unknown, esto daría error
# # ohe_error = OneHotEncoder(sparse_output=False)
# # ohe_error.fit(df_train[['color']])
# # ohe_error.transform(df_test_new[['color']])  # ERROR!

# # ✅ Con handle_unknown='ignore'
# ohe_safe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# ohe_safe.fit(df_train[['color']])

# # Ahora funciona - categorías desconocidas se ponen a 0 en todas las columnas
# test_encoded = ohe_safe.transform(df_test_new[['color']])
# print("Con handle_unknown='ignore', categorías nuevas → todos 0s")
# print(f"Primera fila (era 'amarillo'): {test_encoded[0]}")

print()

# ============================================
# PASO 6: TargetEncoder para Alta Cardinalidad
# ============================================
print('--- Paso 6: TargetEncoder ---')

# TargetEncoder reemplaza cada categoría por la media del target
# Ideal cuando hay muchas categorías únicas

# Descomenta las siguientes líneas:
# from sklearn.preprocessing import TargetEncoder

# # Calcular media de precio por ciudad (manual para entender)
# media_por_ciudad = df.groupby('ciudad')['precio'].mean()
# print("Media de precio por ciudad:")
# print(media_por_ciudad.sort_values(ascending=False).head())

# # Usar TargetEncoder de sklearn
# te = TargetEncoder(smooth='auto')
# ciudad_encoded = te.fit_transform(df[['ciudad']], df['precio'])

# df['ciudad_target'] = ciudad_encoded
# print("\nDataset con ciudad codificada:")
# print(df[['ciudad', 'precio', 'ciudad_target']].head(10))

# # Comparar dimensionalidad
# print(f"\nOneHotEncoder crearía {df['ciudad'].nunique()} columnas")
# print("TargetEncoder crea solo 1 columna")

print()

# ============================================
# PASO 7: Comparación pd.get_dummies vs OneHotEncoder
# ============================================
print('--- Paso 7: get_dummies vs OneHotEncoder ---')

# Descomenta las siguientes líneas:
# # pd.get_dummies - Rápido para exploración
# df_dummies = pd.get_dummies(df[['color']], prefix='color')
# print("pd.get_dummies:")
# print(df_dummies.head())

# # Diferencias clave:
# print("\n=== Comparación ===")
# print("""
# pd.get_dummies:
#   ✓ Rápido para exploración
#   ✓ Sintaxis simple
#   ✗ No guarda el encoder (no hay fit/transform)
#   ✗ Difícil manejar categorías nuevas
#   ✗ No compatible con Pipeline
# 
# OneHotEncoder:
#   ✓ Fit/Transform separados
#   ✓ Compatible con Pipeline
#   ✓ Maneja categorías desconocidas
#   ✓ Ideal para producción
#   ✗ Sintaxis más verbosa
# """)

print()

# ============================================
# EJEMPLO COMPLETO: Pipeline con Encoding
# ============================================
print('--- Ejemplo Completo: Pipeline ---')

# Descomenta las siguientes líneas:
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LinearRegression

# # Definir columnas por tipo
# nominal_cols = ['color']
# ordinal_cols = ['talla']
# numeric_cols = ['precio']  # Si tuviéramos más numéricas

# # Crear transformers
# preprocessor = ColumnTransformer([
#     ('nominal', OneHotEncoder(drop='first', handle_unknown='ignore'), nominal_cols),
#     ('ordinal', OrdinalEncoder(categories=[['S', 'M', 'L', 'XL']]), ordinal_cols),
# ], remainder='passthrough')

# # Ver resultado
# X = df[['color', 'talla']]
# X_transformed = preprocessor.fit_transform(X)
# print(f"Shape original: {X.shape}")
# print(f"Shape transformado: {X_transformed.shape}")
# print(f"\nFeature names: {preprocessor.get_feature_names_out()}")

print()

# ============================================
# RESUMEN
# ============================================
print('=== RESUMEN ===')
print("""
Guía de selección de encoder:

1. OneHotEncoder:
   - Variables nominales (sin orden)
   - Pocas categorías (< 10-15)
   - Usar drop='first' para modelos lineales

2. OrdinalEncoder:
   - Variables ordinales (con orden natural)
   - Definir el orden explícitamente
   - OK para tree-based models

3. TargetEncoder:
   - Alta cardinalidad (muchas categorías)
   - Usar con regularización/smoothing
   - Cuidado con data leakage

4. LabelEncoder:
   - SOLO para variable target en clasificación
   - NUNCA para features

RECUERDA: 
- Fit en train, transform en ambos
- handle_unknown='ignore' para producción
""")
