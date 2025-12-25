# ============================================
# EJERCICIO 03: MANEJO DE MISSING DATA
# ============================================
# Objetivo: Practicar SimpleImputer, KNNImputer
# y Missing Indicator
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================
# PASO 1: Crear Dataset con Missing Values
# ============================================
print('--- Paso 1: Crear Dataset con Missing Values ---')

# Descomenta las siguientes líneas:
# # Crear dataset base
# n_samples = 500
# df = pd.DataFrame({
#     'edad': np.random.randint(18, 80, n_samples).astype(float),
#     'ingresos': np.random.normal(50000, 15000, n_samples),
#     'antiguedad': np.random.randint(0, 30, n_samples).astype(float),
#     'educacion': np.random.choice(['secundaria', 'universidad', 'master'], n_samples),
#     'ciudad': np.random.choice(['Madrid', 'Barcelona', 'Valencia'], n_samples)
# })

# # Introducir missing values (diferentes patrones)
# # Missing completely at random (MCAR) - 10% en edad
# mask_edad = np.random.random(n_samples) < 0.10
# df.loc[mask_edad, 'edad'] = np.nan

# # Missing at random (MAR) - ingresos falta más en jóvenes
# prob_missing = np.where(df['edad'] < 30, 0.3, 0.05)
# mask_ingresos = np.random.random(n_samples) < prob_missing
# df.loc[mask_ingresos, 'ingresos'] = np.nan

# # Missing not at random (MNAR) - antiguedad falta cuando es 0
# mask_antiguedad = (df['antiguedad'] == 0) | (np.random.random(n_samples) < 0.05)
# df.loc[mask_antiguedad, 'antiguedad'] = np.nan

# # Missing en categóricas
# mask_educacion = np.random.random(n_samples) < 0.08
# df.loc[mask_educacion, 'educacion'] = np.nan

# print("Dataset con valores faltantes:")
# print(df.head(15))

print()

# ============================================
# PASO 2: Diagnóstico de Missing Data
# ============================================
print('--- Paso 2: Diagnóstico de Missing Data ---')

# Descomenta las siguientes líneas:
# # Contar missing por columna
# missing_count = df.isnull().sum()
# missing_percent = (df.isnull().sum() / len(df) * 100).round(2)

# missing_summary = pd.DataFrame({
#     'Missing Count': missing_count,
#     'Missing %': missing_percent
# })
# print("Resumen de valores faltantes:")
# print(missing_summary)

# # Visualizar patrón de missing
# import seaborn as sns

# plt.figure(figsize=(10, 6))
# sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
# plt.title('Patrón de Valores Faltantes')
# plt.tight_layout()
# plt.savefig('missing_pattern.png', dpi=150)
# plt.show()
# print("Gráfico guardado como 'missing_pattern.png'")

# # Filas con algún missing
# filas_con_missing = df.isnull().any(axis=1).sum()
# print(f"\nFilas con al menos un missing: {filas_con_missing} ({filas_con_missing/len(df)*100:.1f}%)")

print()

# ============================================
# PASO 3: SimpleImputer - Estrategias Básicas
# ============================================
print('--- Paso 3: SimpleImputer ---')

# Descomenta las siguientes líneas:
# from sklearn.impute import SimpleImputer

# # Separar columnas por tipo
# numeric_cols = ['edad', 'ingresos', 'antiguedad']
# categorical_cols = ['educacion', 'ciudad']

# # --- Imputar numéricas con mediana ---
# imputer_median = SimpleImputer(strategy='median')
# df_numeric_imputed = pd.DataFrame(
#     imputer_median.fit_transform(df[numeric_cols]),
#     columns=numeric_cols
# )

# print("Numéricas imputadas con mediana:")
# print(df_numeric_imputed.isnull().sum())

# # Comparar estadísticas
# print("\nComparación antes/después:")
# print(f"Media edad original: {df['edad'].mean():.2f}")
# print(f"Media edad imputada: {df_numeric_imputed['edad'].mean():.2f}")

# # --- Imputar categóricas con moda ---
# imputer_mode = SimpleImputer(strategy='most_frequent')
# df_cat_imputed = pd.DataFrame(
#     imputer_mode.fit_transform(df[categorical_cols]),
#     columns=categorical_cols
# )

# print("\nCategóricas imputadas con moda:")
# print(df_cat_imputed.isnull().sum())

print()

# ============================================
# PASO 4: KNNImputer
# ============================================
print('--- Paso 4: KNNImputer ---')

# KNNImputer usa los K vecinos más cercanos para imputar
# Puede capturar relaciones entre variables

# Descomenta las siguientes líneas:
# from sklearn.impute import KNNImputer

# # Solo para numéricas
# knn_imputer = KNNImputer(n_neighbors=5)
# df_knn_imputed = pd.DataFrame(
#     knn_imputer.fit_transform(df[numeric_cols]),
#     columns=numeric_cols
# )

# print("Imputación con KNNImputer:")
# print(df_knn_imputed.isnull().sum())

# # Comparar métodos
# print("\nComparación de métodos de imputación:")
# 
# # Encontrar índices que tenían missing en 'ingresos'
# missing_idx = df[df['ingresos'].isnull()].index[:5]
# 
# comparison = pd.DataFrame({
#     'Original': df.loc[missing_idx, 'edad'],  # edad para contexto
#     'Median': df_numeric_imputed.loc[missing_idx, 'ingresos'],
#     'KNN': df_knn_imputed.loc[missing_idx, 'ingresos']
# })
# print(comparison)
# print("\nKNN considera edad y otras features para imputar ingresos")

print()

# ============================================
# PASO 5: Missing Indicator
# ============================================
print('--- Paso 5: Missing Indicator ---')

# A veces el hecho de que falte un valor es informativo
# Creamos columnas binarias que indican si el valor estaba faltante

# Descomenta las siguientes líneas:
# from sklearn.impute import MissingIndicator

# # Crear indicadores de missing
# indicator = MissingIndicator(features='all')
# missing_flags = indicator.fit_transform(df[numeric_cols])

# # Convertir a DataFrame
# indicator_cols = [f'{col}_was_missing' for col in numeric_cols]
# df_indicators = pd.DataFrame(missing_flags, columns=indicator_cols)

# print("Columnas de indicador de missing:")
# print(df_indicators.head(10))

# # Verificar que coincide con los NaN originales
# print("\nVerificación:")
# print(f"NaN en edad: {df['edad'].isnull().sum()}")
# print(f"True en edad_was_missing: {df_indicators['edad_was_missing'].sum()}")

# # Combinar con datos imputados
# df_final = pd.concat([df_numeric_imputed, df_indicators], axis=1)
# print(f"\nDataset final shape: {df_final.shape}")

print()

# ============================================
# PASO 6: Pipeline Completo con Imputación
# ============================================
print('--- Paso 6: Pipeline Completo ---')

# Descomenta las siguientes líneas:
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder

# # Definir columnas
# numeric_features = ['edad', 'ingresos', 'antiguedad']
# categorical_features = ['educacion', 'ciudad']

# # Pipeline numérico: imputar + escalar
# numeric_transformer = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# # Pipeline categórico: imputar + one-hot
# categorical_transformer = Pipeline([
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
# ])

# # Combinar con ColumnTransformer
# preprocessor = ColumnTransformer([
#     ('num', numeric_transformer, numeric_features),
#     ('cat', categorical_transformer, categorical_features)
# ])

# # Aplicar
# X = df.drop(columns=['ciudad'])  # Usamos ciudad como ejemplo
# X_transformed = preprocessor.fit_transform(df)

# print(f"Shape original: {df.shape}")
# print(f"Shape transformado: {X_transformed.shape}")
# print(f"\nNaN en resultado: {np.isnan(X_transformed).sum()}")
# print("✓ Todos los missing han sido manejados")

# # Ver nombres de features
# feature_names = preprocessor.get_feature_names_out()
# print(f"\nFeature names: {feature_names}")

print()

# ============================================
# RESUMEN
# ============================================
print('=== RESUMEN ===')
print("""
Estrategias de imputación:

1. Eliminar filas:
   - Solo si < 5% missing y es aleatorio (MCAR)
   - df.dropna()

2. SimpleImputer:
   - strategy='mean': para distribuciones simétricas
   - strategy='median': robusto a outliers
   - strategy='most_frequent': para categóricas
   - strategy='constant': valor fijo

3. KNNImputer:
   - Usa relaciones entre features
   - Más preciso pero más lento
   - Solo para numéricas

4. IterativeImputer (MICE):
   - Modelo iterativo multivariado
   - Más sofisticado

5. Missing Indicator:
   - Cuando el missing es informativo
   - Combinar con imputación

RECUERDA:
- Diagnosticar antes de imputar
- Diferentes estrategias para diferentes columnas
- Fit en train, transform en test
""")
