# ============================================
# EJERCICIO 01: TRANSFORMACIONES NUMÉRICAS
# ============================================
# Objetivo: Practicar StandardScaler, MinMaxScaler,
# RobustScaler y PowerTransformer
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

# ============================================
# PASO 1: Crear Datos de Ejemplo
# ============================================
print('--- Paso 1: Crear Datos de Ejemplo ---')

# Datos normales
# Descomenta las siguientes líneas:
# normal_data = np.random.normal(loc=100, scale=15, size=1000)

# Datos con outliers
# outliers = np.array([300, 350, 400, -50, -100])
# data_with_outliers = np.concatenate([normal_data, outliers])

# Datos sesgados (distribución exponencial)
# skewed_data = np.random.exponential(scale=50, size=1000)

# Crear DataFrame
# df = pd.DataFrame({
#     'normal': np.random.normal(100, 15, 1000),
#     'con_outliers': np.concatenate([
#         np.random.normal(100, 15, 995),
#         np.array([300, 350, 400, -50, -100])
#     ]),
#     'sesgado': np.random.exponential(50, 1000)
# })

# print("Estadísticas originales:")
# print(df.describe())

print()

# ============================================
# PASO 2: Aplicar StandardScaler
# ============================================
print('--- Paso 2: Aplicar StandardScaler ---')

# El StandardScaler aplica la fórmula: z = (x - μ) / σ
# Resultado: media ≈ 0, std ≈ 1

# Descomenta las siguientes líneas:
# from sklearn.preprocessing import StandardScaler

# scaler_standard = StandardScaler()
# df_standard = pd.DataFrame(
#     scaler_standard.fit_transform(df),
#     columns=[f'{col}_std' for col in df.columns]
# )

# print("Después de StandardScaler:")
# print(df_standard.describe().round(3))

# Verificar media y std
# print(f"\nMedia de 'normal_std': {df_standard['normal_std'].mean():.6f}")
# print(f"Std de 'normal_std': {df_standard['normal_std'].std():.6f}")

print()

# ============================================
# PASO 3: Aplicar MinMaxScaler
# ============================================
print('--- Paso 3: Aplicar MinMaxScaler ---')

# El MinMaxScaler escala a [0, 1]: x' = (x - min) / (max - min)

# Descomenta las siguientes líneas:
# from sklearn.preprocessing import MinMaxScaler

# scaler_minmax = MinMaxScaler()
# df_minmax = pd.DataFrame(
#     scaler_minmax.fit_transform(df),
#     columns=[f'{col}_mm' for col in df.columns]
# )

# print("Después de MinMaxScaler:")
# print(df_minmax.describe().round(3))

# Verificar rango
# print(f"\nMin de 'normal_mm': {df_minmax['normal_mm'].min():.6f}")
# print(f"Max de 'normal_mm': {df_minmax['normal_mm'].max():.6f}")

print()

# ============================================
# PASO 4: Aplicar RobustScaler
# ============================================
print('--- Paso 4: Aplicar RobustScaler ---')

# RobustScaler usa mediana e IQR: x' = (x - Q2) / (Q3 - Q1)
# Es robusto a outliers porque no usa media ni std

# Descomenta las siguientes líneas:
# from sklearn.preprocessing import RobustScaler

# scaler_robust = RobustScaler()
# df_robust = pd.DataFrame(
#     scaler_robust.fit_transform(df),
#     columns=[f'{col}_rob' for col in df.columns]
# )

# print("Después de RobustScaler:")
# print(df_robust.describe().round(3))

# Comparar efecto en columna con outliers
# print("\nComparación en datos con outliers:")
# print(f"StandardScaler range: [{df_standard['con_outliers_std'].min():.2f}, {df_standard['con_outliers_std'].max():.2f}]")
# print(f"RobustScaler range: [{df_robust['con_outliers_rob'].min():.2f}, {df_robust['con_outliers_rob'].max():.2f}]")

print()

# ============================================
# PASO 5: Comparar Escaladores Visualmente
# ============================================
print('--- Paso 5: Comparar Escaladores Visualmente ---')

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# # Original
# axes[0, 0].hist(df['con_outliers'], bins=50, alpha=0.7, color='steelblue')
# axes[0, 0].set_title('Original (con outliers)')
# axes[0, 0].axvline(df['con_outliers'].mean(), color='red', linestyle='--', label='Media')
# axes[0, 0].axvline(df['con_outliers'].median(), color='green', linestyle='--', label='Mediana')
# axes[0, 0].legend()

# # StandardScaler
# axes[0, 1].hist(df_standard['con_outliers_std'], bins=50, alpha=0.7, color='coral')
# axes[0, 1].set_title('StandardScaler')

# # MinMaxScaler
# axes[1, 0].hist(df_minmax['con_outliers_mm'], bins=50, alpha=0.7, color='mediumseagreen')
# axes[1, 0].set_title('MinMaxScaler')

# # RobustScaler
# axes[1, 1].hist(df_robust['con_outliers_rob'], bins=50, alpha=0.7, color='mediumpurple')
# axes[1, 1].set_title('RobustScaler')

# plt.tight_layout()
# plt.savefig('comparacion_escaladores.png', dpi=150)
# plt.show()
# print("Gráfico guardado como 'comparacion_escaladores.png'")

print()

# ============================================
# PASO 6: PowerTransformer para Distribuciones Sesgadas
# ============================================
print('--- Paso 6: PowerTransformer ---')

# PowerTransformer normaliza distribuciones sesgadas
# Box-Cox: solo para valores positivos
# Yeo-Johnson: para cualquier valor

# Descomenta las siguientes líneas:
# from sklearn.preprocessing import PowerTransformer
# from scipy import stats

# # Calcular skewness original
# skew_original = stats.skew(df['sesgado'])
# print(f"Skewness original: {skew_original:.3f}")

# # Aplicar Yeo-Johnson
# pt = PowerTransformer(method='yeo-johnson')
# sesgado_transformed = pt.fit_transform(df[['sesgado']])

# skew_transformed = stats.skew(sesgado_transformed)
# print(f"Skewness después de PowerTransformer: {skew_transformed[0]:.3f}")

# # Visualizar
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# axes[0].hist(df['sesgado'], bins=50, alpha=0.7, color='steelblue')
# axes[0].set_title(f'Original (skew={skew_original:.2f})')

# axes[1].hist(sesgado_transformed, bins=50, alpha=0.7, color='coral')
# axes[1].set_title(f'PowerTransformer (skew={skew_transformed[0]:.2f})')

# plt.tight_layout()
# plt.savefig('power_transformer.png', dpi=150)
# plt.show()
# print("Gráfico guardado como 'power_transformer.png'")

print()

# ============================================
# PASO 7: Principio Fit on Train
# ============================================
print('--- Paso 7: Fit on Train, Transform on Both ---')

# REGLA DE ORO: fit solo en train, transform en train y test

# Descomenta las siguientes líneas:
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Simular datos
# X = np.random.normal(100, 20, (1000, 3))
# y = np.random.randint(0, 2, 1000)

# # Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # ✅ CORRECTO
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
# X_test_scaled = scaler.transform(X_test)        # solo transform

# print("✅ Método correcto:")
# print(f"Media train: {X_train_scaled.mean():.6f}")
# print(f"Media test: {X_test_scaled.mean():.6f}")  # Puede no ser exactamente 0

# # ❌ INCORRECTO (DATA LEAKAGE)
# scaler_bad = StandardScaler()
# X_all_scaled = scaler_bad.fit_transform(X)  # NUNCA hacer esto antes del split

# print("\n❌ El método incorrecto causa data leakage!")
# print("El scaler 've' información del test set durante fit.")

print()

# ============================================
# RESUMEN
# ============================================
print('=== RESUMEN ===')
print("""
Escaladores y cuándo usarlos:

1. StandardScaler: 
   - Datos normales sin outliers extremos
   - SVM, Logistic Regression, PCA

2. MinMaxScaler:
   - Redes neuronales
   - Cuando necesitas rango específico [0,1]
   - ⚠️ Muy sensible a outliers

3. RobustScaler:
   - Datos con outliers
   - Cuando no puedes eliminar outliers

4. PowerTransformer:
   - Distribuciones sesgadas
   - Para normalizar antes de modelos que asumen normalidad

RECUERDA: Siempre fit en train, transform en ambos!
""")
