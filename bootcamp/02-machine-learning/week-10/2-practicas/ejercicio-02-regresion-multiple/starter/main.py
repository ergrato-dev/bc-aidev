"""
Ejercicio 02: Regresión Lineal Múltiple
=======================================
Predecir precios usando múltiples características.

Instrucciones:
- Lee cada paso en README.md
- Descomenta el código correspondiente
- Ejecuta y observa los resultados
"""

# ============================================
# PASO 1: Crear Dataset Multivariable
# ============================================
print('--- Paso 1: Crear Dataset ---')

# Dataset con 3 features que afectan el precio
# Descomenta las siguientes líneas:
# import numpy as np
# import pandas as pd
#
# np.random.seed(42)
# n_samples = 200
#
# data = {
#     'area': np.random.uniform(50, 250, n_samples),
#     'habitaciones': np.random.randint(1, 6, n_samples),
#     'antiguedad': np.random.uniform(0, 50, n_samples)
# }
# df = pd.DataFrame(data)
#
# # Precio real: 25000 + 1200*area + 15000*hab - 800*antig + ruido
# df['precio'] = (
#     25000 +
#     1200 * df['area'] +
#     15000 * df['habitaciones'] +
#     -800 * df['antiguedad'] +
#     np.random.normal(0, 20000, n_samples)
# )
#
# print(df.head())
# print(f'\nShape: {df.shape}')

print()

# ============================================
# PASO 2: Explorar Correlaciones
# ============================================
print('--- Paso 2: Correlaciones ---')

# Correlación de cada feature con el precio
# Descomenta las siguientes líneas:
# print('Correlación con precio:')
# print(df.corr()['precio'].sort_values(ascending=False))

print()

# ============================================
# PASO 3: Preparar Features y Target
# ============================================
print('--- Paso 3: Preparar Datos ---')

# Separar X (features) e y (target)
# Descomenta las siguientes líneas:
# X = df[['area', 'habitaciones', 'antiguedad']]
# y = df['precio']
#
# print(f'Features (X): {X.shape}')
# print(f'Target (y): {y.shape}')

print()

# ============================================
# PASO 4: Escalar Features
# ============================================
print('--- Paso 4: Escalar Features ---')

# StandardScaler: media=0, desviación=1
# Necesario para comparar importancia de coeficientes
# Descomenta las siguientes líneas:
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# print('Estadísticas después de escalar (train):')
# print(f'  Media: {X_train_scaled.mean(axis=0).round(2)}')
# print(f'  Std: {X_train_scaled.std(axis=0).round(2)}')

print()

# ============================================
# PASO 5: Entrenar Modelos
# ============================================
print('--- Paso 5: Entrenar Modelos ---')

# Comparamos modelo con datos originales vs escalados
# Descomenta las siguientes líneas:
# from sklearn.linear_model import LinearRegression
#
# # Modelo con datos originales
# model_raw = LinearRegression()
# model_raw.fit(X_train, y_train)
#
# # Modelo con datos escalados
# model_scaled = LinearRegression()
# model_scaled.fit(X_train_scaled, y_train)
#
# print('Modelos entrenados ✓')

print()

# ============================================
# PASO 6: Interpretar Coeficientes
# ============================================
print('--- Paso 6: Interpretar Coeficientes ---')

# Comparar coeficientes originales vs escalados
# Descomenta las siguientes líneas:
# print('Coeficientes (datos ORIGINALES):')
# print(f'  Intercepto: ${model_raw.intercept_:,.2f}')
# for name, coef in zip(X.columns, model_raw.coef_):
#     print(f'  {name}: ${coef:,.2f}')
#
# print('\nCoeficientes (datos ESCALADOS - importancia relativa):')
# print(f'  Intercepto: ${model_scaled.intercept_:,.2f}')
# for name, coef in zip(X.columns, model_scaled.coef_):
#     print(f'  {name}: {coef:,.2f}')
#
# # Ordenar por importancia
# importancia = list(zip(X.columns, np.abs(model_scaled.coef_)))
# importancia.sort(key=lambda x: x[1], reverse=True)
# print('\nFeatures por importancia:')
# for i, (name, imp) in enumerate(importancia, 1):
#     print(f'  {i}. {name} (|coef|={imp:.2f})')

print()

# ============================================
# PASO 7: Evaluar Modelo
# ============================================
print('--- Paso 7: Evaluar Modelo ---')

# R² y MAE (Mean Absolute Error)
# Descomenta las siguientes líneas:
# from sklearn.metrics import r2_score, mean_absolute_error
#
# y_pred = model_raw.predict(X_test)
#
# r2 = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
#
# print(f'R² (coef. determinación): {r2:.4f}')
# print(f'MAE (error absoluto medio): ${mae:,.2f}')
# print()
# print('Interpretación:')
# print(f'  - El modelo explica {r2*100:.1f}% de la varianza')
# print(f'  - Error promedio de predicción: ${mae:,.0f}')

print()

# ============================================
# PASO 8: Comparar con Regresión Simple
# ============================================
print('--- Paso 8: Comparar con Regresión Simple ---')

# ¿Cuánto mejora usar 3 features vs solo área?
# Descomenta las siguientes líneas:
# # Modelo solo con área
# model_simple = LinearRegression()
# model_simple.fit(X_train[['area']], y_train)
# y_pred_simple = model_simple.predict(X_test[['area']])
# r2_simple = r2_score(y_test, y_pred_simple)
#
# print(f'R² con 1 feature (área):     {r2_simple:.4f}')
# print(f'R² con 3 features (múltiple): {r2:.4f}')
# print(f'Mejora: +{(r2 - r2_simple)*100:.1f} puntos porcentuales')
# print()
# if r2 > r2_simple:
#     print('✅ La regresión múltiple mejora significativamente el modelo')
# else:
#     print('⚠️ Features adicionales no aportan mucho')

print()
print('=== Ejercicio completado ===')
