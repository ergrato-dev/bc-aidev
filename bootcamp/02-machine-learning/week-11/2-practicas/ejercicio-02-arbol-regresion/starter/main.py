"""
Ejercicio 02: Árbol de Decisión para Regresión
==============================================

Aprenderás a:
- Usar DecisionTreeRegressor para predicción de valores continuos
- Comparar con regresión lineal
- Calcular métricas de regresión (MSE, MAE, R²)
- Analizar feature importance

Instrucciones:
- Lee el README.md para entender cada paso
- Descomenta cada sección progresivamente
- Ejecuta y observa los resultados
"""

# ============================================
# PASO 1: Importar Librerías
# ============================================
print('--- Paso 1: Importar Librerías ---')

# Descomenta las siguientes líneas:
# from sklearn.datasets import fetch_california_housing
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import numpy as np
# import matplotlib.pyplot as plt

print('Librerías importadas correctamente')
print()

# ============================================
# PASO 2: Cargar California Housing Dataset
# ============================================
print('--- Paso 2: Cargar Dataset ---')

# Descomenta las siguientes líneas:
# housing = fetch_california_housing()
# X, y = housing.data, housing.target
#
# print(f"Features: {housing.feature_names}")
# print(f"Shape X: {X.shape}")
# print(f"Target: Precio medio de casas (en $100,000s)")
# print(f"Rango precio: {y.min():.2f} - {y.max():.2f}")

print()

# ============================================
# PASO 3: Dividir los Datos
# ============================================
print('--- Paso 3: Dividir Datos ---')

# Descomenta las siguientes líneas:
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42
# )
#
# print(f"Train: {X_train.shape[0]} muestras")
# print(f"Test: {X_test.shape[0]} muestras")

print()

# ============================================
# PASO 4: Entrenar Árbol de Regresión
# ============================================
print('--- Paso 4: Entrenar Árbol ---')

# Descomenta las siguientes líneas:
# tree_reg = DecisionTreeRegressor(
#     max_depth=5,
#     min_samples_leaf=10,
#     random_state=42
# )
#
# tree_reg.fit(X_train, y_train)
# print("Árbol de regresión entrenado")

print()

# ============================================
# PASO 5: Evaluar el Modelo
# ============================================
print('--- Paso 5: Evaluar Modelo ---')

# Descomenta las siguientes líneas:
# y_pred_tree = tree_reg.predict(X_test)
#
# mse = mean_squared_error(y_test, y_pred_tree)
# mae = mean_absolute_error(y_test, y_pred_tree)
# r2 = r2_score(y_test, y_pred_tree)
#
# print(f"\n--- Árbol de Regresión ---")
# print(f"MSE: {mse:.4f}")
# print(f"RMSE: {np.sqrt(mse):.4f}")
# print(f"MAE: {mae:.4f}")
# print(f"R²: {r2:.4f}")

print()

# ============================================
# PASO 6: Comparar con Regresión Lineal
# ============================================
print('--- Paso 6: Comparación con Regresión Lineal ---')

# Descomenta las siguientes líneas:
# lr = LinearRegression()
# lr.fit(X_train, y_train)
# y_pred_lr = lr.predict(X_test)
#
# mse_lr = mean_squared_error(y_test, y_pred_lr)
# r2_lr = r2_score(y_test, y_pred_lr)
#
# print(f"\n--- Regresión Lineal ---")
# print(f"MSE: {mse_lr:.4f}")
# print(f"RMSE: {np.sqrt(mse_lr):.4f}")
# print(f"R²: {r2_lr:.4f}")
#
# print(f"\n--- Comparación ---")
# print(f"Árbol R²: {r2:.4f}")
# print(f"Linear R²: {r2_lr:.4f}")

print()

# ============================================
# PASO 7: Visualizar Predicciones vs Real
# ============================================
print('--- Paso 7: Visualizar Predicciones ---')

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# # Árbol
# axes[0].scatter(y_test, y_pred_tree, alpha=0.5, s=10)
# axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
# axes[0].set_xlabel('Precio Real')
# axes[0].set_ylabel('Precio Predicho')
# axes[0].set_title(f'Decision Tree (R²={r2:.3f})')
#
# # Lineal
# axes[1].scatter(y_test, y_pred_lr, alpha=0.5, s=10)
# axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
# axes[1].set_xlabel('Precio Real')
# axes[1].set_ylabel('Precio Predicho')
# axes[1].set_title(f'Linear Regression (R²={r2_lr:.3f})')
#
# plt.tight_layout()
# plt.savefig('comparacion_regresion.png', dpi=150)
# plt.show()

print()

# ============================================
# PASO 8: Feature Importance
# ============================================
print('--- Paso 8: Feature Importance ---')

# Descomenta las siguientes líneas:
# importance = tree_reg.feature_importances_
# indices = np.argsort(importance)[::-1]
#
# print("\n--- Feature Importance ---")
# for i in range(len(housing.feature_names)):
#     print(f"{housing.feature_names[indices[i]]}: {importance[indices[i]]:.4f}")
#
# plt.figure(figsize=(10, 6))
# plt.bar(range(len(importance)), importance[indices], color='steelblue')
# plt.xticks(range(len(importance)), [housing.feature_names[i] for i in indices], rotation=45)
# plt.xlabel('Feature')
# plt.ylabel('Importancia')
# plt.title('Feature Importance - Decision Tree Regressor')
# plt.tight_layout()
# plt.savefig('feature_importance_regresion.png', dpi=150)
# plt.show()

print()

# ============================================
# PASO 9: Impacto de max_depth
# ============================================
print('--- Paso 9: Impacto de max_depth ---')

# Descomenta las siguientes líneas:
# depths = [2, 5, 10, 15, 20, None]
#
# for depth in depths:
#     tree_exp = DecisionTreeRegressor(max_depth=depth, random_state=42)
#     tree_exp.fit(X_train, y_train)
#
#     train_r2 = tree_exp.score(X_train, y_train)
#     test_r2 = tree_exp.score(X_test, y_test)
#
#     depth_str = str(depth) if depth else "None"
#     print(f"max_depth={depth_str:4s}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")

print()
print('=== Ejercicio completado ===')
