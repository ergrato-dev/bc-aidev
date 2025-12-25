"""
Ejercicio 03: Métricas de Regresión
===================================
Aprende a calcular e interpretar métricas de regresión.
"""

# ============================================
# PASO 1: Preparar Datos de Regresión
# ============================================
print('--- Paso 1: Preparar Datos de Regresión ---')

# Descomenta las siguientes líneas:
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# 
# # Cargar datos (precios de casas en California)
# housing = fetch_california_housing()
# X, y = housing.data, housing.target
# 
# print(f"Muestras: {X.shape[0]}")
# print(f"Features: {X.shape[1]}")
# print(f"Feature names: {housing.feature_names}")
# print(f"Target: Precio medio de casas (en $100,000)")
# print(f"Rango de precios: {y.min():.2f} - {y.max():.2f}")
# 
# # Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# 
# # Escalar features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# 
# # Entrenar modelo
# model = LinearRegression()
# model.fit(X_train_scaled, y_train)
# y_pred = model.predict(X_test_scaled)
# 
# print(f"\nModelo entrenado: LinearRegression")

print()

# ============================================
# PASO 2: MSE y RMSE
# ============================================
print('--- Paso 2: MSE y RMSE ---')

# Descomenta las siguientes líneas:
# from sklearn.metrics import mean_squared_error
# 
# # MSE: Mean Squared Error
# mse = mean_squared_error(y_test, y_pred)
# print(f"MSE: {mse:.4f}")
# 
# # RMSE: Root Mean Squared Error
# rmse = np.sqrt(mse)
# print(f"RMSE: {rmse:.4f}")
# 
# # sklearn 1.4+ tiene root_mean_squared_error directamente
# # from sklearn.metrics import root_mean_squared_error
# # rmse = root_mean_squared_error(y_test, y_pred)
# 
# # Manual verification
# mse_manual = np.mean((y_test - y_pred) ** 2)
# print(f"\nVerificación manual MSE: {mse_manual:.4f}")
# 
# # Interpretación
# print(f"\nInterpretación:")
# print(f"  El error típico es de ${rmse * 100000:.0f}")
# print(f"  (RMSE está en las mismas unidades que el target)")

print()

# ============================================
# PASO 3: MAE
# ============================================
print('--- Paso 3: MAE ---')

# Descomenta las siguientes líneas:
# from sklearn.metrics import mean_absolute_error
# 
# # MAE: Mean Absolute Error
# mae = mean_absolute_error(y_test, y_pred)
# print(f"MAE: {mae:.4f}")
# 
# # Manual
# mae_manual = np.mean(np.abs(y_test - y_pred))
# print(f"Verificación manual: {mae_manual:.4f}")
# 
# # Comparar con RMSE
# print(f"\nComparación:")
# print(f"  RMSE: {rmse:.4f}")
# print(f"  MAE:  {mae:.4f}")
# print(f"  Diferencia: {rmse - mae:.4f}")
# print("\n  → RMSE > MAE indica presencia de errores grandes")
# print("  → Si RMSE ≈ MAE, los errores son uniformes")

print()

# ============================================
# PASO 4: R² (Coeficiente de Determinación)
# ============================================
print('--- Paso 4: R² ---')

# Descomenta las siguientes líneas:
# from sklearn.metrics import r2_score
# 
# # R²: Coefficient of Determination
# r2 = r2_score(y_test, y_pred)
# print(f"R²: {r2:.4f}")
# 
# # Manual
# ss_res = np.sum((y_test - y_pred) ** 2)  # Residual sum of squares
# ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)  # Total sum of squares
# r2_manual = 1 - (ss_res / ss_tot)
# print(f"Verificación manual: {r2_manual:.4f}")
# 
# # Interpretación
# print(f"\nInterpretación:")
# print(f"  El modelo explica el {r2 * 100:.1f}% de la varianza")
# print(f"  Restante {(1-r2) * 100:.1f}% es variabilidad no explicada")
# 
# # Comparar con baseline (predecir la media)
# y_baseline = np.full_like(y_pred, np.mean(y_train))
# r2_baseline = r2_score(y_test, y_baseline)
# print(f"\n  R² del baseline (predecir media): {r2_baseline:.4f}")

print()

# ============================================
# PASO 5: Análisis de Residuos
# ============================================
print('--- Paso 5: Análisis de Residuos ---')

# Descomenta las siguientes líneas:
# # Calcular residuos
# residuals = y_test - y_pred
# 
# print("Estadísticas de residuos:")
# print(f"  Media: {np.mean(residuals):.4f} (debería ser ~0)")
# print(f"  Std:   {np.std(residuals):.4f}")
# print(f"  Min:   {np.min(residuals):.4f}")
# print(f"  Max:   {np.max(residuals):.4f}")
# 
# # Crear visualizaciones
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# 
# # 1. Predicted vs Actual
# ax1 = axes[0, 0]
# ax1.scatter(y_test, y_pred, alpha=0.3, s=10)
# ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# ax1.set_xlabel('Valores Reales')
# ax1.set_ylabel('Predicciones')
# ax1.set_title('Predicted vs Actual')
# 
# # 2. Residuos vs Predicciones
# ax2 = axes[0, 1]
# ax2.scatter(y_pred, residuals, alpha=0.3, s=10)
# ax2.axhline(y=0, color='r', linestyle='--')
# ax2.set_xlabel('Predicciones')
# ax2.set_ylabel('Residuos')
# ax2.set_title('Residuos vs Predicciones')
# 
# # 3. Histograma de residuos
# ax3 = axes[1, 0]
# ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
# ax3.axvline(x=0, color='r', linestyle='--')
# ax3.set_xlabel('Residuos')
# ax3.set_ylabel('Frecuencia')
# ax3.set_title('Distribución de Residuos')
# 
# # 4. Q-Q plot
# ax4 = axes[1, 1]
# from scipy import stats
# stats.probplot(residuals, dist="norm", plot=ax4)
# ax4.set_title('Q-Q Plot (Normalidad)')
# 
# plt.tight_layout()
# plt.savefig('residuals_analysis.png', dpi=100)
# plt.close()
# print("\n✓ Gráfica guardada: residuals_analysis.png")

print()

# ============================================
# PASO 6: Comparar Modelos
# ============================================
print('--- Paso 6: Comparar Modelos ---')

# Descomenta las siguientes líneas:
# from sklearn.linear_model import Ridge, Lasso, ElasticNet
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.neighbors import KNeighborsRegressor
# 
# models = {
#     'LinearRegression': LinearRegression(),
#     'Ridge': Ridge(alpha=1.0),
#     'Lasso': Lasso(alpha=0.01),
#     'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
#     'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
#     'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
#     'KNN': KNeighborsRegressor(n_neighbors=5)
# }
# 
# print("Comparación de Modelos de Regresión:")
# print("-" * 70)
# print(f"{'Modelo':<20} | {'MSE':>10} | {'RMSE':>10} | {'MAE':>10} | {'R²':>8}")
# print("-" * 70)
# 
# results = []
# for name, model in models.items():
#     model.fit(X_train_scaled, y_train)
#     y_pred_model = model.predict(X_test_scaled)
#     
#     mse_m = mean_squared_error(y_test, y_pred_model)
#     rmse_m = np.sqrt(mse_m)
#     mae_m = mean_absolute_error(y_test, y_pred_model)
#     r2_m = r2_score(y_test, y_pred_model)
#     
#     results.append({'name': name, 'mse': mse_m, 'rmse': rmse_m, 'mae': mae_m, 'r2': r2_m})
#     print(f"{name:<20} | {mse_m:>10.4f} | {rmse_m:>10.4f} | {mae_m:>10.4f} | {r2_m:>8.4f}")
# 
# # Mejor modelo
# best = max(results, key=lambda x: x['r2'])
# print("-" * 70)
# print(f"✓ Mejor modelo (por R²): {best['name']} (R² = {best['r2']:.4f})")

print()

# ============================================
# PASO 7: Sensibilidad a Outliers
# ============================================
print('--- Paso 7: Sensibilidad a Outliers ---')

# Descomenta las siguientes líneas:
# # Crear datos simples para demostración
# np.random.seed(42)
# y_true_demo = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# y_pred_good = y_true_demo + np.random.randn(10) * 0.2  # Pequeño ruido
# 
# # Agregar outlier
# y_pred_outlier = y_pred_good.copy()
# y_pred_outlier[-1] = 20  # Un outlier grande
# 
# print("Impacto de un outlier en las métricas:")
# print("-" * 50)
# 
# print("\nSin outlier:")
# print(f"  MSE:  {mean_squared_error(y_true_demo, y_pred_good):.4f}")
# print(f"  RMSE: {np.sqrt(mean_squared_error(y_true_demo, y_pred_good)):.4f}")
# print(f"  MAE:  {mean_absolute_error(y_true_demo, y_pred_good):.4f}")
# 
# print("\nCon outlier (último valor predicho = 20 en vez de ~10):")
# print(f"  MSE:  {mean_squared_error(y_true_demo, y_pred_outlier):.4f}")
# print(f"  RMSE: {np.sqrt(mean_squared_error(y_true_demo, y_pred_outlier)):.4f}")
# print(f"  MAE:  {mean_absolute_error(y_true_demo, y_pred_outlier):.4f}")
# 
# # Calcular incremento
# mse_inc = mean_squared_error(y_true_demo, y_pred_outlier) / mean_squared_error(y_true_demo, y_pred_good)
# mae_inc = mean_absolute_error(y_true_demo, y_pred_outlier) / mean_absolute_error(y_true_demo, y_pred_good)
# 
# print(f"\nIncremento por outlier:")
# print(f"  MSE aumentó {mse_inc:.1f}x")
# print(f"  MAE aumentó {mae_inc:.1f}x")
# print("\n→ MSE es más sensible a outliers que MAE")
# print("→ Usar MAE si los datos tienen outliers")

print()

# ============================================
# RESUMEN
# ============================================
print('--- Resumen ---')

# Descomenta las siguientes líneas:
# print("=" * 55)
# print("RESUMEN DE MÉTRICAS DE REGRESIÓN")
# print("=" * 55)
# print("\nMétrica | Características")
# print("-" * 55)
# print("MSE     | Penaliza errores grandes, sensible a outliers")
# print("RMSE    | MSE en unidades originales, interpretable")
# print("MAE     | Robusto a outliers, error absoluto promedio")
# print("R²      | Varianza explicada (0-1), comparar modelos")
# print("-" * 55)
# print("\nCuándo usar:")
# print("- General sin outliers → RMSE")
# print("- Datos con outliers → MAE")
# print("- Comparar modelos → R²")
# print("- Entrenamiento/optimización → MSE")
# print("=" * 55)

print()
print("¡Ejercicio completado!")
