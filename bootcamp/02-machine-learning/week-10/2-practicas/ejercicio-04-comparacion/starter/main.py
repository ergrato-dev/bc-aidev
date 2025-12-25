"""
Ejercicio 04: Comparación de Modelos
====================================
Comparar LinearRegression, Ridge y Lasso con datos multicolineales.

Instrucciones:
- Lee cada paso en README.md
- Descomenta el código correspondiente
- Ejecuta y observa los resultados
"""

# ============================================
# PASO 1: Crear Dataset con Multicolinealidad
# ============================================
print('--- Paso 1: Crear Dataset ---')

# Features correlacionadas causan problemas en regresión lineal
# Descomenta las siguientes líneas:
# import numpy as np
# import pandas as pd
#
# np.random.seed(42)
# n = 300
#
# # Feature base
# x1 = np.random.uniform(0, 100, n)
#
# # Features con diferentes niveles de correlación
# x2 = x1 * 0.8 + np.random.normal(0, 10, n)  # Muy correlacionada con x1
# x3 = np.random.uniform(0, 50, n)             # Independiente
# x4 = x1 * 0.5 + x3 * 0.3 + np.random.normal(0, 5, n)  # Mixta
#
# # Target: solo depende de x1 y x3 (x2 y x4 son "ruido correlacionado")
# y = 100 + 2*x1 + 0.5*x3 + np.random.normal(0, 20, n)
#
# print(f'Muestras: {n}')
# print(f'Features: x1, x2, x3, x4')
# print(f'Target y: depende de x1 y x3')

print()

# ============================================
# PASO 2: Detectar Multicolinealidad
# ============================================
print('--- Paso 2: Detectar Multicolinealidad ---')

# Correlación alta entre features = multicolinealidad
# Descomenta las siguientes líneas:
# df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})
#
# print('Matriz de correlación:')
# corr = df.corr()
# print(corr.round(2))
# print()
# print('Interpretación:')
# print('  - x1-x2: alta correlación (multicolinealidad)')
# print('  - x3: baja correlación (independiente)')
# print('  - x4: correlación moderada')

print()

# ============================================
# PASO 3: Entrenar Modelos
# ============================================
print('--- Paso 3: Entrenar Modelos ---')

# LinearRegression, Ridge (L2), Lasso (L1)
# Descomenta las siguientes líneas:
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# X = df.values
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# # Escalar es importante para regularización
# scaler = StandardScaler()
# X_train_s = scaler.fit_transform(X_train)
# X_test_s = scaler.transform(X_test)
#
# # Entrenar 3 modelos
# lr = LinearRegression().fit(X_train_s, y_train)
# ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
# lasso = Lasso(alpha=1.0).fit(X_train_s, y_train)
#
# print('Modelos entrenados:')
# print('  1. LinearRegression (sin regularización)')
# print('  2. Ridge (regularización L2)')
# print('  3. Lasso (regularización L1)')

print()

# ============================================
# PASO 4: Comparar Coeficientes
# ============================================
print('--- Paso 4: Comparar Coeficientes ---')

# Lasso tiende a hacer coeficientes = 0 (feature selection)
# Descomenta las siguientes líneas:
# features = ['x1', 'x2', 'x3', 'x4']
#
# print('Coeficientes por modelo:')
# print(f'{"Feature":<10} {"LinReg":>10} {"Ridge":>10} {"Lasso":>10}')
# print('-' * 42)
# for i, col in enumerate(features):
#     print(f'{col:<10} {lr.coef_[i]:>10.2f} {ridge.coef_[i]:>10.2f} {lasso.coef_[i]:>10.2f}')
#
# print()
# print('Observaciones:')
# n_zero_lasso = sum(1 for c in lasso.coef_ if abs(c) < 0.01)
# print(f'  - Lasso eliminó {n_zero_lasso} features (coef ≈ 0)')
# print('  - Ridge reduce pero no elimina coeficientes')
# print('  - LinReg puede tener coefs inestables con multicolinealidad')

print()

# ============================================
# PASO 5: Comparar R² en Test
# ============================================
print('--- Paso 5: Comparar R² ---')

# ¿Qué modelo generaliza mejor?
# Descomenta las siguientes líneas:
# from sklearn.metrics import r2_score, mean_squared_error
#
# models = {
#     'LinearRegression': lr,
#     'Ridge (α=1)': ridge,
#     'Lasso (α=1)': lasso
# }
#
# print(f'{"Modelo":<20} {"R² Test":>10} {"RMSE Test":>12}')
# print('-' * 44)
# for name, model in models.items():
#     y_pred = model.predict(X_test_s)
#     r2 = r2_score(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     print(f'{name:<20} {r2:>10.4f} {rmse:>12.2f}')

print()

# ============================================
# PASO 6: Encontrar Mejor α con CV
# ============================================
print('--- Paso 6: Cross-Validation para α ---')

# RidgeCV y LassoCV prueban múltiples alphas
# Descomenta las siguientes líneas:
# from sklearn.linear_model import RidgeCV, LassoCV
#
# alphas = [0.001, 0.01, 0.1, 1, 10, 100]
#
# ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_train_s, y_train)
# lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000).fit(X_train_s, y_train)
#
# print(f'Mejor α para Ridge: {ridge_cv.alpha_}')
# print(f'Mejor α para Lasso: {lasso_cv.alpha_}')
#
# # Evaluar modelos optimizados
# r2_ridge_cv = r2_score(y_test, ridge_cv.predict(X_test_s))
# r2_lasso_cv = r2_score(y_test, lasso_cv.predict(X_test_s))
#
# print(f'\nR² con α óptimo:')
# print(f'  Ridge: {r2_ridge_cv:.4f}')
# print(f'  Lasso: {r2_lasso_cv:.4f}')

print()

# ============================================
# PASO 7: Visualizar Efecto de α
# ============================================
print('--- Paso 7: Visualizar Efecto de α ---')

# Ver cómo los coeficientes cambian con regularización
# Descomenta las siguientes líneas:
# import matplotlib.pyplot as plt
#
# alphas_plot = np.logspace(-3, 3, 50)
# ridge_coefs = []
# lasso_coefs = []
#
# for a in alphas_plot:
#     ridge_temp = Ridge(alpha=a).fit(X_train_s, y_train)
#     lasso_temp = Lasso(alpha=a, max_iter=10000).fit(X_train_s, y_train)
#     ridge_coefs.append(ridge_temp.coef_)
#     lasso_coefs.append(lasso_temp.coef_)
#
# ridge_coefs = np.array(ridge_coefs)
# lasso_coefs = np.array(lasso_coefs)
#
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# # Ridge
# for i in range(4):
#     axes[0].plot(alphas_plot, ridge_coefs[:, i], label=f'x{i+1}')
# axes[0].set_xscale('log')
# axes[0].set_xlabel('Alpha (regularización)')
# axes[0].set_ylabel('Coeficiente')
# axes[0].set_title('Ridge: Coeficientes vs Alpha')
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)
#
# # Lasso
# for i in range(4):
#     axes[1].plot(alphas_plot, lasso_coefs[:, i], label=f'x{i+1}')
# axes[1].set_xscale('log')
# axes[1].set_xlabel('Alpha (regularización)')
# axes[1].set_ylabel('Coeficiente')
# axes[1].set_title('Lasso: Coeficientes vs Alpha (Feature Selection)')
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig('regularizacion_coefs.png', dpi=150)
# print('Gráfico guardado: regularizacion_coefs.png')
# plt.show()
#
# print()
# print('Observación:')
# print('  - Ridge: coefs → 0 gradualmente (nunca exactamente 0)')
# print('  - Lasso: coefs → 0 bruscamente (feature selection)')

print()
print('=== Ejercicio completado ===')
