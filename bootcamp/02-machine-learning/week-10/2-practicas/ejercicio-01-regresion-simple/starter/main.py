"""
Ejercicio 01: Regresión Lineal Simple
=====================================
Predecir precios de casas basándose en el área (m²).

Instrucciones:
- Lee cada paso en README.md
- Descomenta el código correspondiente
- Ejecuta y observa los resultados
"""

# ============================================
# PASO 1: Crear Datos Sintéticos
# ============================================
print('--- Paso 1: Crear Datos Sintéticos ---')

# Generamos datos con relación lineal: precio = 30000 + 1500*area + ruido
# Descomenta las siguientes líneas:
# import numpy as np
#
# np.random.seed(42)
# area = np.random.uniform(50, 200, 100)  # 100 casas entre 50-200 m²
# precio = 30000 + 1500 * area + np.random.normal(0, 15000, 100)
#
# print(f'Generadas {len(area)} muestras')
# print(f'Área: min={area.min():.1f}, max={area.max():.1f} m²')
# print(f'Precio: min=${precio.min():,.0f}, max=${precio.max():,.0f}')

print()

# ============================================
# PASO 2: Preparar Datos para Scikit-learn
# ============================================
print('--- Paso 2: Preparar Datos ---')

# Scikit-learn requiere X como matriz 2D (n_samples, n_features)
# Descomenta las siguientes líneas:
# X = area.reshape(-1, 1)  # De shape (100,) a (100, 1)
# y = precio
#
# print(f'Shape de X: {X.shape}')
# print(f'Shape de y: {y.shape}')

print()

# ============================================
# PASO 3: Dividir en Train/Test
# ============================================
print('--- Paso 3: Dividir Train/Test ---')

# 80% para entrenar, 20% para evaluar
# Descomenta las siguientes líneas:
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# print(f'Train: {len(X_train)} muestras')
# print(f'Test: {len(X_test)} muestras')

print()

# ============================================
# PASO 4: Entrenar el Modelo
# ============================================
print('--- Paso 4: Entrenar Modelo ---')

# LinearRegression encuentra los mejores β₀ y β₁
# Descomenta las siguientes líneas:
# from sklearn.linear_model import LinearRegression
#
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# print(f'Intercepto (β₀): ${model.intercept_:,.2f}')
# print(f'Pendiente (β₁): ${model.coef_[0]:,.2f} por m²')
# print()
# print('Interpretación:')
# print(f'  - Una casa de 0 m² costaría ${model.intercept_:,.0f} (teórico)')
# print(f'  - Cada m² adicional aumenta el precio en ${model.coef_[0]:,.0f}')

print()

# ============================================
# PASO 5: Evaluar con R²
# ============================================
print('--- Paso 5: Evaluar con R² ---')

# R² = 1 - (SS_res / SS_tot), mide qué tan bien explica el modelo la varianza
# Descomenta las siguientes líneas:
# from sklearn.metrics import r2_score
#
# y_pred_train = model.predict(X_train)
# y_pred_test = model.predict(X_test)
#
# r2_train = r2_score(y_train, y_pred_train)
# r2_test = r2_score(y_test, y_pred_test)
#
# print(f'R² Train: {r2_train:.4f}')
# print(f'R² Test: {r2_test:.4f}')
# print()
# print('Interpretación:')
# print(f'  - El modelo explica {r2_test*100:.1f}% de la varianza en test')
# if abs(r2_train - r2_test) < 0.1:
#     print('  - ✅ No hay overfitting significativo')
# else:
#     print('  - ⚠️ Posible overfitting (diferencia > 0.1)')

print()

# ============================================
# PASO 6: Visualizar Resultado
# ============================================
print('--- Paso 6: Visualizar ---')

# Gráfico de dispersión con línea de regresión
# Descomenta las siguientes líneas:
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 6))
# plt.scatter(X_test, y_test, alpha=0.7, label='Datos reales', color='#3B82F6')
# plt.plot(X_test, y_pred_test, color='#EF4444', linewidth=2, label='Predicción')
# plt.xlabel('Área (m²)')
# plt.ylabel('Precio ($)')
# plt.title(f'Regresión Lineal Simple: Área vs Precio (R²={r2_test:.3f})')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('regresion_simple.png', dpi=150)
# print('Gráfico guardado: regresion_simple.png')
# plt.show()

print()
print('=== Ejercicio completado ===')
