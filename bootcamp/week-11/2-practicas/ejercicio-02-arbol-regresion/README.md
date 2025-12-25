# Ejercicio 02: Árbol de Decisión para Regresión

## 🎯 Objetivo

Usar `DecisionTreeRegressor` para predecir precios de casas con el dataset California Housing, y comparar con regresión lineal.

## 📋 Conceptos Clave

- `DecisionTreeRegressor` para valores continuos
- Criterios de división: `squared_error`, `absolute_error`
- Métricas de regresión: MSE, MAE, R²
- Comparación árbol vs regresión lineal

## ⏱️ Tiempo Estimado

35 minutos

---

## 📝 Instrucciones

### Paso 1: Importar Librerías

```python
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Cargar California Housing Dataset

Dataset con precios de casas en California (20,640 muestras, 8 features).

```python
housing = fetch_california_housing()
X, y = housing.data, housing.target

print(f"Features: {housing.feature_names}")
print(f"Shape X: {X.shape}")
print(f"Target: Precio medio de casas (en $100,000s)")
print(f"Rango precio: {y.min():.2f} - {y.max():.2f}")
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Dividir los Datos

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"Train: {X_train.shape[0]} muestras")
print(f"Test: {X_test.shape[0]} muestras")
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Entrenar Árbol de Regresión

```python
tree_reg = DecisionTreeRegressor(
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)

tree_reg.fit(X_train, y_train)
print("Árbol de regresión entrenado")
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Evaluar el Modelo

Usamos MSE, MAE y R² como métricas.

```python
y_pred_tree = tree_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred_tree)
mae = mean_absolute_error(y_test, y_pred_tree)
r2 = r2_score(y_test, y_pred_tree)

print(f"\n--- Árbol de Regresión ---")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Comparar con Regresión Lineal

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"\n--- Regresión Lineal ---")
print(f"MSE: {mse_lr:.4f}")
print(f"RMSE: {np.sqrt(mse_lr):.4f}")
print(f"R²: {r2_lr:.4f}")

print(f"\n--- Comparación ---")
print(f"Árbol R²: {r2:.4f}")
print(f"Linear R²: {r2_lr:.4f}")
```

**Descomenta** la sección del Paso 6.

---

### Paso 7: Visualizar Predicciones vs Real

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Árbol
axes[0].scatter(y_test, y_pred_tree, alpha=0.5, s=10)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
axes[0].set_xlabel('Precio Real')
axes[0].set_ylabel('Precio Predicho')
axes[0].set_title(f'Decision Tree (R²={r2:.3f})')

# Lineal
axes[1].scatter(y_test, y_pred_lr, alpha=0.5, s=10)
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
axes[1].set_xlabel('Precio Real')
axes[1].set_ylabel('Precio Predicho')
axes[1].set_title(f'Linear Regression (R²={r2_lr:.3f})')

plt.tight_layout()
plt.savefig('comparacion_regresion.png', dpi=150)
plt.show()
```

**Descomenta** la sección del Paso 7.

---

### Paso 8: Feature Importance

```python
importance = tree_reg.feature_importances_
indices = np.argsort(importance)[::-1]

print("\n--- Feature Importance ---")
for i in range(len(housing.feature_names)):
    print(f"{housing.feature_names[indices[i]]}: {importance[indices[i]]:.4f}")

plt.figure(figsize=(10, 6))
plt.bar(range(len(importance)), importance[indices], color='steelblue')
plt.xticks(range(len(importance)), [housing.feature_names[i] for i in indices], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importancia')
plt.title('Feature Importance - Decision Tree Regressor')
plt.tight_layout()
plt.savefig('feature_importance_regresion.png', dpi=150)
plt.show()
```

**Descomenta** la sección del Paso 8.

---

### Paso 9: Impacto de max_depth

```python
print("\n--- Impacto de max_depth ---")
depths = [2, 5, 10, 15, 20, None]

for depth in depths:
    tree_exp = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree_exp.fit(X_train, y_train)

    train_r2 = tree_exp.score(X_train, y_train)
    test_r2 = tree_exp.score(X_test, y_test)

    depth_str = str(depth) if depth else "None"
    print(f"max_depth={depth_str:4s}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")
```

**Descomenta** la sección del Paso 9.

---

## ✅ Resultado Esperado

1. Dataset California Housing cargado (20,640 muestras)
2. Árbol de regresión con R² ~0.62-0.70
3. Comparación visual árbol vs regresión lineal
4. Feature importance (MedInc suele ser la más importante)
5. Análisis del impacto de max_depth

---

## 🔬 Experimenta

1. Prueba `criterion='absolute_error'` en lugar del default
2. Ajusta `min_samples_split` y `min_samples_leaf`
3. ¿Qué profundidad da el mejor balance train/test?

---

## 📚 Recursos

- [DecisionTreeRegressor - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
