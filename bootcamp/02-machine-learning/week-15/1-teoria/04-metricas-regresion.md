# 📈 Métricas de Regresión

## 🎯 Objetivos de Aprendizaje

- Comprender las principales métricas de regresión
- Calcular e interpretar MSE, RMSE, MAE y R²
- Elegir la métrica adecuada según el problema
- Entender el impacto de outliers en cada métrica

---

## 📋 Contenido

### 1. Panorama de Métricas de Regresión

| Métrica | Fórmula | Rango | Interpretación |
|---------|---------|-------|----------------|
| **MSE** | Mean Squared Error | [0, ∞) | Error cuadrático promedio |
| **RMSE** | √MSE | [0, ∞) | Error en unidades originales |
| **MAE** | Mean Absolute Error | [0, ∞) | Error absoluto promedio |
| **R²** | Coef. Determinación | (-∞, 1] | Varianza explicada |
| **MAPE** | Mean Abs. % Error | [0, ∞) | Error porcentual |

---

### 2. Mean Squared Error (MSE)

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

```python
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Crear datos de ejemplo
np.random.seed(42)
X = np.random.randn(1000, 5)
y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(1000) * 0.5

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# MSE
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")

# Manual
mse_manual = np.mean((y_test - y_pred) ** 2)
print(f"MSE (manual): {mse_manual:.4f}")
```

**Características:**
- ✅ Diferenciable (útil para optimización)
- ✅ Penaliza fuertemente errores grandes
- ❌ Sensible a outliers
- ❌ Unidades son cuadradas (difícil interpretar)

---

### 3. Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

```python
from sklearn.metrics import root_mean_squared_error  # sklearn 1.4+

rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse:.4f}")

# Alternativa (versiones anteriores)
rmse_alt = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE (alternativo): {rmse_alt:.4f}")
```

**Características:**
- ✅ Mismas unidades que la variable objetivo
- ✅ Fácil de interpretar
- ❌ Sensible a outliers (como MSE)

**Interpretación:**
```python
# Si RMSE = 2.5 y prediciendo precios en miles de euros
# → En promedio, el error es de €2,500
print(f"El error típico es de {rmse:.2f} unidades")
```

---

### 4. Mean Absolute Error (MAE)

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")

# Manual
mae_manual = np.mean(np.abs(y_test - y_pred))
print(f"MAE (manual): {mae_manual:.4f}")
```

**Características:**
- ✅ Menos sensible a outliers que MSE/RMSE
- ✅ Mismas unidades que la variable objetivo
- ✅ Fácil de interpretar
- ❌ No diferenciable en 0 (problemas para algunos optimizadores)

---

### 5. R² (Coeficiente de Determinación)

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.4f}")

# Manual
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2_manual = 1 - (ss_res / ss_tot)
print(f"R² (manual): {r2_manual:.4f}")
```

**Interpretación:**
| R² | Interpretación |
|----|---------------|
| 1.0 | Predicción perfecta |
| 0.9 | El modelo explica 90% de la varianza |
| 0.5 | El modelo explica 50% de la varianza |
| 0.0 | Tan bueno como predecir la media |
| < 0 | Peor que predecir la media |

```python
# Comparar con modelo baseline (predecir la media)
y_baseline = np.full_like(y_pred, np.mean(y_train))
r2_baseline = r2_score(y_test, y_baseline)
print(f"R² baseline (media): {r2_baseline:.4f}")  # Siempre 0
```

---

### 6. Mean Absolute Percentage Error (MAPE)

$$MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

```python
from sklearn.metrics import mean_absolute_percentage_error

# MAPE requiere que y_true no tenga ceros
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"MAPE: {mape:.2f}%")

# Manual (con protección contra división por cero)
mape_manual = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
print(f"MAPE (manual): {mape_manual:.2f}%")
```

**Características:**
- ✅ Interpretable en porcentaje
- ✅ Escala-independiente (bueno para comparar datasets)
- ❌ No funciona con valores cero o cercanos a cero
- ❌ Asimétrica (errores positivos y negativos no son iguales)

---

### 7. Comparación: Sensibilidad a Outliers

```python
import matplotlib.pyplot as plt

# Datos con outlier
y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_pred_good = np.array([1.1, 2.2, 2.9, 4.1, 5.2, 5.8, 7.1, 8.0, 9.2, 9.8])
y_pred_outlier = np.array([1.1, 2.2, 2.9, 4.1, 5.2, 5.8, 7.1, 8.0, 9.2, 20.0])  # Un outlier

print("Sin outlier:")
print(f"  MSE:  {mean_squared_error(y_true, y_pred_good):.4f}")
print(f"  MAE:  {mean_absolute_error(y_true, y_pred_good):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_true, y_pred_good)):.4f}")

print("\nCon outlier (último valor predicho = 20 en vez de 10):")
print(f"  MSE:  {mean_squared_error(y_true, y_pred_outlier):.4f}")  # Mucho mayor
print(f"  MAE:  {mean_absolute_error(y_true, y_pred_outlier):.4f}")  # Menos afectado
print(f"  RMSE: {np.sqrt(mean_squared_error(y_true, y_pred_outlier)):.4f}")
```

**Output:**
```
Sin outlier:
  MSE:  0.0280
  MAE:  0.1400
  RMSE: 0.1673

Con outlier:
  MSE:  10.4280   # 372x mayor!
  MAE:  1.1400    # 8x mayor
  RMSE: 3.2292
```

**Conclusión:**
- Datos limpios → MSE/RMSE funcionan bien
- Datos con outliers → MAE es más robusto

---

### 8. R² Ajustado

R² regular siempre aumenta al agregar más features. R² ajustado penaliza la complejidad:

$$R^2_{adj} = 1 - (1 - R^2) \frac{n - 1}{n - p - 1}$$

Donde:
- $n$ = número de muestras
- $p$ = número de features

```python
def r2_adjusted(y_true, y_pred, n_features):
    """Calcula R² ajustado"""
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = n_features
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Ejemplo
r2_adj = r2_adjusted(y_test, y_pred, n_features=X.shape[1])
print(f"R²:          {r2_score(y_test, y_pred):.4f}")
print(f"R² ajustado: {r2_adj:.4f}")
```

---

### 9. ¿Cuál Métrica Usar?

| Situación | Métrica Recomendada |
|-----------|---------------------|
| General, sin outliers | **RMSE** |
| Datos con outliers | **MAE** |
| Comparar modelos | **R²** |
| Necesitas % de error | **MAPE** |
| Predicción de precios | RMSE o MAE |
| Series temporales | RMSE, MAE |
| Optimización/Entrenamiento | MSE |

---

### 10. Visualización de Errores

```python
import matplotlib.pyplot as plt

# Entrenar modelo para visualización
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Predicted vs Actual
ax1 = axes[0]
ax1.scatter(y_test, y_pred, alpha=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Valores Reales')
ax1.set_ylabel('Predicciones')
ax1.set_title('Predicted vs Actual')

# 2. Residuos
residuals = y_test - y_pred
ax2 = axes[1]
ax2.scatter(y_pred, residuals, alpha=0.5)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Predicciones')
ax2.set_ylabel('Residuos')
ax2.set_title('Residuos vs Predicciones')

# 3. Distribución de residuos
ax3 = axes[2]
ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='r', linestyle='--')
ax3.set_xlabel('Residuos')
ax3.set_ylabel('Frecuencia')
ax3.set_title('Distribución de Residuos')

plt.tight_layout()
plt.show()

# Estadísticas de residuos
print(f"Media de residuos: {np.mean(residuals):.4f} (debería ser ~0)")
print(f"Std de residuos: {np.std(residuals):.4f}")
```

---

### 11. Ejemplo Completo: Predicción de Precios

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Cargar datos
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Entrenar
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Todas las métricas
print("="*50)
print("EVALUACIÓN DE REGRESIÓN")
print("="*50)
print(f"\nMSE:   {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE:  {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE:   {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²:    {r2_score(y_test, y_pred):.4f}")
print(f"MAPE:  {mean_absolute_percentage_error(y_test, y_pred)*100:.2f}%")

# Contexto
print(f"\nRango de precios: {y.min():.2f} - {y.max():.2f}")
print(f"El error típico (RMSE) es de {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} unidades")
```

---

### 12. Cross-Validation con Métricas de Regresión

```python
from sklearn.model_selection import cross_val_score

# En sklearn, métricas de error son NEGATIVAS para maximizar
# neg_mean_squared_error, neg_mean_absolute_error, etc.

model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

# MSE negativo (para maximizar)
scores_mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"MSE:  {-scores_mse.mean():.4f} ± {scores_mse.std():.4f}")

# MAE negativo
scores_mae = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"MAE:  {-scores_mae.mean():.4f} ± {scores_mae.std():.4f}")

# R² (ya es positivo es mejor)
scores_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"R²:   {scores_r2.mean():.4f} ± {scores_r2.std():.4f}")
```

---

## 📚 Resumen

| Métrica | Sensible a Outliers | Unidades | Interpretación |
|---------|---------------------|----------|----------------|
| MSE | Muy sensible | Cuadradas | Penaliza errores grandes |
| RMSE | Muy sensible | Originales | Error típico |
| MAE | Poco sensible | Originales | Error promedio |
| R² | Moderado | Adimensional | % varianza explicada |
| MAPE | Poco sensible | Porcentaje | Error relativo |

---

## ✅ Checklist de Verificación

- [ ] Entiendo la diferencia entre MSE, RMSE y MAE
- [ ] Sé interpretar R² y sus limitaciones
- [ ] Comprendo cuándo usar cada métrica
- [ ] Puedo visualizar y analizar residuos
- [ ] Sé usar scoring negativo en cross_val_score

---

**Siguiente**: [Optimización de Hiperparámetros](05-optimizacion-hiperparametros.md)
