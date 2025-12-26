# 📈 Ejercicio 03: Métricas de Regresión

## 🎯 Objetivo

Comprender y aplicar las métricas de regresión: MSE, RMSE, MAE, R² y analizar residuos.

---

## 📋 Descripción

En este ejercicio aprenderás a:

1. Calcular MSE, RMSE, MAE y R²
2. Entender cuándo usar cada métrica
3. Visualizar y analizar residuos
4. Comparar modelos de regresión

---

## 📁 Archivos

- `starter/main.py` - Código inicial para descomentar
- `solution/main.py` - Solución completa

---

## 🔨 Pasos

### Paso 1: Preparar Datos de Regresión

Usamos el dataset California Housing para predicción de precios.

```python
from sklearn.datasets import fetch_california_housing
X, y = fetch_california_housing(return_X_y=True)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: MSE y RMSE

Calculamos el error cuadrático medio y su raíz.

```python
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 2.

---

### Paso 3: MAE

Calculamos el error absoluto medio, más robusto a outliers.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 3.

---

### Paso 4: R² (Coeficiente de Determinación)

Calculamos la proporción de varianza explicada.

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
# 1.0 = perfecto, 0.0 = igual que predecir la media
```

**Abre `starter/main.py`** y descomenta la sección del Paso 4.

---

### Paso 5: Análisis de Residuos

Visualizamos los residuos para detectar patrones.

```python
residuals = y_test - y_pred
# Gráfico de residuos vs predicciones
# Histograma de residuos
```

**Abre `starter/main.py`** y descomenta la sección del Paso 5.

---

### Paso 6: Comparar Modelos

Evaluamos múltiples modelos de regresión.

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForest': RandomForestRegressor()
}
```

**Abre `starter/main.py`** y descomenta la sección del Paso 6.

---

### Paso 7: Sensibilidad a Outliers

Comparamos MSE vs MAE con datos con outliers.

```python
# Agregar outliers y ver cómo cambian las métricas
```

**Abre `starter/main.py`** y descomenta la sección del Paso 7.

---

## ✅ Criterios de Evaluación

| Criterio                            | Puntos |
| ----------------------------------- | ------ |
| MSE y RMSE calculados correctamente | 2      |
| MAE calculado correctamente         | 1      |
| R² calculado e interpretado         | 2      |
| Análisis de residuos completo       | 2      |
| Comparación de modelos              | 2      |
| Análisis de sensibilidad a outliers | 1      |
| **Total**                           | **10** |

---

## 🔗 Recursos

- [Regression metrics scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
