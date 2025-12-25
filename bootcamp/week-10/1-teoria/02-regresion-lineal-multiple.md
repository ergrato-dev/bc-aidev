# Regresión Lineal Múltiple

## 🎯 Objetivos

- Extender regresión lineal a múltiples features
- Comprender la ecuación matricial
- Interpretar coeficientes en contexto multivariable
- Identificar problemas de multicolinealidad

## 📖 De Simple a Múltiple

Cuando tenemos **múltiples features** (variables independientes), usamos regresión lineal múltiple:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n$$

O en forma matricial:

$$\hat{y} = X\beta$$

Donde:

- $X$: matriz de features (n samples × p features)
- $\beta$: vector de coeficientes

![Regresión Lineal Múltiple](../0-assets/03-regresion-multiple.svg)

## 📊 Ejemplo: Predicción de Precios de Casas

Supongamos que queremos predecir el precio de una casa usando:

- $x_1$: área en m²
- $x_2$: número de habitaciones
- $x_3$: edad de la casa (años)

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Datos de ejemplo
data = {
    'area': [100, 150, 200, 120, 180, 220, 90, 160],
    'habitaciones': [2, 3, 4, 2, 3, 4, 2, 3],
    'edad': [10, 5, 2, 15, 8, 1, 20, 12],
    'precio': [200000, 300000, 450000, 180000, 350000, 500000, 150000, 280000]
}
df = pd.DataFrame(data)

# Features y target
X = df[['area', 'habitaciones', 'edad']]
y = df['precio']

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Coeficientes
print('Intercepto (β₀):', f'{modelo.intercept_:,.0f}')
print('\nCoeficientes:')
for feature, coef in zip(X.columns, modelo.coef_):
    print(f'  {feature}: {coef:,.0f}')
```

**Salida**:

```
Intercepto (β₀): -50,000
Coeficientes:
  area: 1,500
  habitaciones: 30,000
  edad: -2,000
```

## 🔍 Interpretación de Coeficientes

### Interpretación Ceteris Paribus

Cada coeficiente se interpreta **manteniendo las demás variables constantes**:

| Coeficiente       | Valor   | Interpretación                                           |
| ----------------- | ------- | -------------------------------------------------------- |
| β₁ (área)         | +1,500  | Por cada m² adicional, el precio aumenta $1,500          |
| β₂ (habitaciones) | +30,000 | Por cada habitación adicional, el precio aumenta $30,000 |
| β₃ (edad)         | -2,000  | Por cada año de antigüedad, el precio disminuye $2,000   |

### Predicción

```python
# Predecir precio de casa: 150m², 3 hab, 5 años
casa = np.array([[150, 3, 5]])
precio = modelo.predict(casa)
print(f'Precio predicho: ${precio[0]:,.0f}')

# Cálculo manual:
# -50000 + 1500*150 + 30000*3 - 2000*5 = $255,000
```

## ⚠️ Multicolinealidad

### ¿Qué es?

La **multicolinealidad** ocurre cuando las features están altamente correlacionadas entre sí. Esto causa:

- Coeficientes inestables
- Interpretación difícil
- Alta varianza en estimaciones

### Detectar Multicolinealidad

#### 1. Matriz de Correlación

```python
import seaborn as sns
import matplotlib.pyplot as plt

correlacion = df[['area', 'habitaciones', 'edad']].corr()
print(correlacion)

# Visualizar
sns.heatmap(correlacion, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()
```

#### 2. VIF (Variance Inflation Factor)

$$VIF_j = \frac{1}{1 - R_j^2}$$

Donde $R_j^2$ es el R² de regresionar $x_j$ contra las demás features.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calcular VIF para cada feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
```

**Interpretación VIF**:

- VIF = 1: Sin correlación
- VIF > 5: Moderada multicolinealidad
- VIF > 10: Alta multicolinealidad (problemático)

### Soluciones a Multicolinealidad

1. **Eliminar features** correlacionadas
2. **PCA** (Análisis de Componentes Principales)
3. **Regularización** (Ridge, Lasso)

## 📐 Escalado de Features

### ¿Por qué escalar?

Los coeficientes dependen de la escala de las features. Para comparar importancia, debemos escalar:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

modelo_scaled = LinearRegression()
modelo_scaled.fit(X_scaled, y)

print('Coeficientes estandarizados:')
for feature, coef in zip(X.columns, modelo_scaled.coef_):
    print(f'  {feature}: {coef:,.0f}')
```

**Interpretación**: Coeficientes estandarizados muestran el impacto relativo de cada feature.

## 📈 R² Ajustado

El R² regular siempre aumenta al agregar features (incluso irrelevantes). El **R² ajustado** penaliza agregar features:

$$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

Donde:

- $n$: número de samples
- $p$: número de features

```python
from sklearn.metrics import r2_score

y_pred = modelo.predict(X)
r2 = r2_score(y, y_pred)

n = len(y)
p = X.shape[1]
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f'R²: {r2:.4f}')
print(f'R² ajustado: {r2_adj:.4f}')
```

## 💻 Pipeline Completo

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline: escalar + regresión
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Entrenar
pipeline.fit(X_train, y_train)

# Evaluar
y_pred = pipeline.predict(X_test)
print(f'R²: {r2_score(y_test, y_pred):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}')
```

## ✅ Checklist de Verificación

- [ ] Entiendo la diferencia entre regresión simple y múltiple
- [ ] Puedo interpretar coeficientes en contexto multivariable
- [ ] Sé detectar multicolinealidad (correlación, VIF)
- [ ] Entiendo cuándo y cómo escalar features
- [ ] Conozco la diferencia entre R² y R² ajustado

## 🔗 Recursos Adicionales

- [Sklearn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [VIF en Statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html)
