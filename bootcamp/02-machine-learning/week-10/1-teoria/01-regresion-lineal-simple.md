# Regresión Lineal Simple

## 🎯 Objetivos

- Comprender qué es la regresión lineal simple
- Entender la ecuación de la recta y sus componentes
- Interpretar coeficientes (intercepto y pendiente)
- Conocer el método de mínimos cuadrados

## 📖 ¿Qué es la Regresión Lineal?

La **regresión lineal** es el algoritmo de Machine Learning más fundamental. Modela la relación entre una variable dependiente (target) y una o más variables independientes (features) mediante una línea recta.

### Regresión Lineal Simple

Cuando tenemos **una sola feature**, hablamos de regresión lineal simple:

$$\hat{y} = \beta_0 + \beta_1 x$$

Donde:

- $\hat{y}$: valor predicho
- $\beta_0$: intercepto (valor de y cuando x = 0)
- $\beta_1$: pendiente (cambio en y por cada unidad de x)
- $x$: feature de entrada

![Regresión Lineal Simple](../0-assets/01-regresion-lineal.svg)

## 📊 Interpretación de Coeficientes

### Intercepto (β₀)

El **intercepto** representa el valor de $y$ cuando $x = 0$.

**Ejemplo**: Si predecimos el precio de una casa basándonos en metros cuadrados:

- $\beta_0 = 50,000$ significa que una casa de 0 m² tendría un precio base de $50,000
- Aunque no tiene sentido práctico, representa el "precio base"

### Pendiente (β₁)

La **pendiente** indica cuánto cambia $y$ por cada unidad de cambio en $x$.

**Ejemplo**:

- $\beta_1 = 1,500$ significa que por cada m² adicional, el precio aumenta $1,500

```python
# Si β₀ = 50,000 y β₁ = 1,500
# Casa de 100 m²:
precio = 50000 + 1500 * 100  # = $200,000
```

## 🔢 Método de Mínimos Cuadrados (OLS)

### ¿Cómo encontramos la mejor línea?

Buscamos los valores de $\beta_0$ y $\beta_1$ que **minimicen el error** entre los valores reales y predichos.

### Función de Costo (RSS - Residual Sum of Squares)

$$J(\beta_0, \beta_1) = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i)^2$$

### Solución Analítica

Para regresión lineal simple, existe una solución exacta:

$$\beta_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

$$\beta_0 = \bar{y} - \beta_1\bar{x}$$

Donde $\bar{x}$ y $\bar{y}$ son las medias de x e y.

## 💻 Implementación con Scikit-learn

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Datos de ejemplo
X = np.array([[100], [150], [200], [250], [300]])  # metros cuadrados
y = np.array([200000, 275000, 350000, 425000, 500000])  # precios

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Ver coeficientes
print(f'Intercepto (β₀): {modelo.intercept_}')
print(f'Pendiente (β₁): {modelo.coef_[0]}')

# Predecir
casa_nueva = np.array([[175]])
precio_predicho = modelo.predict(casa_nueva)
print(f'Precio predicho para 175 m²: ${precio_predicho[0]:,.0f}')
```

**Salida esperada**:

```
Intercepto (β₀): 50000.0
Pendiente (β₁): 1500.0
Precio predicho para 175 m²: $312,500
```

## 📈 Métricas de Evaluación

### R² (Coeficiente de Determinación)

Mide qué proporción de la varianza en $y$ es explicada por el modelo.

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2} = 1 - \frac{RSS}{TSS}$$

**Interpretación**:

- $R^2 = 1$: Predicción perfecta
- $R^2 = 0$: El modelo no explica nada
- $R^2 = 0.85$: El modelo explica el 85% de la varianza

```python
from sklearn.metrics import r2_score

r2 = modelo.score(X, y)
# o
r2 = r2_score(y, modelo.predict(X))
print(f'R²: {r2:.4f}')
```

### Error Cuadrático Medio (MSE)

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### Raíz del Error Cuadrático Medio (RMSE)

$$RMSE = \sqrt{MSE}$$

```python
from sklearn.metrics import mean_squared_error
import numpy as np

y_pred = modelo.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f'MSE: {mse:,.0f}')
print(f'RMSE: {rmse:,.0f}')
```

### Error Absoluto Medio (MAE)

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y, y_pred)
print(f'MAE: {mae:,.0f}')
```

## ⚠️ Supuestos de la Regresión Lineal

Para que los resultados sean válidos, se asumen:

1. **Linealidad**: La relación entre X e y es lineal
2. **Independencia**: Los errores son independientes entre sí
3. **Homocedasticidad**: Varianza constante de los errores
4. **Normalidad**: Los errores siguen distribución normal

## 🎯 Cuándo Usar Regresión Lineal

✅ **Usar cuando**:

- Relación aproximadamente lineal entre variables
- Variable target es continua
- Necesitas interpretabilidad

❌ **No usar cuando**:

- Relación claramente no lineal
- Variable target es categórica (usar clasificación)
- Hay muchos outliers

## ✅ Checklist de Verificación

- [ ] Entiendo la ecuación y = β₀ + β₁x
- [ ] Puedo interpretar intercepto y pendiente
- [ ] Sé calcular e interpretar R², MSE, RMSE, MAE
- [ ] Puedo implementar LinearRegression con sklearn
- [ ] Conozco los supuestos del modelo

## 🔗 Recursos Adicionales

- [Sklearn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=7ArmBVF2dCs)
