# 📖 Glosario - Semana 10: Regresión Lineal y Logística

## A

### Accuracy (Exactitud)

Proporción de predicciones correctas sobre el total. En clasificación: (TP + TN) / (TP + TN + FP + FN).

### Alpha (α)

Hiperparámetro de regularización en Ridge y Lasso. Valores mayores = más regularización.

## B

### Bias (Sesgo)

En regresión lineal, es el término independiente (intercepto β₀). También se refiere al error sistemático de un modelo.

### Binary Classification (Clasificación Binaria)

Problema de clasificación con solo dos clases posibles (0/1, Sí/No, Positivo/Negativo).

## C

### Coefficient (Coeficiente)

En regresión lineal, los valores β que multiplican a cada feature. Representan el cambio en y por unidad de cambio en x.

### Confusion Matrix (Matriz de Confusión)

Tabla que muestra TP, TN, FP, FN para evaluar clasificación:

```
              Predicho
              0    1
Real    0   [TN] [FP]
        1   [FN] [TP]
```

### Cross-Validation (Validación Cruzada)

Técnica para evaluar modelos dividiendo datos en k folds, entrenando en k-1 y validando en 1, rotando.

## D

### Decision Boundary (Frontera de Decisión)

En regresión logística, la línea/superficie donde P(y=1) = 0.5. Separa las clases.

## E

### Elastic Net

Regularización que combina L1 (Lasso) y L2 (Ridge):
$$J = MSE + \alpha \cdot \rho \cdot ||w||_1 + \alpha \cdot (1-\rho) \cdot ||w||_2^2$$

## F

### F1-Score

Media armónica de precision y recall:
$$F1 = 2 \cdot \frac{precision \cdot recall}{precision + recall}$$

### False Negative (FN)

Predicción de clase 0 cuando la clase real es 1 (error tipo II).

### False Positive (FP)

Predicción de clase 1 cuando la clase real es 0 (error tipo I).

### Feature (Característica)

Variable independiente usada para hacer predicciones. En regresión: las x.

### Feature Selection

Proceso de seleccionar las features más relevantes. Lasso lo hace automáticamente.

## G

### Gradient Descent (Descenso de Gradiente)

Algoritmo de optimización que minimiza la función de costo iterativamente siguiendo el gradiente negativo.

## H

### Hyperparameter (Hiperparámetro)

Parámetro que no se aprende del datos sino que se define antes del entrenamiento (ej: α en Ridge).

## I

### Intercept (Intercepto)

Término independiente β₀ en regresión lineal. Valor de y cuando todas las x son 0.

## L

### L1 Regularization (Regularización L1)

Penalización basada en la suma de valores absolutos de coeficientes. Usada en Lasso.
$$Penalty = \lambda \sum |w_i|$$

### L2 Regularization (Regularización L2)

Penalización basada en la suma de cuadrados de coeficientes. Usada en Ridge.
$$Penalty = \lambda \sum w_i^2$$

### Lasso (Least Absolute Shrinkage and Selection Operator)

Regresión lineal con regularización L1. Puede hacer coeficientes exactamente 0.

### Learning Rate (Tasa de Aprendizaje)

En gradient descent, el tamaño del paso en cada iteración.

### Linear Regression (Regresión Lineal)

Modelo que asume relación lineal entre features y target:
$$y = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n$$

### Log-Loss (Binary Cross-Entropy)

Función de costo para regresión logística:
$$J = -\frac{1}{n}\sum[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

### Logistic Regression (Regresión Logística)

Modelo de clasificación que predice probabilidades usando función sigmoide.

## M

### MAE (Mean Absolute Error)

Error absoluto promedio:
$$MAE = \frac{1}{n}\sum|y_i - \hat{y}_i|$$

### MSE (Mean Squared Error)

Error cuadrático medio:
$$MSE = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$$

### Multicollinearity (Multicolinealidad)

Correlación alta entre features. Causa coeficientes inestables en regresión lineal.

### Multiple Regression (Regresión Múltiple)

Regresión lineal con más de una feature independiente.

## N

### Normalization (Normalización)

Escalar datos a un rango específico (típicamente [0,1]).

## O

### OLS (Ordinary Least Squares)

Método de mínimos cuadrados ordinarios para ajustar regresión lineal minimizando MSE.

### Overfitting (Sobreajuste)

Modelo que memoriza datos de entrenamiento y no generaliza bien a datos nuevos.

## P

### Precision (Precisión)

De las predicciones positivas, cuántas son correctas:
$$Precision = \frac{TP}{TP + FP}$$

### Prediction (Predicción)

Valor estimado por el modelo (ŷ).

### Probability (Probabilidad)

En regresión logística, P(y=1|x) devuelto por predict_proba().

## R

### R² (R-squared, Coeficiente de Determinación)

Proporción de varianza explicada por el modelo:
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

### Recall (Sensibilidad, TPR)

De los positivos reales, cuántos se detectaron:
$$Recall = \frac{TP}{TP + FN}$$

### Regularization (Regularización)

Técnica para prevenir overfitting añadiendo penalización a coeficientes grandes.

### Residual (Residuo)

Diferencia entre valor real y predicho: $e_i = y_i - \hat{y}_i$

### Ridge Regression

Regresión lineal con regularización L2. Reduce coeficientes pero no los hace 0.

### RMSE (Root Mean Squared Error)

Raíz del error cuadrático medio:
$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$$

## S

### Sigmoid Function (Función Sigmoide)

Función que mapea cualquier valor a [0,1]:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Simple Linear Regression (Regresión Lineal Simple)

Regresión con una sola feature: $y = \beta_0 + \beta_1 x$

### Slope (Pendiente)

Coeficiente β₁ en regresión simple. Indica cambio en y por unidad de x.

### StandardScaler

Transformación que centra datos (media=0) y escala (desviación=1):
$$z = \frac{x - \mu}{\sigma}$$

## T

### Target (Variable Objetivo)

Variable que queremos predecir (y).

### Threshold (Umbral)

En clasificación, valor de probabilidad para decidir clase. Default: 0.5.

### True Negative (TN)

Predicción correcta de clase 0.

### True Positive (TP)

Predicción correcta de clase 1.

## U

### Underfitting (Subajuste)

Modelo demasiado simple que no captura patrones en los datos.

## V

### VIF (Variance Inflation Factor)

Métrica para detectar multicolinealidad. VIF > 5 indica problemas:
$$VIF_i = \frac{1}{1 - R_i^2}$$

## W

### Weight (Peso)

Sinónimo de coeficiente en el contexto de redes neuronales.

---

## 📊 Fórmulas Clave Resumidas

| Métrica   | Fórmula                           |
| --------- | --------------------------------- |
| MSE       | $\frac{1}{n}\sum(y - \hat{y})^2$  |
| RMSE      | $\sqrt{MSE}$                      |
| MAE       | $\frac{1}{n}\sum\|y - \hat{y}\|$  |
| R²        | $1 - \frac{SS_{res}}{SS_{tot}}$   |
| Sigmoid   | $\frac{1}{1 + e^{-z}}$            |
| Precision | $\frac{TP}{TP + FP}$              |
| Recall    | $\frac{TP}{TP + FN}$              |
| F1        | $\frac{2 \cdot P \cdot R}{P + R}$ |
