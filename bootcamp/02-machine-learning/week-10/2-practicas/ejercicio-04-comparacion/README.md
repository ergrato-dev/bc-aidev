# Ejercicio 04: Comparación de Modelos

## 🎯 Objetivo

Comparar regresión lineal vs logística y aplicar regularización (Ridge, Lasso).

## 📋 Conceptos Cubiertos

- Cuándo usar regresión vs clasificación
- Regularización L1 (Lasso) y L2 (Ridge)
- Selección de hiperparámetro α con cross-validation
- Comparación de métricas entre modelos

## 🛠️ Requisitos

```bash
pip install numpy pandas matplotlib scikit-learn
```

## 📝 Instrucciones

Sigue los pasos en orden, descomentando el código en `starter/main.py`.

---

### Paso 1: Crear Dataset con Multicolinealidad

Dataset donde algunas features están correlacionadas (problemático para regresión lineal):

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 300

# Feature base
x1 = np.random.uniform(0, 100, n)

# Features correlacionadas con x1
x2 = x1 * 0.8 + np.random.normal(0, 10, n)  # Muy correlacionada
x3 = np.random.uniform(0, 50, n)            # Independiente
x4 = x1 * 0.5 + x3 * 0.3 + np.random.normal(0, 5, n)  # Mixta

# Target
y = 100 + 2*x1 + 0.5*x3 + np.random.normal(0, 20, n)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Detectar Multicolinealidad

```python
df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})
print("Matriz de correlación:")
print(df.corr().round(2))
```

Correlación > 0.8 indica multicolinealidad.

**Descomenta** la sección del Paso 2.

---

### Paso 3: Entrenar Modelos con y sin Regularización

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Modelos
lr = LinearRegression().fit(X_train_s, y_train)
ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
lasso = Lasso(alpha=1.0).fit(X_train_s, y_train)
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Comparar Coeficientes

```python
print("Coeficientes por modelo:")
print(f"{'Feature':<10} {'LinReg':>10} {'Ridge':>10} {'Lasso':>10}")
for i, col in enumerate(['x1', 'x2', 'x3', 'x4']):
    print(f"{col:<10} {lr.coef_[i]:>10.2f} {ridge.coef_[i]:>10.2f} {lasso.coef_[i]:>10.2f}")
```

**Lasso** tiende a hacer coeficientes exactamente 0 (feature selection).

**Descomenta** la sección del Paso 4.

---

### Paso 5: Comparar R² en Test

```python
from sklearn.metrics import r2_score

models = {'LinearRegression': lr, 'Ridge': ridge, 'Lasso': lasso}

for name, model in models.items():
    y_pred = model.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: R² = {r2:.4f}")
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Encontrar Mejor α con Cross-Validation

```python
from sklearn.linear_model import RidgeCV, LassoCV

alphas = [0.001, 0.01, 0.1, 1, 10, 100]

ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_train_s, y_train)
lasso_cv = LassoCV(alphas=alphas, cv=5).fit(X_train_s, y_train)

print(f"Mejor α Ridge: {ridge_cv.alpha_}")
print(f"Mejor α Lasso: {lasso_cv.alpha_}")
```

**Descomenta** la sección del Paso 6.

---

### Paso 7: Visualizar Efecto de α en Coeficientes

```python
import matplotlib.pyplot as plt

alphas_plot = np.logspace(-3, 3, 50)
ridge_coefs = []

for a in alphas_plot:
    ridge_temp = Ridge(alpha=a).fit(X_train_s, y_train)
    ridge_coefs.append(ridge_temp.coef_)

ridge_coefs = np.array(ridge_coefs)

plt.figure(figsize=(10, 6))
for i in range(4):
    plt.plot(alphas_plot, ridge_coefs[:, i], label=f'x{i+1}')
plt.xscale('log')
plt.xlabel('Alpha (regularización)')
plt.ylabel('Coeficiente')
plt.title('Ridge: Coeficientes vs Alpha')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ridge_coefs.png', dpi=150)
plt.show()
```

**Descomenta** la sección del Paso 7.

---

## ✅ Resultado Esperado

```
Coeficientes:
Feature    LinReg      Ridge      Lasso
x1          XX.XX      XX.XX      XX.XX
x2          XX.XX      XX.XX       0.00  ← Lasso elimina
x3          XX.XX      XX.XX      XX.XX
x4          XX.XX      XX.XX       0.00  ← Lasso elimina
```

---

## 📚 Recursos

- [Ridge Regression - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Lasso Regression - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [Regularización explicada](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)
