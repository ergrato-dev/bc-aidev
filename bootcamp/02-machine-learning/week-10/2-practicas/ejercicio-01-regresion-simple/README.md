# Ejercicio 01: Regresión Lineal Simple

## 🎯 Objetivo

Implementar regresión lineal simple para predecir precios basados en una única característica (área).

## 📋 Conceptos Cubiertos

- Regresión lineal simple con scikit-learn
- Visualización de la línea de regresión
- Cálculo e interpretación de R²
- Coeficientes: pendiente (β₁) e intercepto (β₀)

## 🛠️ Requisitos

```bash
pip install numpy pandas matplotlib scikit-learn
```

## 📝 Instrucciones

Sigue los pasos en orden, descomentando el código en `starter/main.py`.

---

### Paso 1: Crear Datos Sintéticos

Generamos datos de área (m²) y precio con relación lineal más ruido:

```python
import numpy as np

np.random.seed(42)
area = np.random.uniform(50, 200, 100)  # 100 casas entre 50-200 m²
precio = 30000 + 1500 * area + np.random.normal(0, 15000, 100)
```

**Relación real**: precio = 30000 + 1500 × área (+ ruido)

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Preparar Datos para Scikit-learn

Scikit-learn requiere X como matriz 2D:

```python
X = area.reshape(-1, 1)  # De (100,) a (100, 1)
y = precio
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Dividir en Train/Test

Siempre separamos datos para evaluar en datos no vistos:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Entrenar el Modelo

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Intercepto (β₀): ${model.intercept_:,.2f}")
print(f"Pendiente (β₁): ${model.coef_[0]:,.2f} por m²")
```

**Interpretación**: Por cada m² adicional, el precio aumenta ~$1,500.

**Descomenta** la sección del Paso 4.

---

### Paso 5: Evaluar con R²

```python
from sklearn.metrics import r2_score

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"R² Train: {r2_train:.4f}")
print(f"R² Test: {r2_test:.4f}")
```

**R² cercano a 1** = buen ajuste. Diferencia grande train/test = overfitting.

**Descomenta** la sección del Paso 5.

---

### Paso 6: Visualizar Resultado

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.7, label='Datos reales')
plt.plot(X_test, y_pred_test, color='red', linewidth=2, label='Predicción')
plt.xlabel('Área (m²)')
plt.ylabel('Precio ($)')
plt.title('Regresión Lineal Simple: Área vs Precio')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('regresion_simple.png', dpi=150)
plt.show()
```

**Descomenta** la sección del Paso 6.

---

## ✅ Resultado Esperado

```
Intercepto (β₀): $28,XXX.XX
Pendiente (β₁): $1,4XX.XX por m²
R² Train: 0.8X
R² Test: 0.8X
```

Y un gráfico mostrando la línea de regresión ajustada a los datos.

---

## 🔍 Ejercicio Extra

1. Cambia el `random_state` y observa cómo varían los coeficientes
2. Aumenta el ruido (`15000` → `30000`) y observa cómo baja R²
3. Usa menos datos (20 en vez de 100) y observa la variabilidad

---

## 📚 Recursos

- [LinearRegression - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [train_test_split - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
