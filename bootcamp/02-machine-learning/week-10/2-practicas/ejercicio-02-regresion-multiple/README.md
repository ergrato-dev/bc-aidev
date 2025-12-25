# Ejercicio 02: Regresión Lineal Múltiple

## 🎯 Objetivo

Implementar regresión lineal múltiple usando varias características para predecir precios.

## 📋 Conceptos Cubiertos

- Regresión con múltiples features
- Importancia del escalado de features
- Interpretación de coeficientes múltiples
- Comparación con regresión simple

## 🛠️ Requisitos

```bash
pip install numpy pandas matplotlib scikit-learn
```

## 📝 Instrucciones

Sigue los pasos en orden, descomentando el código en `starter/main.py`.

---

### Paso 1: Crear Dataset Multivariable

Creamos datos con 3 características que influyen en el precio:

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 200

data = {
    'area': np.random.uniform(50, 250, n_samples),
    'habitaciones': np.random.randint(1, 6, n_samples),
    'antiguedad': np.random.uniform(0, 50, n_samples)
}
df = pd.DataFrame(data)

# Precio: depende de las 3 features
df['precio'] = (
    25000 +                           # Base
    1200 * df['area'] +               # +$1200 por m²
    15000 * df['habitaciones'] +      # +$15000 por habitación
    -800 * df['antiguedad'] +         # -$800 por año de antigüedad
    np.random.normal(0, 20000, n_samples)
)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Explorar Correlaciones

```python
print(df.corr()['precio'].sort_values(ascending=False))
```

Esto muestra qué tan correlacionada está cada feature con el precio.

**Descomenta** la sección del Paso 2.

---

### Paso 3: Preparar Features y Target

```python
X = df[['area', 'habitaciones', 'antiguedad']]
y = df['precio']
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Escalar Features (StandardScaler)

**Importante**: Para comparar coeficientes entre features con diferentes escalas (m² vs años), debemos estandarizar:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Entrenar y Comparar Modelos

Entrenamos con y sin escalado para ver la diferencia:

```python
from sklearn.linear_model import LinearRegression

# Modelo sin escalar
model_raw = LinearRegression()
model_raw.fit(X_train, y_train)

# Modelo con datos escalados
model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Interpretar Coeficientes

```python
# Coeficientes sin escalar (unidades originales)
print("Coeficientes (datos originales):")
for name, coef in zip(X.columns, model_raw.coef_):
    print(f"  {name}: {coef:,.2f}")

# Coeficientes escalados (comparables entre sí)
print("\nCoeficientes (datos escalados - importancia relativa):")
for name, coef in zip(X.columns, model_scaled.coef_):
    print(f"  {name}: {coef:,.2f}")
```

**Con datos escalados**: el coeficiente más grande indica la feature más importante.

**Descomenta** la sección del Paso 6.

---

### Paso 7: Evaluar Modelo

```python
from sklearn.metrics import r2_score, mean_absolute_error

y_pred = model_raw.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R²: {r2:.4f}")
print(f"MAE: ${mae:,.2f}")
```

**MAE** (Mean Absolute Error) = error promedio en dólares.

**Descomenta** la sección del Paso 7.

---

### Paso 8: Comparar con Regresión Simple

¿Cuánto mejora usar 3 features vs solo 1?

```python
# Solo área
model_simple = LinearRegression()
model_simple.fit(X_train[['area']], y_train)
r2_simple = r2_score(y_test, model_simple.predict(X_test[['area']]))

print(f"R² con 1 feature (área): {r2_simple:.4f}")
print(f"R² con 3 features: {r2:.4f}")
print(f"Mejora: +{(r2 - r2_simple)*100:.1f} puntos porcentuales")
```

**Descomenta** la sección del Paso 8.

---

## ✅ Resultado Esperado

```
Coeficientes (datos originales):
  area: ~1,200
  habitaciones: ~15,000
  antiguedad: ~-800

R² con 1 feature: ~0.65
R² con 3 features: ~0.90
Mejora: ~+25 puntos porcentuales
```

---

## 🔍 Ejercicio Extra

1. Añade una 4ta feature (`distancia_centro`) y observa si mejora R²
2. Calcula el VIF para detectar multicolinealidad
3. Usa `model.predict([[150, 3, 10]])` para predecir precio de casa específica

---

## 📚 Recursos

- [StandardScaler - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Multicolinealidad - Wikipedia](https://es.wikipedia.org/wiki/Multicolinealidad)
