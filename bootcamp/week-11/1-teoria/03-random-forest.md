# Random Forest: Ensemble de Árboles

## 🎯 Objetivos

- Entender qué es un ensemble y por qué funciona
- Comprender Bagging y Bootstrap Sampling
- Dominar Random Forest y sus componentes
- Conocer OOB Score para validación

## 📋 Contenido

### 1. El Problema de los Árboles Individuales

Un árbol de decisión tiene:

- ✅ Alta varianza (sensible a cambios en datos)
- ✅ Tendencia al overfitting
- ✅ Decisiones inestables

**Solución**: Combinar múltiples árboles en un **ensemble**.

![Random Forest](../0-assets/03-random-forest.svg)

### 2. Bagging (Bootstrap Aggregating)

Técnica propuesta por Leo Breiman (1996).

#### Proceso

1. **Bootstrap Sampling**: Crear N subconjuntos aleatorios con reemplazo
2. **Entrenar**: Un modelo en cada subconjunto
3. **Agregar**: Combinar predicciones (votación o promedio)

```python
# Bootstrap sampling conceptual
import numpy as np

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]
```

#### ¿Por qué funciona?

- Cada árbol ve datos ligeramente diferentes
- Reduce varianza al promediar predicciones
- Errores individuales se cancelan

### 3. Random Forest

**Random Forest = Bagging + Feature Randomness**

#### Componentes Clave

| Componente           | Descripción                                 |
| -------------------- | ------------------------------------------- |
| **Bootstrap**        | Muestreo con reemplazo                      |
| **Feature Sampling** | Subconjunto aleatorio de features por split |
| **Agregación**       | Voting (clasificación) / Mean (regresión)   |

#### Feature Subsampling

En cada división, solo se consideran `max_features` features:

- Clasificación: `sqrt(n_features)` (default)
- Regresión: `n_features` (todos)

```python
# Si tenemos 16 features:
# - Clasificación: sqrt(16) = 4 features por split
# - Regresión: 16 features por split (o n_features/3)
```

### 4. Implementación con Scikit-learn

#### Clasificación

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Cargar datos
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear y entrenar Random Forest
rf = RandomForestClassifier(
    n_estimators=100,       # Número de árboles
    max_depth=None,         # Profundidad máxima
    max_features='sqrt',    # Features por split
    random_state=42,
    n_jobs=-1               # Usar todos los cores
)

rf.fit(X_train, y_train)

# Evaluar
print(f"Train accuracy: {rf.score(X_train, y_train):.4f}")
print(f"Test accuracy: {rf.score(X_test, y_test):.4f}")
```

#### Regresión

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Datos sintéticos
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    max_features=1.0,  # Usar todos los features
    random_state=42
)

rf_reg.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

y_pred = rf_reg.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")
```

### 5. Out-of-Bag (OOB) Score

Aproximadamente **37% de muestras** no se usan en cada bootstrap (out-of-bag).

$$P(no\ seleccionado) = \left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx 0.368$$

#### Usar OOB para Validación

```python
rf_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # Activar OOB
    random_state=42,
    n_jobs=-1
)

rf_oob.fit(X_train, y_train)

print(f"OOB Score: {rf_oob.oob_score_:.4f}")
print(f"Test Score: {rf_oob.score(X_test, y_test):.4f}")
```

**Ventaja**: Validación "gratuita" sin separar datos.

### 6. Feature Importance

Random Forest calcula la importancia de cada feature.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Entrenar modelo
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Importancia de features
importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True)

# Visualizar
plt.figure(figsize=(10, 6))
plt.barh(importance['feature'], importance['importance'], color='steelblue')
plt.xlabel('Importancia')
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=150)
plt.show()
```

![Feature Importance](../0-assets/05-feature-importance.svg)

### 7. Predicción con Probabilidades

```python
# Obtener probabilidades de cada clase
proba = rf.predict_proba(X_test[:5])
print("Probabilidades por clase:")
print(proba)

# Predicción = clase con mayor probabilidad
predictions = rf.predict(X_test[:5])
print(f"\nPredicciones: {predictions}")
```

### 8. Ventajas y Limitaciones

#### ✅ Ventajas

- Robusto al overfitting (vs árbol individual)
- Maneja muchos features sin selección previa
- No requiere normalización
- Feature importance incluida
- Paralelizable (`n_jobs=-1`)
- OOB para validación gratuita

#### ❌ Limitaciones

- Menos interpretable que un árbol
- Más lento en predicción
- Usa más memoria
- No extrapola (regresión)
- Puede sobreajustar con datos ruidosos

### 9. Random Forest vs Árbol Individual

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Comparación
tree = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

tree_scores = cross_val_score(tree, X, y, cv=5)
rf_scores = cross_val_score(rf, X, y, cv=5)

print(f"Decision Tree: {tree_scores.mean():.4f} ± {tree_scores.std():.4f}")
print(f"Random Forest: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
```

**Resultado típico**: Random Forest tiene menor varianza y mejor generalización.

---

## ✅ Checklist de Verificación

- [ ] Entiendo qué es Bagging y Bootstrap Sampling
- [ ] Sé por qué combinar árboles reduce varianza
- [ ] Conozco el rol del feature subsampling
- [ ] Puedo usar RandomForestClassifier y Regressor
- [ ] Sé interpretar y usar OOB Score
- [ ] Puedo extraer e interpretar feature importance

---

## 📚 Recursos

- [Random Forest - sklearn](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- [Original Paper - Breiman 2001](https://link.springer.com/article/10.1023/A:1010933404324)
- [Understanding Random Forest](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)
