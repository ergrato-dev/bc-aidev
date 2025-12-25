# 🔄 Ejercicio 01: Cross-Validation en Práctica

## 🎯 Objetivo

Dominar las diferentes estrategias de Cross-Validation y comparar modelos de forma robusta.

---

## 📋 Descripción

En este ejercicio aprenderás a:
1. Implementar K-Fold y Stratified K-Fold
2. Usar `cross_val_score` y `cross_validate`
3. Comparar múltiples modelos con CV
4. Entender el impacto de K en los resultados

---

## 📁 Archivos

- `starter/main.py` - Código inicial para descomentar
- `solution/main.py` - Solución completa

---

## 🔨 Pasos

### Paso 1: Setup y Datos

Cargamos un dataset de clasificación y exploramos su estructura.

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
print(f"Muestras: {data.data.shape[0]}")
print(f"Features: {data.data.shape[1]}")
print(f"Distribución de clases: {pd.Series(data.target).value_counts().to_dict()}")
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: K-Fold Manual

Implementamos K-Fold manualmente para entender su funcionamiento.

```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    # Dividir datos
    # Entrenar modelo
    # Evaluar en validación
    pass
```

**Abre `starter/main.py`** y descomenta la sección del Paso 2.

---

### Paso 3: cross_val_score

Usamos la función simplificada para CV.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Abre `starter/main.py`** y descomenta la sección del Paso 3.

---

### Paso 4: Stratified K-Fold

Comparamos K-Fold regular vs Stratified K-Fold.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 4.

---

### Paso 5: Comparar Múltiples Modelos

Evaluamos varios modelos con CV para elegir el mejor.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Abre `starter/main.py`** y descomenta la sección del Paso 5.

---

### Paso 6: cross_validate (Múltiples Métricas)

Obtenemos múltiples métricas en una sola ejecución.

```python
from sklearn.model_selection import cross_validate

results = cross_validate(
    model, X, y, cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1'],
    return_train_score=True
)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 6.

---

### Paso 7: Impacto de K

Evaluamos cómo cambian los resultados con diferentes valores de K.

```python
for k in [3, 5, 10, 20]:
    scores = cross_val_score(model, X, y, cv=k)
    print(f"K={k}: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Abre `starter/main.py`** y descomenta la sección del Paso 7.

---

## ✅ Criterios de Evaluación

| Criterio | Puntos |
|----------|--------|
| K-Fold manual implementado correctamente | 2 |
| cross_val_score funcionando | 2 |
| Comparación de modelos completa | 3 |
| Análisis de impacto de K | 2 |
| Código limpio y comentado | 1 |
| **Total** | **10** |

---

## 🔗 Recursos

- [Cross-validation scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)
