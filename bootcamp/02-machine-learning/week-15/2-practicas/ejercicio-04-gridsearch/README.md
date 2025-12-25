# ⚙️ Ejercicio 04: GridSearchCV y Optimización

## 🎯 Objetivo

Dominar las técnicas de optimización de hiperparámetros usando GridSearchCV y RandomizedSearchCV.

---

## 📋 Descripción

En este ejercicio aprenderás a:
1. Usar GridSearchCV para búsqueda exhaustiva
2. Usar RandomizedSearchCV para búsqueda eficiente
3. Optimizar pipelines completos
4. Implementar Nested Cross-Validation

---

## 📁 Archivos

- `starter/main.py` - Código inicial para descomentar
- `solution/main.py` - Solución completa

---

## 🔨 Pasos

### Paso 1: GridSearchCV Básico

Búsqueda exhaustiva de hiperparámetros para RandomForest.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Analizar Resultados

Exploramos los resultados de la búsqueda.

```python
print(f"Mejores hiperparámetros: {grid_search.best_params_}")
print(f"Mejor score: {grid_search.best_score_}")

# Acceder a todos los resultados
results = pd.DataFrame(grid_search.cv_results_)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 2.

---

### Paso 3: RandomizedSearchCV

Búsqueda aleatoria para espacios grandes de hiperparámetros.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 50)
}

random_search = RandomizedSearchCV(model, param_distributions, n_iter=20)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 3.

---

### Paso 4: Pipeline con GridSearch

Optimizamos un pipeline completo (preprocesamiento + modelo).

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [5, 10]
}
```

**Abre `starter/main.py`** y descomenta la sección del Paso 4.

---

### Paso 5: Nested Cross-Validation

Evaluación honesta del proceso de optimización.

```python
from sklearn.model_selection import cross_val_score

# CV interno: GridSearch
# CV externo: Evaluación
nested_scores = cross_val_score(grid_search, X, y, cv=5)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 5.

---

### Paso 6: Diferentes Scorers

Optimizamos para diferentes métricas.

```python
# Por defecto optimiza accuracy
# Podemos cambiar a f1, precision, recall, roc_auc
grid_f1 = GridSearchCV(model, param_grid, cv=5, scoring='f1')
grid_auc = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
```

**Abre `starter/main.py`** y descomenta la sección del Paso 6.

---

## ✅ Criterios de Evaluación

| Criterio | Puntos |
|----------|--------|
| GridSearchCV implementado correctamente | 2 |
| Análisis de resultados completo | 2 |
| RandomizedSearchCV funcionando | 2 |
| Pipeline optimizado | 2 |
| Nested CV implementado | 2 |
| **Total** | **10** |

---

## 🔗 Recursos

- [GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Tuning Hyperparameters](https://scikit-learn.org/stable/modules/grid_search.html)
