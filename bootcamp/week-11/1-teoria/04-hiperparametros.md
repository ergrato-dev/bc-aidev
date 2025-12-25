# Hiperparámetros y Tuning

## 🎯 Objetivos

- Conocer los principales hiperparámetros de árboles y Random Forest
- Entender el impacto de cada hiperparámetro
- Dominar GridSearchCV y RandomizedSearchCV
- Aplicar estrategias de tuning efectivas

## 📋 Contenido

### 1. Hiperparámetros de Decision Tree

![Overfitting en Árboles](../0-assets/04-overfitting-arboles.svg)

| Parámetro           | Descripción               | Default | Efecto        |
| ------------------- | ------------------------- | ------- | ------------- |
| `max_depth`         | Profundidad máxima        | None    | ↓ Overfitting |
| `min_samples_split` | Min muestras para dividir | 2       | ↓ Overfitting |
| `min_samples_leaf`  | Min muestras en hoja      | 1       | ↓ Overfitting |
| `max_features`      | Features por split        | None    | ↓ Varianza    |
| `max_leaf_nodes`    | Número máximo de hojas    | None    | ↓ Complejidad |
| `ccp_alpha`         | Parámetro de poda         | 0.0     | ↓ Overfitting |

### 2. Hiperparámetros de Random Forest

| Parámetro           | Descripción           | Default | Efecto                |
| ------------------- | --------------------- | ------- | --------------------- |
| `n_estimators`      | Número de árboles     | 100     | ↑ = Mejor (más lento) |
| `max_depth`         | Profundidad por árbol | None    | ↓ Overfitting         |
| `min_samples_split` | Min para dividir      | 2       | ↓ Overfitting         |
| `min_samples_leaf`  | Min en hoja           | 1       | ↓ Overfitting         |
| `max_features`      | Features por split    | 'sqrt'  | ↓ Correlación árboles |
| `bootstrap`         | Usar bootstrap        | True    | True = Random Forest  |
| `oob_score`         | Calcular OOB          | False   | Validación gratuita   |
| `n_jobs`            | Cores paralelos       | None    | -1 = Todos            |

### 3. Impacto de max_depth

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Datos sintéticos
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                          n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Probar diferentes profundidades
depths = range(1, 21)
train_scores = []
test_scores = []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

# Visualizar
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'b-o', label='Train')
plt.plot(depths, test_scores, 'r-o', label='Test')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Impacto de max_depth en Decision Tree')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('max_depth_impact.png', dpi=150)
plt.show()
```

### 4. Impacto de n_estimators

```python
from sklearn.ensemble import RandomForestClassifier

# Probar diferentes números de árboles
n_trees = [1, 5, 10, 25, 50, 100, 200, 500]
oob_scores = []
test_scores = []

for n in n_trees:
    rf = RandomForestClassifier(
        n_estimators=n,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    oob_scores.append(rf.oob_score_)
    test_scores.append(rf.score(X_test, y_test))

# Visualizar
plt.figure(figsize=(10, 6))
plt.plot(n_trees, oob_scores, 'g-o', label='OOB Score')
plt.plot(n_trees, test_scores, 'b-o', label='Test Score')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Impacto de n_estimators en Random Forest')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.savefig('n_estimators_impact.png', dpi=150)
plt.show()
```

### 5. GridSearchCV

Búsqueda exhaustiva sobre una grilla de hiperparámetros.

```python
from sklearn.model_selection import GridSearchCV

# Definir grilla de búsqueda
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Crear modelo base
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='accuracy',      # Métrica a optimizar
    n_jobs=-1,               # Paralelizar
    verbose=1                # Mostrar progreso
)

# Ejecutar búsqueda
grid_search.fit(X_train, y_train)

# Resultados
print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Mejor score CV: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")
```

### 6. RandomizedSearchCV

Más eficiente para espacios grandes de hiperparámetros.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Distribuciones de parámetros
param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=50,               # Número de combinaciones a probar
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"Mejores parámetros: {random_search.best_params_}")
print(f"Mejor score CV: {random_search.best_score_:.4f}")
```

### 7. Cost-Complexity Pruning (ccp_alpha)

Poda basada en complejidad para evitar overfitting.

```python
from sklearn.tree import DecisionTreeClassifier

# Obtener ruta de poda
tree = DecisionTreeClassifier(random_state=42)
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Probar diferentes alphas
train_scores = []
test_scores = []

for alpha in ccp_alphas:
    tree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    tree.fit(X_train, y_train)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

# Visualizar
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, 'b-', label='Train', alpha=0.7)
plt.plot(ccp_alphas, test_scores, 'r-', label='Test', alpha=0.7)
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.title('Cost-Complexity Pruning')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ccp_pruning.png', dpi=150)
plt.show()

# Encontrar mejor alpha
best_idx = np.argmax(test_scores)
best_alpha = ccp_alphas[best_idx]
print(f"Mejor ccp_alpha: {best_alpha:.6f}")
```

### 8. Estrategias de Tuning

#### Orden Recomendado

1. **n_estimators**: Empezar con 100-200, aumentar si hay tiempo
2. **max_depth**: Probar [5, 10, 15, 20, None]
3. **min_samples_split / leaf**: Regularización fina
4. **max_features**: ['sqrt', 'log2', None]

#### Pipeline con Preprocesamiento

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Opcional para RF
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Parámetros del pipeline
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 20, None],
    'rf__min_samples_split': [2, 5]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Mejores parámetros: {grid.best_params_}")
```

### 9. Validación con OOB vs Cross-Validation

```python
# Comparar OOB con CV
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)

print(f"OOB Score: {rf.oob_score_:.4f}")
print(f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Test Score: {rf.score(X_test, y_test):.4f}")
```

### 10. Tips Prácticos

#### ⚡ Rendimiento

```python
# Siempre usar n_jobs=-1
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# Para datasets grandes, usar max_samples
rf = RandomForestClassifier(
    n_estimators=100,
    max_samples=0.8,  # Usar 80% de datos por árbol
    n_jobs=-1
)
```

#### 🎯 Mejores Prácticas

1. **Empezar simple**: Probar defaults primero
2. **OOB Score**: Usar para estimación rápida
3. **n_estimators**: Más es mejor (hasta cierto punto)
4. **max_depth**: Limitar si hay overfitting
5. **Reproducibilidad**: Siempre fijar `random_state`

---

## ✅ Checklist de Verificación

- [ ] Conozco los principales hiperparámetros de árboles y RF
- [ ] Entiendo el tradeoff bias-variance en max_depth
- [ ] Sé usar GridSearchCV para búsqueda exhaustiva
- [ ] Sé usar RandomizedSearchCV para espacios grandes
- [ ] Puedo aplicar cost-complexity pruning
- [ ] Conozco estrategias prácticas de tuning

---

## 📚 Recursos

- [Hyperparameter Tuning - sklearn](https://scikit-learn.org/stable/modules/grid_search.html)
- [Random Forest Parameters - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Pruning Decision Trees](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)
