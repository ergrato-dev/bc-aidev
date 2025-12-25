# K-Nearest Neighbors (KNN)

## 🎯 Objetivos

- Entender el algoritmo KNN y su funcionamiento
- Conocer las métricas de distancia
- Elegir el valor óptimo de k
- Implementar KNN con scikit-learn

## 📋 Contenido

### 1. ¿Qué es KNN?

K-Nearest Neighbors es un algoritmo de **aprendizaje basado en instancias** (lazy learning):

- **No entrena** un modelo explícito
- **Guarda** todos los datos de entrenamiento
- **Predice** basándose en los k vecinos más cercanos

![KNN Distancias](../0-assets/01-knn-distancias.svg)

### 2. Funcionamiento

#### Clasificación

1. Calcular distancia del nuevo punto a todos los puntos de entrenamiento
2. Seleccionar los k puntos más cercanos
3. **Votación mayoritaria**: la clase más común entre los k vecinos

#### Regresión

1. Mismos pasos 1-2
2. **Promedio**: el valor predicho es la media de los k vecinos

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Clasificación
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)
y_pred = knn_clf.predict(X_test)

# Regresión
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)
y_pred = knn_reg.predict(X_test)
```

### 3. Métricas de Distancia

| Distancia      | Fórmula                       | Uso                  |
| -------------- | ----------------------------- | -------------------- |
| **Euclidiana** | $\sqrt{\sum(x_i - y_i)^2}$    | Default, uso general |
| **Manhattan**  | $\sum\|x_i - y_i\|$           | Alta dimensionalidad |
| **Minkowski**  | $(\sum\|x_i - y_i\|^p)^{1/p}$ | Generalización       |

```python
# Euclidiana (default, p=2)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# Manhattan (p=1)
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')

# Minkowski con p personalizado
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)
```

### 4. Eligiendo el k Óptimo

El valor de k afecta el **tradeoff bias-variance**:

| k pequeño           | k grande         |
| ------------------- | ---------------- |
| Baja bias           | Alta bias        |
| Alta varianza       | Baja varianza    |
| Overfitting         | Underfitting     |
| Fronteras complejas | Fronteras suaves |

#### Encontrar k óptimo

```python
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

# Visualizar
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores, 'b-o')
plt.xlabel('Valor de k')
plt.ylabel('Accuracy (CV)')
plt.title('Elección de k óptimo')
plt.grid(True, alpha=0.3)
plt.show()

# Mejor k
best_k = k_range[k_scores.index(max(k_scores))]
print(f"Mejor k: {best_k}")
```

### 5. Importancia de la Normalización

KNN es **sensible a la escala** de las features. Features con valores grandes dominarán la distancia.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ⚠️ SIEMPRE normalizar antes de KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### 6. Curse of Dimensionality

Con muchas features:

- Las distancias se vuelven **similares** entre todos los puntos
- KNN pierde efectividad
- Solución: reducción de dimensionalidad (PCA) o selección de features

### 7. Parámetros Importantes

| Parámetro     | Descripción                             | Default     |
| ------------- | --------------------------------------- | ----------- |
| `n_neighbors` | Número de vecinos (k)                   | 5           |
| `weights`     | 'uniform' o 'distance'                  | 'uniform'   |
| `metric`      | Métrica de distancia                    | 'minkowski' |
| `p`           | Parámetro para Minkowski                | 2           |
| `algorithm`   | 'auto', 'ball_tree', 'kd_tree', 'brute' | 'auto'      |

#### Ponderación por distancia

```python
# Vecinos más cercanos tienen más peso
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
```

### 8. Ventajas y Desventajas

#### ✅ Ventajas

- Simple de entender e implementar
- No requiere entrenamiento (lazy)
- Naturalmente multiclase
- No asume distribución de datos

#### ❌ Desventajas

- Lento en predicción (O(n) por cada predicción)
- Sensible a features irrelevantes
- Requiere normalización
- Mal rendimiento en alta dimensionalidad
- Guarda todo el dataset en memoria

### 9. Ejemplo Completo

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Cargar datos
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline con normalización
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# GridSearch para encontrar mejor k
param_grid = {
    'knn__n_neighbors': range(1, 21),
    'knn__weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Mejor score CV: {grid_search.best_score_:.4f}")

# Evaluar
y_pred = grid_search.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## ✅ Checklist de Verificación

- [ ] Entiendo cómo funciona KNN para clasificación y regresión
- [ ] Conozco las diferentes métricas de distancia
- [ ] Sé cómo elegir el valor óptimo de k
- [ ] Comprendo la importancia de normalizar features
- [ ] Puedo implementar KNN con sklearn

---

## 📚 Recursos

- [KNN - sklearn](https://scikit-learn.org/stable/modules/neighbors.html)
- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
