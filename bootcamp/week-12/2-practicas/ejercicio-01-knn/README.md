# Ejercicio 01: K-Nearest Neighbors con Iris

## 🎯 Objetivo

Implementar KNN para clasificación, explorar diferentes valores de k y métricas de distancia.

## 📋 Instrucciones

### Paso 1: Cargar y Explorar Datos

Cargamos el dataset Iris y exploramos su estructura.

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
print(df['target'].value_counts())
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

### Paso 2: Dividir y Normalizar

KNN es sensible a la escala, siempre normalizamos.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Descomenta** la sección del Paso 2.

### Paso 3: KNN Básico

Entrenamos KNN con k=5 (default).

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

**Descomenta** la sección del Paso 3.

### Paso 4: Encontrar k Óptimo

Probamos diferentes valores de k con validación cruzada.

```python
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

k_range = range(1, 21)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    k_scores.append(scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores, 'b-o')
plt.xlabel('k')
plt.ylabel('CV Accuracy')
plt.title('Elección de k óptimo')
plt.grid(True, alpha=0.3)
plt.savefig('k_optimization.png', dpi=100, bbox_inches='tight')
plt.show()

best_k = list(k_range)[k_scores.index(max(k_scores))]
print(f"Mejor k: {best_k} (Accuracy: {max(k_scores):.4f})")
```

**Descomenta** la sección del Paso 4.

### Paso 5: Comparar Métricas de Distancia

Probamos Euclidiana vs Manhattan.

```python
metrics = ['euclidean', 'manhattan']

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Descomenta** la sección del Paso 5.

### Paso 6: Modelo Final con Pipeline

Creamos pipeline completo y evaluamos.

```python
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=best_k, weights='distance'))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"Test Accuracy: {pipeline.score(X_test, y_test):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

**Descomenta** la sección del Paso 6.

## ✅ Resultado Esperado

- Gráfico de k vs accuracy guardado como `k_optimization.png`
- Accuracy test ≥ 0.93
- Identificación del k óptimo

## 🔗 Recursos

- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
