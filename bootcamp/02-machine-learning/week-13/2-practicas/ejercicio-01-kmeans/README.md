# 🎯 Ejercicio 01: K-Means desde Cero

## 🎯 Objetivos

- Implementar el algoritmo K-Means paso a paso
- Aplicar el método del codo para seleccionar K
- Visualizar clusters y centroides
- Comparar implementación propia con scikit-learn

---

## 📋 Contexto

En este ejercicio implementarás K-Means desde cero para entender completamente el algoritmo, y luego lo compararás con la implementación de scikit-learn.

---

## 📝 Instrucciones

Abre el archivo `starter/main.py` y sigue los pasos indicados.

### Paso 1: Generar Datos de Prueba

Comenzamos creando un dataset sintético con clusters bien definidos:

```python
import numpy as np
from sklearn.datasets import make_blobs

# Generar datos con 4 clusters
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.6, random_state=42)
```

**Descomenta** las líneas correspondientes en `starter/main.py`.

### Paso 2: Implementar Cálculo de Distancia

La distancia euclidiana es fundamental para K-Means:

```python
def euclidean_distance(x1, x2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((x1 - x2) ** 2))
```

### Paso 3: Inicialización de Centroides

Seleccionar K puntos aleatorios como centroides iniciales:

```python
def initialize_centroids(X, k):
    """Randomly select k points as initial centroids."""
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices].copy()
```

### Paso 4: Asignación de Clusters

Asignar cada punto al centroide más cercano:

```python
def assign_clusters(X, centroids):
    """Assign each point to nearest centroid."""
    n_samples = X.shape[0]
    k = len(centroids)
    labels = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        distances = [euclidean_distance(X[i], c) for c in centroids]
        labels[i] = np.argmin(distances)

    return labels
```

### Paso 5: Actualización de Centroides

Recalcular centroides como la media de los puntos asignados:

```python
def update_centroids(X, labels, k):
    """Update centroids as mean of assigned points."""
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features))

    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            centroids[i] = cluster_points.mean(axis=0)

    return centroids
```

### Paso 6: Algoritmo Completo

Integrar todo en una función:

```python
def kmeans_scratch(X, k, max_iters=100, tol=1e-4):
    """K-Means implementation from scratch."""
    centroids = initialize_centroids(X, k)

    for iteration in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)

        # Check convergence
        shift = np.sum((new_centroids - centroids) ** 2)
        if shift < tol:
            print(f"Converged at iteration {iteration}")
            break

        centroids = new_centroids

    return centroids, labels
```

### Paso 7: Método del Codo

Encontrar el K óptimo:

```python
def elbow_method(X, k_range):
    """Calculate inertia for different K values."""
    inertias = []
    for k in k_range:
        centroids, labels = kmeans_scratch(X, k)
        inertia = calculate_inertia(X, centroids, labels)
        inertias.append(inertia)
    return inertias
```

### Paso 8: Visualización

Graficar clusters y comparar resultados:

```python
def visualize_clusters(X, labels, centroids, title):
    """Visualize clustering results."""
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1],
               c='red', marker='X', s=200, edgecolors='black')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
```

---

## ✅ Verificación

Tu implementación debe:

1. [ ] Converger en menos de 50 iteraciones
2. [ ] Encontrar los 4 clusters originales
3. [ ] Mostrar inercia decreciente con más K
4. [ ] Producir resultados similares a sklearn

---

## 🔗 Navegación

| ⬅️ Anterior                                                     | 🏠 Inicio                    | Siguiente ➡️                                     |
| --------------------------------------------------------------- | ---------------------------- | ------------------------------------------------ |
| [Teoría Jerárquico](../../1-teoria/04-clustering-jerarquico.md) | [Semana 13](../../README.md) | [Ejercicio 02](../ejercicio-02-dbscan/README.md) |
