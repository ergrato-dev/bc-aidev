# 🔬 Ejercicio 02: DBSCAN - Clustering Basado en Densidad

## 🎯 Objetivo

Implementar el algoritmo DBSCAN desde cero y entender cómo detecta clusters de forma arbitraria basándose en la densidad de puntos.

---

## 📋 Descripción

En este ejercicio implementarás DBSCAN (Density-Based Spatial Clustering of Applications with Noise), un algoritmo que:

- ✅ No requiere especificar el número de clusters
- ✅ Detecta clusters de formas arbitrarias
- ✅ Identifica automáticamente puntos de ruido (outliers)
- ✅ Se basa en densidad de puntos, no en distancias a centroides

---

## 📚 Conceptos Clave

### Tipos de Puntos en DBSCAN

```
Punto Core (Núcleo):
  Tiene al menos min_samples vecinos dentro de epsilon

Punto Border (Frontera):
  No es core, pero está dentro de epsilon de un punto core

Punto Noise (Ruido):
  No es core ni está cerca de ningún punto core
```

### Parámetros

| Parámetro     | Descripción                     | Efecto                          |
| ------------- | ------------------------------- | ------------------------------- |
| `epsilon (ε)` | Radio de vecindad               | ε grande → clusters más grandes |
| `min_samples` | Mínimo de vecinos para ser core | Mayor → clusters más densos     |

---

## 🔄 Pasos del Ejercicio

### Paso 1: Generar Datos con Formas Complejas

Usaremos datos que K-Means no puede manejar bien:

```python
from sklearn.datasets import make_moons, make_circles

# Datos en forma de lunas
X_moons, y_moons = make_moons(n_samples=300, noise=0.05)

# Datos en forma de círculos concéntricos
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5)
```

### Paso 2: Implementar Region Query

Encontrar todos los vecinos dentro de epsilon:

```python
def region_query(X, point_idx, epsilon):
    """
    Find all points within epsilon distance of a point.

    Returns:
        List of neighbor indices
    """
    neighbors = []
    for i in range(len(X)):
        if euclidean_distance(X[point_idx], X[i]) <= epsilon:
            neighbors.append(i)
    return neighbors
```

### Paso 3: Expandir Cluster

Cuando encontramos un punto core, expandimos el cluster:

```python
def expand_cluster(X, labels, point_idx, neighbors,
                   cluster_id, epsilon, min_samples):
    """
    Expand cluster by adding all density-reachable points.
    """
    labels[point_idx] = cluster_id

    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]

        if labels[neighbor_idx] == -1:  # Was noise
            labels[neighbor_idx] = cluster_id
        elif labels[neighbor_idx] == 0:  # Unvisited
            labels[neighbor_idx] = cluster_id

            # Check if this neighbor is also a core point
            neighbor_neighbors = region_query(X, neighbor_idx, epsilon)
            if len(neighbor_neighbors) >= min_samples:
                neighbors.extend(neighbor_neighbors)

        i += 1
```

### Paso 4: Algoritmo DBSCAN Completo

```python
def dbscan_scratch(X, epsilon, min_samples):
    """
    DBSCAN clustering algorithm.

    Labels:
        -1: Noise
         0: Unvisited
        >0: Cluster ID
    """
    n_samples = X.shape[0]
    labels = np.zeros(n_samples, dtype=int)  # 0 = unvisited
    cluster_id = 0

    for point_idx in range(n_samples):
        if labels[point_idx] != 0:  # Already processed
            continue

        neighbors = region_query(X, point_idx, epsilon)

        if len(neighbors) < min_samples:
            labels[point_idx] = -1  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(X, labels, point_idx, neighbors,
                          cluster_id, epsilon, min_samples)

    return labels
```

### Paso 5: Selección de Epsilon (k-distance graph)

Método del codo para elegir epsilon:

```python
from sklearn.neighbors import NearestNeighbors

def find_optimal_epsilon(X, min_samples):
    """
    Plot k-distance graph to find optimal epsilon.
    """
    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)

    # Sort distances to k-th neighbor
    k_distances = np.sort(distances[:, -1])

    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{min_samples}-distance')
    plt.title('K-Distance Graph for Epsilon Selection')
    plt.grid(True)
    plt.show()
```

### Paso 6: Visualización y Comparación

Comparar DBSCAN con K-Means en datos complejos:

```python
# DBSCAN
labels_dbscan = dbscan_scratch(X_moons, epsilon=0.2, min_samples=5)

# K-Means (para comparación)
kmeans = KMeans(n_clusters=2)
labels_kmeans = kmeans.fit_predict(X_moons)

# Visualizar
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_dbscan, cmap='viridis')
axes[0].set_title('DBSCAN')
axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_kmeans, cmap='viridis')
axes[1].set_title('K-Means')
plt.show()
```

---

## 📁 Estructura del Ejercicio

```
ejercicio-02-dbscan/
├── README.md              # Este archivo
└── starter/
    └── main.py            # Código a completar
```

---

## ✅ Criterios de Éxito

- [ ] Region query encuentra correctamente vecinos dentro de epsilon
- [ ] Algoritmo distingue puntos core, border y noise
- [ ] Expansión de cluster funciona correctamente
- [ ] K-distance graph ayuda a elegir epsilon óptimo
- [ ] DBSCAN supera a K-Means en datos con formas complejas
- [ ] Visualización clara de clusters y puntos de ruido

---

## 🎯 Resultado Esperado

Al ejecutar el código completo deberías ver:

- DBSCAN detectando correctamente las formas de luna/círculo
- K-Means fallando en estos mismos datos
- Puntos de ruido identificados (etiqueta -1)
- K-distance graph mostrando el "codo" para epsilon óptimo

---

## 📚 Recursos

- [DBSCAN Paper Original](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)
- [Scikit-learn DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [Visualización Interactiva](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)

---

## ⏱️ Tiempo Estimado

- **Total**: 45 minutos
- Por paso:
  - Pasos 1-2: 10 min
  - Pasos 3-4: 20 min
  - Pasos 5-6: 15 min

---

## 🔗 Navegación

| ⬅️ Anterior                                      | 🏠 Ejercicios    | Siguiente ➡️                                            |
| ------------------------------------------------ | ---------------- | ------------------------------------------------------- |
| [Ejercicio 01: K-Means](../ejercicio-01-kmeans/) | [Prácticas](../) | [Ejercicio 03: Jerárquico](../ejercicio-03-jerarquico/) |
