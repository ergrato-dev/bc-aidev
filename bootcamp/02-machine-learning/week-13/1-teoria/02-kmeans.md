# 🎯 K-Means Clustering

## 🎯 Objetivos de Aprendizaje

- Comprender el algoritmo K-Means paso a paso
- Implementar K-Means desde cero y con scikit-learn
- Aplicar el método del codo para seleccionar K óptimo
- Conocer K-Means++ y otras variantes
- Identificar limitaciones y cuándo usar K-Means

---

## 📋 ¿Qué es K-Means?

K-Means es un algoritmo de clustering por partición que divide n observaciones en **K clusters**, donde cada observación pertenece al cluster con el **centroide más cercano**.

### Idea Principal

```
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│     ●●●                         Objetivo:                     │
│    ●●●●●    ★                   Minimizar la suma de          │
│     ●●●                         distancias² de cada punto     │
│                                 a su centroide                │
│              ▲▲▲                                              │
│             ▲▲▲▲▲   ★           ★ = Centroide                 │
│              ▲▲▲                                              │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## 🔄 El Algoritmo K-Means

### Pasos del Algoritmo

1. **Inicializar** K centroides aleatoriamente
2. **Asignar** cada punto al centroide más cercano
3. **Actualizar** cada centroide como la media de sus puntos asignados
4. **Repetir** pasos 2-3 hasta convergencia

![Algoritmo K-Means](../0-assets/01-kmeans-algoritmo.svg)

### Implementación desde Cero

```python
import numpy as np

def kmeans_from_scratch(X: np.ndarray, k: int, max_iters: int = 100,
                        tol: float = 1e-4) -> tuple:
    """
    Implement K-Means clustering from scratch.

    Args:
        X: Data matrix of shape (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        centroids: Final centroid positions
        labels: Cluster assignment for each point
    """
    n_samples, n_features = X.shape

    # Step 1: Initialize centroids randomly from data points
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices].copy()

    for iteration in range(max_iters):
        # Step 2: Assign each point to nearest centroid
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.sqrt(np.sum((X - centroids[i])**2, axis=1))
        labels = np.argmin(distances, axis=1)

        # Step 3: Update centroids as mean of assigned points
        new_centroids = np.zeros((k, n_features))
        for i in range(k):
            if np.sum(labels == i) > 0:  # Avoid empty clusters
                new_centroids[i] = X[labels == i].mean(axis=0)
            else:
                new_centroids[i] = centroids[i]  # Keep old centroid

        # Check convergence
        if np.all(np.abs(new_centroids - centroids) < tol):
            print(f"Converged at iteration {iteration}")
            break

        centroids = new_centroids

    return centroids, labels


# Ejemplo de uso
np.random.seed(42)
# Generar datos de ejemplo
X = np.vstack([
    np.random.randn(50, 2) + [0, 0],
    np.random.randn(50, 2) + [5, 5],
    np.random.randn(50, 2) + [10, 0]
])

centroids, labels = kmeans_from_scratch(X, k=3)
print(f"Centroides:\n{centroids}")
```

---

## 📐 Función Objetivo: Inercia (WCSS)

K-Means minimiza la **inercia** (Within-Cluster Sum of Squares):

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

Donde:

- $C_k$ es el conjunto de puntos en cluster k
- $\mu_k$ es el centroide del cluster k

```python
def calculate_inertia(X: np.ndarray, centroids: np.ndarray,
                      labels: np.ndarray) -> float:
    """
    Calculate the inertia (WCSS) of a clustering.

    Args:
        X: Data matrix
        centroids: Cluster centroids
        labels: Cluster assignments

    Returns:
        inertia: Sum of squared distances to centroids
    """
    inertia = 0
    for k in range(len(centroids)):
        cluster_points = X[labels == k]
        inertia += np.sum((cluster_points - centroids[k])**2)
    return inertia
```

---

## 🔢 Selección de K: Método del Codo

### El Problema

K-Means requiere especificar K, pero ¿cómo elegir el valor correcto?

### El Método del Codo (Elbow Method)

1. Ejecutar K-Means para K = 1, 2, 3, ...
2. Calcular inercia para cada K
3. Graficar K vs Inercia
4. Buscar el "codo" donde la mejora se estabiliza

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def elbow_method(X: np.ndarray, k_range: range) -> None:
    """
    Apply the elbow method to find optimal K.

    Args:
        X: Data matrix
        k_range: Range of K values to try
    """
    inertias = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (K)', fontsize=12)
    plt.ylabel('Inercia (WCSS)', fontsize=12)
    plt.title('Método del Codo para Selección de K', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()

    # Print inertias
    for k, inertia in zip(k_range, inertias):
        print(f"K={k}: Inercia={inertia:.2f}")


# Ejemplo
elbow_method(X, k_range=range(1, 11))
```

### Interpretación del Codo

```
Inercia
   │
   │\
   │ \
   │  \
   │   ╲____ ← Codo: K óptimo
   │        ╲____
   │             ╲____
   └───────────────────── K
        1  2  3  4  5  6
```

El codo indica el punto donde agregar más clusters no mejora significativamente la compacidad.

---

## 🚀 K-Means++ : Inicialización Inteligente

### Problema con Inicialización Aleatoria

La inicialización aleatoria puede llevar a:

- Convergencia a mínimos locales
- Clusters subóptimos
- Resultados inconsistentes

### Solución: K-Means++

Algoritmo de inicialización que:

1. Elige el primer centroide al azar
2. Para cada nuevo centroide:
   - Calcula D(x)² = distancia al centroide más cercano
   - Elige nuevo centroide con probabilidad proporcional a D(x)²
3. Resultado: centroides iniciales bien dispersos

```python
def kmeans_plus_plus_init(X: np.ndarray, k: int) -> np.ndarray:
    """
    Initialize centroids using K-Means++ algorithm.

    Args:
        X: Data matrix
        k: Number of clusters

    Returns:
        centroids: Initial centroid positions
    """
    n_samples = X.shape[0]
    centroids = []

    # First centroid: random
    first_idx = np.random.randint(n_samples)
    centroids.append(X[first_idx])

    for _ in range(1, k):
        # Calculate squared distances to nearest centroid
        distances = np.zeros(n_samples)
        for i in range(n_samples):
            min_dist = float('inf')
            for c in centroids:
                dist = np.sum((X[i] - c)**2)
                min_dist = min(min_dist, dist)
            distances[i] = min_dist

        # Choose next centroid with probability proportional to D²
        probabilities = distances / distances.sum()
        next_idx = np.random.choice(n_samples, p=probabilities)
        centroids.append(X[next_idx])

    return np.array(centroids)


# En scikit-learn, K-Means++ es el default
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
```

---

## 🐍 K-Means con Scikit-learn

### Uso Básico

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Preparar datos
X = np.random.randn(300, 2)  # 300 puntos, 2 features

# Normalizar (importante para K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear y entrenar modelo
kmeans = KMeans(
    n_clusters=3,          # Número de clusters
    init='k-means++',      # Inicialización inteligente
    n_init=10,             # Número de ejecuciones con diferentes init
    max_iter=300,          # Iteraciones máximas
    random_state=42        # Reproducibilidad
)

# Fit y predict
kmeans.fit(X_scaled)
labels = kmeans.labels_

# O en un solo paso
labels = kmeans.fit_predict(X_scaled)
```

### Atributos Importantes

```python
# Después de fit()
print(f"Centroides:\n{kmeans.cluster_centers_}")
print(f"Labels: {kmeans.labels_}")
print(f"Inercia: {kmeans.inertia_}")
print(f"Iteraciones: {kmeans.n_iter_}")

# Predecir cluster para nuevos puntos
new_points = np.array([[0, 0], [5, 5]])
new_points_scaled = scaler.transform(new_points)
predictions = kmeans.predict(new_points_scaled)
```

### Visualización Completa

```python
import matplotlib.pyplot as plt

def visualize_kmeans(X: np.ndarray, kmeans: KMeans) -> None:
    """
    Visualize K-Means clustering results.

    Args:
        X: Data matrix (2D for visualization)
        kmeans: Fitted KMeans model
    """
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    plt.figure(figsize=(10, 8))

    # Plot points colored by cluster
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels,
                         cmap='viridis', alpha=0.6, s=50)

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1],
               c='red', marker='X', s=200, edgecolors='black',
               linewidths=2, label='Centroides')

    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'K-Means Clustering (K={kmeans.n_clusters})\n'
              f'Inercia: {kmeans.inertia_:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


visualize_kmeans(X_scaled, kmeans)
```

---

## ⚠️ Limitaciones de K-Means

### 1. Requiere Especificar K

```python
# No siempre sabemos cuántos clusters hay
# Usar método del codo o silhouette
```

### 2. Asume Clusters Esféricos

K-Means no funciona bien con:

```
# Clusters con formas irregulares
from sklearn.datasets import make_moons

X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_moons)
# Resultado: MAL - no puede separar las "lunas"
```

### 3. Sensible a Outliers

```python
# Un outlier puede "arrastrar" un centroide
X_with_outlier = np.vstack([X, [[100, 100]]])  # Outlier extremo
kmeans.fit(X_with_outlier)
# El centroide se moverá hacia el outlier
```

### 4. Sensible a la Escala

```python
# SIEMPRE normalizar antes de K-Means
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)
```

### 5. Mínimos Locales

```python
# Ejecutar múltiples veces y tomar el mejor
kmeans = KMeans(n_clusters=3, n_init=10)  # 10 inicializaciones
# Automáticamente guarda la mejor (menor inercia)
```

---

## 🔧 Variantes de K-Means

### Mini-Batch K-Means

Para datasets muy grandes:

```python
from sklearn.cluster import MiniBatchKMeans

# Más rápido, usa subconjuntos (batches) aleatorios
mbkmeans = MiniBatchKMeans(
    n_clusters=3,
    batch_size=100,      # Tamaño del batch
    random_state=42
)
mbkmeans.fit(X_large)
```

### K-Medoids (PAM)

Usa puntos reales como centroides (más robusto a outliers):

```python
from sklearn_extra.cluster import KMedoids

kmedoids = KMedoids(n_clusters=3, random_state=42)
kmedoids.fit(X)
# cluster_centers_ son puntos reales de los datos
```

---

## 📊 Ejemplo Completo: Segmentación de Clientes

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Simular datos de clientes
np.random.seed(42)
n_customers = 500

data = {
    'annual_income': np.concatenate([
        np.random.normal(30000, 5000, 200),   # Bajo ingreso
        np.random.normal(60000, 10000, 200),  # Medio
        np.random.normal(120000, 20000, 100)  # Alto
    ]),
    'spending_score': np.concatenate([
        np.random.normal(30, 10, 200),   # Bajo gasto
        np.random.normal(60, 15, 200),   # Medio
        np.random.normal(80, 10, 100)    # Alto
    ])
}
df = pd.DataFrame(data)

# Preparar datos
X = df[['annual_income', 'spending_score']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Método del codo
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Elegir K=3 basado en el codo
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Analizar segmentos
print("\n📊 Análisis de Segmentos:")
print(df.groupby('cluster').agg({
    'annual_income': ['mean', 'std', 'count'],
    'spending_score': ['mean', 'std']
}).round(2))

# Nombrar segmentos basado en características
segment_names = {0: 'Ahorradores', 1: 'Consumidores Moderados', 2: 'Premium'}
df['segment'] = df['cluster'].map(segment_names)
```

---

## ✅ Checklist de Verificación

- [ ] Entiendo los pasos del algoritmo K-Means
- [ ] Puedo implementar K-Means desde cero
- [ ] Sé usar el método del codo para elegir K
- [ ] Comprendo K-Means++ y por qué es mejor
- [ ] Conozco las limitaciones de K-Means
- [ ] Sé cuándo usar Mini-Batch K-Means

---

## 📚 Recursos Adicionales

- [K-Means Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Visualizing K-Means](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
- [K-Means++ Paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)

---

## 🔗 Navegación

| ⬅️ Anterior                                   | 🏠 Inicio                 | Siguiente ➡️           |
| --------------------------------------------- | ------------------------- | ---------------------- |
| [Introducción](01-introduccion-clustering.md) | [Semana 13](../README.md) | [DBSCAN](03-dbscan.md) |
