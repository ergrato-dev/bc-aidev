# 📊 Ejercicio 04: Evaluación de Clustering

## 🎯 Objetivo

Aprender a evaluar y comparar algoritmos de clustering utilizando métricas internas (sin ground truth) y externas (con ground truth).

---

## 📋 Descripción

Evaluar clustering es más complejo que evaluar clasificación supervisada porque a menudo no tenemos etiquetas "correctas". En este ejercicio aprenderás:

- ✅ Métricas internas: Silhouette, Davies-Bouldin, Calinski-Harabasz
- ✅ Métricas externas: ARI, NMI, Homogeneity, Completeness
- ✅ Cómo elegir el número óptimo de clusters
- ✅ Comparar algoritmos de forma objetiva

---

## 📚 Conceptos Clave

### Métricas Internas (Sin Ground Truth)

| Métrica               | Rango   | Mejor | Qué mide                       |
| --------------------- | ------- | ----- | ------------------------------ |
| **Silhouette**        | [-1, 1] | Mayor | Cohesión vs separación         |
| **Davies-Bouldin**    | [0, ∞)  | Menor | Ratio de dispersión/separación |
| **Calinski-Harabasz** | [0, ∞)  | Mayor | Varianza inter/intra cluster   |

### Métricas Externas (Con Ground Truth)

| Métrica                    | Rango   | Mejor | Qué mide                             |
| -------------------------- | ------- | ----- | ------------------------------------ |
| **Adjusted Rand Index**    | [-1, 1] | Mayor | Concordancia de pares ajustada       |
| **Normalized Mutual Info** | [0, 1]  | Mayor | Información mutua normalizada        |
| **Homogeneity**            | [0, 1]  | Mayor | Cada cluster contiene solo una clase |
| **Completeness**           | [0, 1]  | Mayor | Todos de una clase están juntos      |

---

## 🔄 Pasos del Ejercicio

### Paso 1: Calcular Silhouette Score

```python
from sklearn.metrics import silhouette_score, silhouette_samples

# Score global
score = silhouette_score(X, labels)

# Score por muestra (para visualización)
sample_scores = silhouette_samples(X, labels)
```

### Paso 2: Silhouette Plot

```python
def silhouette_plot(X, labels):
    """
    Visualize silhouette score per sample and cluster.
    """
    sample_scores = silhouette_samples(X, labels)
    n_clusters = len(np.unique(labels))

    y_lower = 10
    for i in range(n_clusters):
        cluster_scores = sample_scores[labels == i]
        cluster_scores.sort()

        y_upper = y_lower + len(cluster_scores)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_scores)
        y_lower = y_upper + 10
```

### Paso 3: Encontrar K Óptimo

```python
# Probar diferentes K
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# El K con mayor silhouette es el óptimo
```

### Paso 4: Davies-Bouldin Index

```python
from sklearn.metrics import davies_bouldin_score

dbi = davies_bouldin_score(X, labels)
# Menor es mejor
```

### Paso 5: Métricas Externas

```python
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

# Comparar con etiquetas reales
ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
homo = homogeneity_score(y_true, y_pred)
comp = completeness_score(y_true, y_pred)
```

### Paso 6: Comparar Algoritmos

```python
algorithms = [
    ('K-Means', KMeans(n_clusters=4)),
    ('DBSCAN', DBSCAN(eps=0.5, min_samples=5)),
    ('Hierarchical', AgglomerativeClustering(n_clusters=4))
]

for name, algo in algorithms:
    labels = algo.fit_predict(X)
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    print(f"{name}: Silhouette={sil:.3f}, DBI={dbi:.3f}")
```

---

## 📁 Estructura del Ejercicio

```
ejercicio-04-evaluacion/
├── README.md              # Este archivo
└── starter/
    └── main.py            # Código a completar
```

---

## ✅ Criterios de Éxito

- [ ] Calcular e interpretar silhouette score
- [ ] Crear silhouette plots por cluster
- [ ] Usar método del silhouette para encontrar K óptimo
- [ ] Aplicar Davies-Bouldin y Calinski-Harabasz
- [ ] Evaluar con métricas externas cuando hay ground truth
- [ ] Comparar algoritmos de forma objetiva

---

## 🎯 Resultado Esperado

Al ejecutar el código completo deberías ver:

- Gráficos de silhouette mostrando calidad de clusters
- Curvas de métricas para diferentes valores de K
- Tabla comparativa de algoritmos
- Visualización de clusters buenos vs malos

---

## 📚 Recursos

- [Sklearn Clustering Metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation)
- [Silhouette Analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
- [Comparing Clustering Algorithms](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)

---

## ⏱️ Tiempo Estimado

- **Total**: 50 minutos
- Por paso:
  - Pasos 1-2: 15 min
  - Pasos 3-4: 15 min
  - Pasos 5-6: 20 min

---

## 🔗 Navegación

| ⬅️ Anterior                                             | 🏠 Ejercicios    | Siguiente ➡️                  |
| ------------------------------------------------------- | ---------------- | ----------------------------- |
| [Ejercicio 03: Jerárquico](../ejercicio-03-jerarquico/) | [Prácticas](../) | [Proyecto](../../3-proyecto/) |
