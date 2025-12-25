# 🌳 Ejercicio 03: Clustering Jerárquico

## 🎯 Objetivo

Implementar clustering jerárquico aglomerativo y aprender a interpretar dendrogramas para determinar el número óptimo de clusters.

---

## 📋 Descripción

El clustering jerárquico construye una jerarquía de clusters que se puede visualizar como un árbol (dendrograma). A diferencia de K-Means:

- ✅ No requiere especificar K de antemano
- ✅ Proporciona una estructura jerárquica completa
- ✅ Permite visualizar relaciones entre clusters
- ✅ Diferentes métodos de enlace dan diferentes resultados

---

## 📚 Conceptos Clave

### Tipos de Clustering Jerárquico

| Tipo             | Descripción | Proceso                                            |
| ---------------- | ----------- | -------------------------------------------------- |
| **Aglomerativo** | Bottom-up   | Cada punto empieza como cluster, se van fusionando |
| **Divisivo**     | Top-down    | Todos empiezan juntos, se van dividiendo           |

### Métodos de Enlace (Linkage)

```
Single Linkage:
  Distancia entre clusters = mínima distancia entre cualquier par
  → Tiende a crear clusters alargados (efecto cadena)

Complete Linkage:
  Distancia entre clusters = máxima distancia entre cualquier par
  → Clusters más compactos y esféricos

Average Linkage:
  Distancia entre clusters = promedio de todas las distancias
  → Balance entre single y complete

Ward:
  Minimiza la varianza total intra-cluster
  → Clusters de tamaño similar, muy usado
```

---

## 🔄 Pasos del Ejercicio

### Paso 1: Calcular Matriz de Distancias

```python
from scipy.spatial.distance import pdist, squareform

# Distancias en forma condensada
distances_condensed = pdist(X, metric='euclidean')

# Matriz cuadrada de distancias
distance_matrix = squareform(distances_condensed)
```

### Paso 2: Construir Dendrograma con SciPy

```python
from scipy.cluster.hierarchy import dendrogram, linkage

# Calcular linkage
Z = linkage(X, method='ward')

# Crear dendrograma
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title('Dendrograma - Método Ward')
plt.xlabel('Índice de muestra')
plt.ylabel('Distancia')
plt.show()
```

### Paso 3: Cortar el Dendrograma

```python
from scipy.cluster.hierarchy import fcluster

# Cortar a una distancia específica
labels_by_distance = fcluster(Z, t=5, criterion='distance')

# Cortar para obtener k clusters
labels_by_k = fcluster(Z, t=3, criterion='maxclust')
```

### Paso 4: Comparar Métodos de Linkage

```python
methods = ['single', 'complete', 'average', 'ward']

for method in methods:
    Z = linkage(X, method=method)
    dendrogram(Z)
    plt.title(f'Linkage: {method}')
```

### Paso 5: Usando Scikit-learn

```python
from sklearn.cluster import AgglomerativeClustering

# Especificando número de clusters
clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = clustering.fit_predict(X)

# Sin especificar (cortar por distancia)
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=5,
    linkage='ward'
)
labels = clustering.fit_predict(X)
```

---

## 📁 Estructura del Ejercicio

```
ejercicio-03-jerarquico/
├── README.md              # Este archivo
└── starter/
    └── main.py            # Código a completar
```

---

## ✅ Criterios de Éxito

- [ ] Calcular correctamente la matriz de distancias
- [ ] Construir dendrogramas con diferentes métodos de linkage
- [ ] Interpretar el dendrograma para elegir número de clusters
- [ ] Cortar el dendrograma por distancia y por número de clusters
- [ ] Comparar resultados de diferentes métodos de linkage
- [ ] Implementar versión básica desde cero

---

## 🎯 Resultado Esperado

Al ejecutar el código completo deberías ver:

- Dendrogramas claros mostrando la jerarquía de fusiones
- Diferencias visuales entre métodos de linkage
- Clusters asignados correctamente según el corte elegido
- Comparación visual con K-Means y DBSCAN

---

## 📚 Recursos

- [SciPy Hierarchical Clustering](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- [Sklearn AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [Understanding Dendrograms](https://www.displayr.com/what-is-dendrogram/)

---

## ⏱️ Tiempo Estimado

- **Total**: 45 minutos
- Por paso:
  - Pasos 1-2: 15 min
  - Pasos 3-4: 15 min
  - Paso 5: 15 min

---

## 🔗 Navegación

| ⬅️ Anterior                                     | 🏠 Ejercicios    | Siguiente ➡️                                            |
| ----------------------------------------------- | ---------------- | ------------------------------------------------------- |
| [Ejercicio 02: DBSCAN](../ejercicio-02-dbscan/) | [Prácticas](../) | [Ejercicio 04: Evaluación](../ejercicio-04-evaluacion/) |
