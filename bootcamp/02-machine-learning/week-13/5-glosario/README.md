# 📖 Glosario - Semana 13: Clustering

Términos clave ordenados alfabéticamente.

---

## A

### Agglomerative Clustering

**Clustering aglomerativo** - Método jerárquico bottom-up que comienza con cada punto como su propio cluster y los va fusionando iterativamente según su proximidad.

```python
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels = clustering.fit_predict(X)
```

### Adjusted Rand Index (ARI)

**Índice de Rand Ajustado** - Métrica externa que mide la similitud entre clustering predicho y etiquetas reales, ajustada por azar. Rango: [-1, 1], mayor es mejor.

$$ARI = \frac{RI - E[RI]}{max(RI) - E[RI]}$$

---

## B

### Border Point

**Punto frontera** - En DBSCAN, punto que no es core pero está dentro del radio epsilon de un punto core. Pertenece al cluster pero no lo extiende.

---

## C

### Calinski-Harabasz Index

**Índice de Calinski-Harabasz** - También llamado Variance Ratio Criterion. Mide la relación entre dispersión inter-cluster e intra-cluster. Mayor es mejor.

$$CH = \frac{SS_B / (k-1)}{SS_W / (n-k)}$$

### Centroid

**Centroide** - Punto central de un cluster, calculado como la media de todos los puntos del cluster. Usado en K-Means.

```python
centroid = X[labels == cluster_id].mean(axis=0)
```

### Cluster

**Grupo/Conglomerado** - Conjunto de puntos que son más similares entre sí que con puntos de otros grupos.

### Complete Linkage

**Enlace completo** - Método de linkage que define la distancia entre clusters como la máxima distancia entre cualquier par de puntos. Produce clusters compactos.

### Core Point

**Punto núcleo** - En DBSCAN, punto que tiene al menos `min_samples` vecinos dentro del radio `epsilon`. Puede expandir el cluster.

---

## D

### Davies-Bouldin Index

**Índice de Davies-Bouldin** - Métrica interna que mide la "compacidad" promedio de clusters relativa a la separación entre ellos. Menor es mejor.

$$DB = \frac{1}{k}\sum_{i=1}^{k} max_{j \neq i} \frac{s_i + s_j}{d_{ij}}$$

### DBSCAN

**Density-Based Spatial Clustering of Applications with Noise** - Algoritmo que agrupa puntos densamente conectados y marca como ruido los puntos en regiones de baja densidad.

```python
from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=0.5, min_samples=5)
labels = clustering.fit_predict(X)
```

### Dendrogram

**Dendrograma** - Diagrama de árbol que muestra la secuencia de fusiones (o divisiones) en clustering jerárquico. El eje Y muestra la distancia de fusión.

### Density-Based Clustering

**Clustering basado en densidad** - Familia de algoritmos (DBSCAN, OPTICS, HDBSCAN) que identifican clusters como regiones de alta densidad separadas por regiones de baja densidad.

### Divisive Clustering

**Clustering divisivo** - Método jerárquico top-down que comienza con todos los puntos en un cluster y los va dividiendo. Opuesto al aglomerativo.

---

## E

### Elbow Method

**Método del codo** - Técnica para elegir el número óptimo de clusters (K) graficando la inercia vs K y buscando el "codo" donde la mejora se estabiliza.

### Epsilon (ε)

**Radio de vecindad** - En DBSCAN, distancia máxima para considerar dos puntos como vecinos. Parámetro crítico del algoritmo.

### Euclidean Distance

**Distancia euclidiana** - Medida de distancia más común, la "línea recta" entre dos puntos.

$$d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

---

## H

### Hierarchical Clustering

**Clustering jerárquico** - Familia de algoritmos que construyen una jerarquía de clusters, visualizable como dendrograma. Puede ser aglomerativo o divisivo.

### Homogeneity

**Homogeneidad** - Métrica externa que mide si cada cluster contiene solo miembros de una única clase. Rango [0,1].

---

## I

### Inertia

**Inercia** - También llamada WCSS (Within-Cluster Sum of Squares). Suma de distancias cuadradas de cada punto a su centroide. K-Means minimiza esta métrica.

$$Inertia = \sum_{i=1}^{n} ||x_i - c_{y_i}||^2$$

---

## K

### K-Distance Graph

**Gráfico de K-distancia** - Técnica para elegir epsilon en DBSCAN. Grafica la distancia al k-ésimo vecino más cercano para cada punto, ordenados. El "codo" sugiere epsilon.

### K-Means

**K-Medias** - Algoritmo de clustering que particiona datos en K clusters minimizando la suma de distancias cuadradas intra-cluster.

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)
```

### K-Means++

**Inicialización inteligente** - Método de inicialización para K-Means que selecciona centroides iniciales de forma que estén dispersos, mejorando convergencia.

---

## L

### Linkage

**Enlace/Unión** - Criterio para medir distancia entre clusters en clustering jerárquico. Tipos: single, complete, average, ward.

---

## M

### Manhattan Distance

**Distancia Manhattan** - Suma de diferencias absolutas en cada dimensión. También llamada distancia L1 o city-block.

$$d(x,y) = \sum_{i=1}^{n}|x_i - y_i|$$

### Min_samples

**Mínimo de muestras** - En DBSCAN, número mínimo de puntos requeridos en el vecindario epsilon para que un punto sea considerado core.

---

## N

### Noise Points

**Puntos de ruido** - En DBSCAN, puntos que no son core ni border. Se etiquetan como -1 y se consideran outliers.

### Normalized Mutual Information (NMI)

**Información mutua normalizada** - Métrica externa que mide cuánta información comparten las etiquetas predichas y reales. Rango [0,1].

---

## P

### Partitional Clustering

**Clustering particional** - Algoritmos que dividen datos en K particiones disjuntas sin jerarquía. Ejemplo: K-Means.

---

## R

### RFM Analysis

**Análisis RFM** - Técnica de segmentación de clientes basada en Recency (recencia), Frequency (frecuencia) y Monetary (valor monetario).

---

## S

### Silhouette Score

**Coeficiente de silueta** - Métrica que mide qué tan similar es un punto a su cluster comparado con otros clusters. Rango [-1, 1], mayor es mejor.

$$s(i) = \frac{b(i) - a(i)}{max(a(i), b(i))}$$

Donde:

- $a(i)$ = distancia promedio a puntos del mismo cluster
- $b(i)$ = distancia promedio al cluster más cercano

### Single Linkage

**Enlace simple** - Método de linkage que define distancia entre clusters como la mínima distancia entre cualquier par de puntos. Susceptible a "chain effect".

---

## U

### Unsupervised Learning

**Aprendizaje no supervisado** - Rama del ML donde no hay etiquetas de salida. El objetivo es descubrir estructura en los datos (clustering, reducción de dimensionalidad).

---

## V

### V-Measure

**Medida V** - Media armónica de homogeneity y completeness. Combina ambas métricas externas.

$$V = \frac{2 \times homogeneity \times completeness}{homogeneity + completeness}$$

---

## W

### Ward's Method

**Método de Ward** - Método de linkage que minimiza la varianza total intra-cluster al fusionar. Tiende a crear clusters de tamaño similar. Muy usado en práctica.

### WCSS

**Within-Cluster Sum of Squares** - Ver: Inertia

---

## 🔗 Navegación

| ⬅️ Recursos                   | 🏠 Semana                  | Siguiente ➡️              |
| ----------------------------- | -------------------------- | ------------------------- |
| [Recursos](../../4-recursos/) | [Week 13](../../README.md) | [Week 14](../../week-14/) |
