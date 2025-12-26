# 📖 Glosario - Semana 16: Clustering

## A

### Agglomerative Clustering

Enfoque jerárquico que comienza con cada punto como su propio cluster y fusiona iterativamente los más cercanos.

### Average Linkage

Método de enlace que usa la distancia promedio entre todos los pares de puntos de dos clusters.

## B

### Border Point (DBSCAN)

Punto que está dentro de eps de un core point pero no tiene min_samples vecinos propios.

## C

### Calinski-Harabasz Index

Métrica que mide la razón entre dispersión inter-cluster y dispersión intra-cluster. Mayor es mejor.

### Centroide

Punto central de un cluster, calculado como la media de todos los puntos del cluster.

### Cluster

Grupo de puntos de datos similares entre sí y diferentes a los de otros grupos.

### Clustering

Técnica de aprendizaje no supervisado para agrupar datos similares sin etiquetas previas.

### Complete Linkage

Método de enlace que usa la distancia máxima entre puntos de dos clusters.

### Core Point (DBSCAN)

Punto con al menos min_samples vecinos dentro de la distancia eps.

## D

### Davies-Bouldin Index

Métrica que evalúa la separación entre clusters. Menor es mejor.

### DBSCAN

Density-Based Spatial Clustering of Applications with Noise. Algoritmo basado en densidad.

### Dendrograma

Diagrama de árbol que muestra la jerarquía de fusiones en clustering jerárquico.

### Distancia Euclidiana

$$d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$
Medida de distancia más común en clustering.

### Distancia Manhattan

$$d(p, q) = \sum_{i=1}^{n}|p_i - q_i|$$
Distancia como suma de diferencias absolutas.

## E

### Elbow Method (Método del Codo)

Técnica para seleccionar K óptimo buscando el "codo" en la curva de inercia vs K.

### eps (DBSCAN)

Radio de vecindad para determinar puntos cercanos. Hiperparámetro crítico.

## H

### Hierarchical Clustering

Clustering que construye jerarquía de clusters, visualizable como dendrograma.

## I

### Inercia (Inertia)

Suma de distancias cuadradas de cada punto a su centroide más cercano (WCSS).
$$\text{Inercia} = \sum_{i=1}^{n} \min_{\mu_j} ||x_i - \mu_j||^2$$

## K

### K-Means

Algoritmo que particiona datos en K clusters minimizando la inercia.

### K-Means++

Inicialización inteligente de centroides para K-Means que mejora convergencia.

## L

### Linkage

Método para medir distancia entre clusters en clustering jerárquico.

## M

### min_samples (DBSCAN)

Número mínimo de puntos para formar una región densa (core point).

## N

### Noise Point (DBSCAN)

Punto que no es core ni border. Considerado outlier/anomalía.

### NMI (Normalized Mutual Information)

Métrica que compara clustering con ground truth. Requiere etiquetas reales.

## R

### Rand Index (ARI)

Adjusted Rand Index. Métrica de similitud entre dos asignaciones de clusters.

## S

### Silhouette Coefficient

Para un punto $i$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
Donde $a(i)$ = distancia media intra-cluster, $b(i)$ = distancia media al cluster más cercano.

### Silhouette Score

Promedio del coeficiente silhouette de todos los puntos. Rango [-1, 1], mayor es mejor.

### Single Linkage

Método de enlace que usa la distancia mínima entre puntos de dos clusters.

## U

### Unsupervised Learning

Aprendizaje sin etiquetas. Encuentra patrones y estructura en datos no etiquetados.

## W

### Ward Linkage

Método que minimiza la varianza total intra-cluster al fusionar. Produce clusters compactos.

### WCSS

Within-Cluster Sum of Squares. Sinónimo de inercia.

---

## 📊 Comparación de Algoritmos

| Aspecto        | K-Means         | DBSCAN      | Jerárquico      |
| -------------- | --------------- | ----------- | --------------- |
| Forma clusters | Esféricos       | Arbitraria  | Variable        |
| Requiere K     | ✅ Sí           | ❌ No       | ✅ Sí (o corte) |
| Outliers       | ❌ No detecta   | ✅ Detecta  | ❌ No detecta   |
| Escalabilidad  | ✅ Buena        | ⚠️ Moderada | ❌ Limitada     |
| Reproducible   | ⚠️ Depende init | ✅ Sí       | ✅ Sí           |

---

## 🔗 Navegación

| ⬅️ Recursos                           | 🏠 Semana 16           |
| ------------------------------------- | ---------------------- |
| [4-recursos](../4-recursos/README.md) | [README](../README.md) |
