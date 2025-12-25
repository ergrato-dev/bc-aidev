# 🔮 Introducción al Clustering

## 🎯 Objetivos de Aprendizaje

- Comprender el paradigma del aprendizaje no supervisado
- Diferenciar clustering de clasificación
- Identificar aplicaciones reales del clustering
- Conocer los tipos principales de algoritmos de clustering

---

## 📋 ¿Qué es el Aprendizaje No Supervisado?

### Supervisado vs No Supervisado

En **aprendizaje supervisado** tenemos:

- Datos de entrada (features) **X**
- Etiquetas conocidas (labels) **y**
- Objetivo: aprender la relación X → y

En **aprendizaje no supervisado**:

- Solo tenemos datos de entrada **X**
- **No hay etiquetas**
- Objetivo: descubrir estructura oculta en los datos

```python
# Supervisado: tenemos etiquetas
X_train = [[1.2, 3.4], [2.1, 1.8], [5.2, 7.1]]
y_train = ['gato', 'gato', 'perro']  # ← etiquetas conocidas

# No supervisado: solo datos
X = [[1.2, 3.4], [2.1, 1.8], [5.2, 7.1], [4.8, 6.9]]
# No hay y - queremos descubrir grupos naturales
```

### ¿Qué es Clustering?

**Clustering** (agrupamiento) es la tarea de agrupar objetos similares en conjuntos llamados **clusters**, de manera que:

- Objetos **dentro** del mismo cluster sean **similares** entre sí
- Objetos en **diferentes** clusters sean **distintos** entre sí

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    ●●●                    ▲▲▲                               │
│   ●●●●●                  ▲▲▲▲▲                ■■■           │
│    ●●●                    ▲▲▲                ■■■■■          │
│                                               ■■■           │
│  Cluster 1              Cluster 2          Cluster 3        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🌍 Aplicaciones del Clustering

### 1. Segmentación de Clientes

```python
# Agrupar clientes por comportamiento de compra
customer_data = {
    'frecuencia_compras': [12, 2, 15, 1, 8],
    'gasto_promedio': [500, 50, 600, 30, 400],
    'antiguedad_meses': [24, 3, 36, 1, 18]
}
# Resultado: "VIP", "Ocasional", "Nuevo"
```

### 2. Compresión de Imágenes

Reducir colores agrupando píxeles similares:

```python
from sklearn.cluster import KMeans
import numpy as np

# Imagen con millones de colores → reducir a 16
pixels = image.reshape(-1, 3)  # RGB values
kmeans = KMeans(n_clusters=16)
compressed = kmeans.cluster_centers_[kmeans.predict(pixels)]
```

### 3. Detección de Anomalías

Identificar puntos que no pertenecen a ningún grupo:

```python
from sklearn.cluster import DBSCAN

# Transacciones bancarias
transactions = load_transactions()
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(transactions)

# labels == -1 son potenciales fraudes
anomalies = transactions[labels == -1]
```

### 4. Otras Aplicaciones

| Dominio        | Aplicación                                |
| -------------- | ----------------------------------------- |
| Biología       | Agrupación de genes con expresión similar |
| Redes Sociales | Detección de comunidades                  |
| Documentos     | Agrupación de artículos por tema          |
| Astronomía     | Clasificación de estrellas/galaxias       |
| Marketing      | Segmentación de mercado                   |

---

## 🔬 Tipos de Algoritmos de Clustering

### 1. Basados en Partición

Dividen los datos en K grupos no superpuestos.

```
Ejemplos: K-Means, K-Medoids
Características:
- Requieren especificar K
- Clusters esféricos
- Rápidos y escalables
```

### 2. Basados en Densidad

Definen clusters como regiones densas separadas por regiones de baja densidad.

```
Ejemplos: DBSCAN, OPTICS, HDBSCAN
Características:
- No requieren especificar K
- Detectan formas arbitrarias
- Identifican outliers
```

### 3. Jerárquicos

Crean una jerarquía de clusters (árbol).

```
Ejemplos: Agglomerative, Divisive
Características:
- Producen dendrograma
- No requieren K inicial
- Costosos computacionalmente
```

### 4. Basados en Modelos

Asumen que los datos provienen de una distribución.

```
Ejemplos: Gaussian Mixture Models (GMM)
Características:
- Probabilísticos
- Clusters suaves (soft clustering)
- Más flexibles pero complejos
```

### Comparación Visual

```
┌──────────────────────────────────────────────────────────────┐
│  K-Means           DBSCAN             Jerárquico             │
│                                                              │
│    ○○○               ~~~                  ┌──┬──┐            │
│   ○○○○○             ~~~~~                 │  │  │            │
│    ○○○               ~~~                ┌─┴──┴──┴─┐          │
│                    (formas             │ │ │ │ │ │          │
│  (esféricos)      arbitrarias)        A B C D E F          │
│                                       (dendrograma)         │
└──────────────────────────────────────────────────────────────┘
```

---

## 📏 Concepto de Distancia y Similitud

### ¿Por qué es importante?

El clustering agrupa por **similitud**, que se mide inversamente con **distancia**:

- Menor distancia → Mayor similitud
- Mayor distancia → Menor similitud

### Distancia Euclidiana

La más común, mide la línea recta entre dos puntos:

$$d_{euclidean}(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

```python
import numpy as np

def euclidean_distance(x, y):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((x - y) ** 2))

# Ejemplo
a = np.array([1, 2])
b = np.array([4, 6])
print(euclidean_distance(a, b))  # 5.0
```

### Distancia Manhattan

Suma de diferencias absolutas (como caminar en una cuadrícula):

$$d_{manhattan}(x, y) = \sum_{i=1}^{n}|x_i - y_i|$$

```python
def manhattan_distance(x, y):
    """Calculate Manhattan distance between two points."""
    return np.sum(np.abs(x - y))

print(manhattan_distance(a, b))  # 7
```

### Similitud Coseno

Mide el ángulo entre vectores (útil para texto):

$$sim_{cosine}(x, y) = \frac{x \cdot y}{||x|| \cdot ||y||}$$

```python
def cosine_similarity(x, y):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return dot_product / (norm_x * norm_y)
```

### ¿Cuál usar?

| Distancia  | Cuándo usar                                    |
| ---------- | ---------------------------------------------- |
| Euclidiana | Datos numéricos continuos, escala similar      |
| Manhattan  | Datos con outliers, dimensiones independientes |
| Coseno     | Texto, datos de alta dimensionalidad           |

---

## ⚠️ Preprocesamiento para Clustering

### Normalización: Crucial para K-Means

K-Means usa distancias euclidianas, por lo que las features con mayor escala dominarán:

```python
from sklearn.preprocessing import StandardScaler

# Sin normalizar: edad (0-100) domina sobre ingresos (0-1M)
X_raw = np.array([
    [25, 50000],
    [30, 80000],
    [65, 45000]
])

# Normalizar siempre
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
```

### Reducción de Dimensionalidad

Para visualizar y mejorar clustering en alta dimensionalidad:

```python
from sklearn.decomposition import PCA

# Reducir a 2D para visualización
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

# O usar para mejorar el clustering
pca = PCA(n_components=0.95)  # Mantener 95% varianza
X_reduced = pca.fit_transform(X_scaled)
```

---

## 🎯 El Desafío del Clustering

### No hay "respuesta correcta"

A diferencia de la clasificación supervisada:

- No tenemos etiquetas para validar
- Múltiples agrupaciones pueden ser válidas
- La interpretación requiere conocimiento del dominio

### Preguntas clave

1. **¿Cuántos clusters?** - No siempre obvio
2. **¿Qué algoritmo usar?** - Depende de los datos
3. **¿Cómo evaluar calidad?** - Métricas intrínsecas vs extrínsecas
4. **¿Tienen sentido los clusters?** - Validación con expertos

---

## 🐍 Clustering en Scikit-learn

### API Consistente

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering

# Todos siguen el mismo patrón
model = KMeans(n_clusters=3)
labels = model.fit_predict(X)

# O en dos pasos
model.fit(X)
labels = model.labels_
```

### Atributos Comunes

```python
# Después de fit()
model.labels_          # Etiqueta de cluster para cada punto
model.cluster_centers_ # Centroides (K-Means)
model.inertia_         # Inercia (K-Means)
```

---

## ✅ Checklist de Verificación

- [ ] Entiendo la diferencia entre aprendizaje supervisado y no supervisado
- [ ] Puedo explicar qué es clustering y para qué sirve
- [ ] Conozco los tipos principales de algoritmos de clustering
- [ ] Comprendo el concepto de distancia y similitud
- [ ] Sé por qué es importante normalizar antes de clustering

---

## 📚 Recursos Adicionales

- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Comparison of Clustering Algorithms](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html)
- [The Elements of Statistical Learning - Chapter 14](https://web.stanford.edu/~hastie/ElemStatLearn/)

---

## 🔗 Navegación

| ⬅️ Anterior                          | 🏠 Inicio                 | Siguiente ➡️            |
| ------------------------------------ | ------------------------- | ----------------------- |
| [Semana 12](../../week-12/README.md) | [Semana 13](../README.md) | [K-Means](02-kmeans.md) |
