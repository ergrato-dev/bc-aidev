# 📚 Introducción al Clustering

## 🎯 Objetivos

- Entender qué es el aprendizaje no supervisado
- Conocer las aplicaciones del clustering
- Diferenciar tipos de algoritmos de clustering

---

## 1. Aprendizaje Supervisado vs No Supervisado

### Supervisado
- **Datos**: Features (X) + Etiquetas (y)
- **Objetivo**: Predecir etiquetas para nuevos datos
- **Ejemplos**: Clasificación, Regresión

### No Supervisado
- **Datos**: Solo features (X), sin etiquetas
- **Objetivo**: Encontrar estructura/patrones ocultos
- **Ejemplos**: Clustering, Reducción dimensional

```python
# Supervisado: tenemos las etiquetas
X_train, y_train  # y_train = [0, 1, 1, 0, ...]

# No supervisado: solo features
X_data  # Sin etiquetas, buscamos estructura
```

---

## 2. ¿Qué es Clustering?

**Clustering** es la tarea de agrupar objetos similares en grupos llamados **clusters**.

### Intuición
- Puntos dentro del mismo cluster son **similares** entre sí
- Puntos en diferentes clusters son **diferentes** entre sí

![Clustering Overview](../0-assets/01-clustering-overview.svg)

### Características
- No hay "respuesta correcta" predefinida
- Es exploratorio: descubrimos estructura
- La calidad depende de cómo definimos "similitud"

---

## 3. Aplicaciones del Clustering

| Dominio | Aplicación |
|---------|------------|
| **Marketing** | Segmentación de clientes |
| **Biología** | Agrupación de genes, taxonomía |
| **Documentos** | Organización de noticias, temas |
| **Imágenes** | Compresión, segmentación |
| **Anomalías** | Detección de fraude, outliers |
| **Redes** | Detección de comunidades |

### Ejemplo: Segmentación de Clientes

```python
# Datos de clientes (sin etiquetas)
# Features: edad, ingresos, gastos, frecuencia de compra

# Clustering encuentra grupos naturales:
# - Cluster 0: Jóvenes alto consumo digital
# - Cluster 1: Familias tradicionales
# - Cluster 2: Seniors bajo consumo
# - Cluster 3: Profesionales alto ingreso
```

---

## 4. Tipos de Algoritmos de Clustering

### 4.1 Basados en Partición
- Dividen datos en K grupos no superpuestos
- **Ejemplos**: K-Means, K-Medoids
- **Característica**: Requieren especificar K

### 4.2 Basados en Densidad
- Grupos = regiones de alta densidad
- **Ejemplos**: DBSCAN, OPTICS, HDBSCAN
- **Característica**: Detectan outliers, formas arbitrarias

### 4.3 Jerárquicos
- Crean jerarquía de clusters (dendrograma)
- **Ejemplos**: Aglomerativo, Divisivo
- **Característica**: No requieren K a priori

### 4.4 Basados en Modelo
- Asumen distribución de probabilidad
- **Ejemplos**: Gaussian Mixture Models (GMM)
- **Característica**: Soft clustering (probabilidades)

---

## 5. Medidas de Distancia

El clustering depende de cómo medimos "similitud" (distancia):

### Distancia Euclidiana
```python
import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Ejemplo
a = np.array([1, 2])
b = np.array([4, 6])
print(euclidean_distance(a, b))  # 5.0
```

### Otras Distancias
```python
from scipy.spatial.distance import cdist

# Manhattan (L1)
dist_manhattan = cdist(X, Y, metric='cityblock')

# Coseno (para texto, vectores dispersos)
dist_cosine = cdist(X, Y, metric='cosine')

# Correlación
dist_corr = cdist(X, Y, metric='correlation')
```

### ⚠️ Importancia del Escalado

```python
from sklearn.preprocessing import StandardScaler

# SIEMPRE escalar antes de clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sin escalar: features con mayor magnitud dominan
# Con escalar: todas las features contribuyen igual
```

---

## 6. Scikit-learn: Clustering API

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Todos siguen el mismo patrón:
# 1. Crear modelo
model = KMeans(n_clusters=3, random_state=42)

# 2. Ajustar a los datos
model.fit(X_scaled)

# 3. Obtener etiquetas de cluster
labels = model.labels_

# O en un solo paso:
labels = model.fit_predict(X_scaled)
```

---

## 7. Visualización Básica

```python
import matplotlib.pyplot as plt

def plot_clusters(X, labels, title="Clusters"):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()

# Uso
plot_clusters(X_scaled, labels, "K-Means Clustering")
```

---

## ✅ Checklist de Aprendizaje

- [ ] Entiendo la diferencia entre supervisado y no supervisado
- [ ] Conozco qué es clustering y sus aplicaciones
- [ ] Identifico los tipos principales de algoritmos
- [ ] Comprendo la importancia del escalado
- [ ] Sé usar la API básica de sklearn para clustering

---

## 🔗 Referencias

- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Introduction to Clustering - Google ML](https://developers.google.com/machine-learning/clustering)
