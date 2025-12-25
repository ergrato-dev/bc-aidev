# 📚 Recursos - Semana 13: Clustering

## 📖 eBooks Gratuitos

### Machine Learning y Clustering

1. **"Introduction to Statistical Learning"** - James, Witten, Hastie, Tibshirani

   - Capítulo 10: Unsupervised Learning
   - [Enlace oficial (PDF gratis)](https://www.statlearning.com/)
   - 🌟 Excelente introducción teórica con ejemplos en R

2. **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman

   - Capítulo 14: Unsupervised Learning
   - [Enlace oficial (PDF gratis)](https://hastie.su.domains/ElemStatLearn/)
   - 🌟 Más avanzado, fundamentos matemáticos sólidos

3. **"Python Data Science Handbook"** - Jake VanderPlas

   - Capítulo sobre K-Means y clustering
   - [Leer online gratis](https://jakevdp.github.io/PythonDataScienceHandbook/)
   - 🌟 Enfoque práctico con código Python

4. **"Scikit-learn User Guide"** - Clustering Section
   - Documentación oficial completa
   - [Enlace](https://scikit-learn.org/stable/modules/clustering.html)
   - 🌟 Referencia técnica definitiva

---

## 🎥 Videografía

### Canales de YouTube Recomendados

| Video                   | Canal     | Duración | Idioma |
| ----------------------- | --------- | -------- | ------ |
| K-Means Clustering      | StatQuest | 15 min   | 🇬🇧     |
| DBSCAN Explained        | StatQuest | 12 min   | 🇬🇧     |
| Hierarchical Clustering | StatQuest | 18 min   | 🇬🇧     |
| Clustering en Español   | Dot CSV   | 20 min   | 🇪🇸     |
| Customer Segmentation   | Ken Jee   | 45 min   | 🇬🇧     |

### Playlists Recomendadas

1. **StatQuest - Clustering**

   - [Playlist completa](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)
   - Explicaciones visuales excelentes

2. **Sentdex - Machine Learning con Python**

   - [K-Means section](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)
   - Código paso a paso

3. **Andrew Ng - Coursera**
   - Semana de Clustering en ML Course
   - Fundamentos sólidos

---

## 🔗 Webgrafía

### Documentación Oficial

| Recurso                 | Descripción                    | Enlace                                                                                |
| ----------------------- | ------------------------------ | ------------------------------------------------------------------------------------- |
| Scikit-learn Clustering | Guía completa de algoritmos    | [Link](https://scikit-learn.org/stable/modules/clustering.html)                       |
| SciPy Hierarchical      | Documentación de scipy.cluster | [Link](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)             |
| Sklearn Metrics         | Métricas de clustering         | [Link](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation) |

### Tutoriales y Artículos

| Título                                  | Fuente               | Nivel  |
| --------------------------------------- | -------------------- | ------ |
| Clustering Algorithms Overview          | Towards Data Science | ⭐⭐   |
| K-Means from Scratch                    | Real Python          | ⭐⭐   |
| DBSCAN: A Practical Guide               | Analytics Vidhya     | ⭐⭐   |
| Customer Segmentation Guide             | Kaggle               | ⭐⭐⭐ |
| Choosing the Right Clustering Algorithm | Google Cloud         | ⭐⭐   |

### Visualizaciones Interactivas

1. **K-Means Visualization**

   - [Naftali Harris](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
   - Excelente para entender el algoritmo

2. **DBSCAN Visualization**

   - [Same author](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)
   - Ver cómo DBSCAN detecta clusters

3. **Hierarchical Clustering**
   - [Visualgo](https://visualgo.net/en/dfsbfs)
   - Visualización de estructuras de datos

---

## 📊 Datasets para Práctica

### Datasets Clásicos

1. **Iris Dataset** (incluido en sklearn)

   - 150 muestras, 4 features, 3 clusters
   - Ideal para primeras pruebas

2. **Mall Customers** (Kaggle)

   - Segmentación de clientes de mall
   - [Descargar](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

3. **Wine Dataset** (incluido en sklearn)
   - 178 muestras, 13 features
   - Buenos para clustering no trivial

### Datasets Avanzados

1. **Wholesale Customers** (UCI)

   - Datos de ventas mayoristas
   - [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers)

2. **Credit Card Dataset** (Kaggle)

   - Segmentación de usuarios de tarjetas
   - [Descargar](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

3. **Online Retail** (UCI)
   - Transacciones de e-commerce
   - Ideal para RFM analysis

---

## 🛠️ Herramientas Útiles

### Librerías Python

```python
# Core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Métricas
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Hierarchical
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Preprocessing
from sklearn.preprocessing import StandardScaler
```

### Extensiones de VS Code

| Extensión     | Propósito              |
| ------------- | ---------------------- |
| Python        | Soporte Python         |
| Jupyter       | Notebooks              |
| Data Wrangler | Visualización de datos |
| Rainbow CSV   | Ver archivos CSV       |

---

## 📝 Papers Fundamentales

1. **K-Means Original** (1967)

   - MacQueen, J. "Some methods for classification and analysis of multivariate observations"

2. **DBSCAN Original** (1996)

   - Ester, M., et al. "A density-based algorithm for discovering clusters"
   - [PDF](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)

3. **Ward's Method** (1963)
   - Ward, J. H. "Hierarchical grouping to optimize an objective function"

---

## 🔗 Navegación

| ⬅️ Proyecto                   | 🏠 Semana                  | Siguiente ➡️                  |
| ----------------------------- | -------------------------- | ----------------------------- |
| [Proyecto](../../3-proyecto/) | [Week 13](../../README.md) | [Glosario](../../5-glosario/) |
