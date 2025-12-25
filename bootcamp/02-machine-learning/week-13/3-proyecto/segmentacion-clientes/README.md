# 🎯 Proyecto: Segmentación de Clientes

## 📋 Descripción

En este proyecto aplicarás los tres algoritmos de clustering (K-Means, DBSCAN, Jerárquico) para segmentar clientes de un e-commerce basándote en su comportamiento de compra.

**Contexto de negocio**: Una empresa de retail online necesita entender mejor a sus clientes para personalizar campañas de marketing y mejorar la retención.

---

## 🎯 Objetivos

- ✅ Aplicar técnicas de preprocesamiento para datos de clientes
- ✅ Implementar y comparar K-Means, DBSCAN y Clustering Jerárquico
- ✅ Evaluar clusters con métricas apropiadas
- ✅ Interpretar y describir cada segmento de clientes
- ✅ Generar recomendaciones de marketing basadas en segmentos

---

## 📊 Dataset

Utilizaremos características derivadas de compras de clientes:

| Feature            | Descripción                     |
| ------------------ | ------------------------------- |
| `recency`          | Días desde última compra        |
| `frequency`        | Número total de compras         |
| `monetary`         | Gasto total acumulado           |
| `avg_basket`       | Promedio de gasto por compra    |
| `purchase_variety` | Número de categorías diferentes |
| `tenure`           | Días desde primera compra       |

---

## 🔄 Flujo del Proyecto

### Fase 1: Preparación de Datos (30 min)

1. **Carga y exploración**

   - Cargar dataset
   - Estadísticas descriptivas
   - Distribución de variables

2. **Preprocesamiento**
   - Manejo de outliers
   - Normalización/Estandarización
   - Feature engineering adicional

### Fase 2: Clustering (45 min)

3. **K-Means**

   - Método del codo
   - Silhouette analysis
   - Selección de K óptimo

4. **DBSCAN**

   - K-distance graph para epsilon
   - Identificar clientes atípicos

5. **Clustering Jerárquico**
   - Dendrograma
   - Comparar linkages

### Fase 3: Evaluación (20 min)

6. **Comparar algoritmos**
   - Silhouette Score
   - Davies-Bouldin Index
   - Visualización 2D/3D

### Fase 4: Interpretación (25 min)

7. **Análisis de segmentos**

   - Caracterizar cada cluster
   - Nombrar segmentos
   - Visualizaciones por segmento

8. **Recomendaciones**
   - Estrategias de marketing por segmento
   - Acciones prioritarias

---

## 📁 Estructura del Proyecto

```
segmentacion-clientes/
├── README.md                    # Este archivo
├── starter/
│   └── main.py                  # Código base con TODOs
├── data/
│   └── README.md                # Instrucciones para generar datos
└── solution/
    └── main.py                  # Solución completa (opcional)
```

---

## 📝 Entregables

1. **Código completo** (`main.py`)

   - Todas las funciones implementadas
   - Código documentado
   - Resultados reproducibles

2. **Visualizaciones**

   - `eda_distributions.png` - Distribución de variables
   - `elbow_silhouette.png` - Selección de K
   - `clusters_comparison.png` - Comparación de algoritmos
   - `segment_profiles.png` - Perfiles de segmentos

3. **Informe de Segmentos** (en comentarios o print)
   - Descripción de cada segmento
   - Tamaño y características
   - Recomendaciones de marketing

---

## ✅ Criterios de Evaluación

### Conocimiento 🧠 (30%)

- [ ] Justificación de preprocesamiento
- [ ] Explicación de elección de K
- [ ] Interpretación correcta de métricas

### Desempeño 💪 (40%)

- [ ] Código funcional y completo
- [ ] Uso correcto de sklearn
- [ ] Visualizaciones claras

### Producto 📦 (30%)

- [ ] Segmentos bien caracterizados
- [ ] Recomendaciones de negocio relevantes
- [ ] Documentación clara

---

## 💡 Hints

1. **Preprocesamiento**

   ```python
   # Eliminar outliers extremos (> 3 std)
   from scipy import stats
   z_scores = np.abs(stats.zscore(X))
   X_clean = X[(z_scores < 3).all(axis=1)]
   ```

2. **Silhouette por cluster**

   ```python
   # Ver qué clusters son más cohesivos
   sample_scores = silhouette_samples(X, labels)
   for i in range(n_clusters):
       cluster_scores = sample_scores[labels == i]
       print(f"Cluster {i}: {cluster_scores.mean():.3f}")
   ```

3. **Caracterizar segmentos**
   ```python
   # Perfil de cada cluster
   for cluster in range(n_clusters):
       mask = labels == cluster
       print(f"\nCluster {cluster} ({mask.sum()} clientes)")
       print(df[mask].describe())
   ```

---

## 📚 Recursos

- [RFM Segmentation](<https://en.wikipedia.org/wiki/RFM_(market_research)>)
- [Customer Segmentation with Python](https://towardsdatascience.com/customer-segmentation-using-k-means-clustering-d33964f238c3)
- [Sklearn Customer Clustering Example](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html)

---

## ⏱️ Tiempo Estimado

- **Total**: 2 horas
- Distribución:
  - Fase 1: 30 min
  - Fase 2: 45 min
  - Fase 3: 20 min
  - Fase 4: 25 min

---

## 🔗 Navegación

| ⬅️ Prácticas                     | 🏠 Semana                  | Siguiente ➡️                  |
| -------------------------------- | -------------------------- | ----------------------------- |
| [Ejercicios](../../2-practicas/) | [Week 13](../../README.md) | [Recursos](../../4-recursos/) |
