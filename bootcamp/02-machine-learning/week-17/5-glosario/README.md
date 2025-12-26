# 📖 Glosario - Semana 17: Reducción de Dimensionalidad

## A

### Autovalor (Eigenvalue)

Escalar $\lambda$ que satisface $Av = \lambda v$ para una matriz $A$ y vector $v$. En PCA, representa la varianza capturada por cada componente principal.

### Autovector (Eigenvector)

Vector $v$ que satisface $Av = \lambda v$. En PCA, los autovectores de la matriz de covarianza definen las direcciones de los componentes principales.

## B

### Biplot

Visualización que muestra simultáneamente las observaciones proyectadas y las contribuciones de las variables originales en el espacio de componentes principales.

## C

### Componente Principal (Principal Component)

Nueva variable creada como combinación lineal de las variables originales, orientada en la dirección de máxima varianza.

### Covarianza

Medida de cómo dos variables varían juntas. La matriz de covarianza es fundamental para calcular PCA.

$$\text{Cov}(X, Y) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$

### Curse of Dimensionality (Maldición de la Dimensionalidad)

Fenómeno donde los datos se vuelven escasos en espacios de alta dimensión, dificultando el análisis y la clasificación.

## D

### Distancia Euclidiana

Distancia en línea recta entre dos puntos:

$$d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$

## E

### Embedding

Representación de datos de alta dimensión en un espacio de menor dimensión que preserva alguna estructura del espacio original.

### Explained Variance Ratio

Proporción de varianza total explicada por cada componente principal. Suma 1.0 para todos los componentes.

## F

### Feature Extraction

Técnica que transforma las características originales en nuevas características de menor dimensión (ej: PCA, t-SNE).

### Feature Selection

Técnica que selecciona un subconjunto de las características originales sin transformarlas.

## G

### Gradient Descent

Algoritmo de optimización usado por t-SNE y UMAP para minimizar la función de costo y encontrar el embedding óptimo.

## K

### KL Divergence (Kullback-Leibler Divergence)

Medida de qué tan diferente es una distribución de probabilidad de otra. t-SNE minimiza la KL divergence entre las distribuciones de alta y baja dimensión.

$$KL(P||Q) = \sum_i P(i) \log\frac{P(i)}{Q(i)}$$

## L

### Lineal

PCA es una técnica lineal porque los componentes son combinaciones lineales de las variables originales.

### Local vs Global Structure

- **Local**: Relaciones entre puntos cercanos (preservada por t-SNE)
- **Global**: Estructura general del dataset (mejor preservada por PCA y UMAP)

## M

### Manifold

Superficie de baja dimensión embebida en un espacio de alta dimensión. t-SNE y UMAP asumen que los datos yacen en un manifold.

### min_dist (UMAP)

Parámetro que controla qué tan compactos son los clusters en el embedding. Valores bajos = clusters más densos.

## N

### n_components

Número de dimensiones objetivo en la reducción dimensional.

### n_neighbors (UMAP)

Número de vecinos cercanos usados para construir el grafo de vecindad. Controla el balance entre estructura local y global.

### Non-linear

Técnicas como t-SNE y UMAP que pueden capturar relaciones no lineales entre los datos.

## O

### Out-of-sample

Capacidad de transformar nuevos datos no vistos durante el entrenamiento. PCA y UMAP lo soportan; t-SNE no.

## P

### PCA (Principal Component Analysis)

Técnica lineal que encuentra las direcciones de máxima varianza en los datos y proyecta sobre ellas.

### Perplexity

Parámetro de t-SNE que controla el número efectivo de vecinos. Típicamente entre 5 y 50. Afecta el balance entre estructura local y global.

### Preservación de Distancias

Propiedad de mantener las relaciones de distancia del espacio original en el espacio reducido.

## R

### Reconstrucción

Proceso de transformar datos reducidos de vuelta al espacio original. En PCA: $X_{rec} = X_{pca} \cdot V^T + \mu$.

### Reducción de Dimensionalidad

Proceso de reducir el número de características de un dataset mientras se preserva información relevante.

## S

### Scree Plot

Gráfico que muestra la varianza explicada por cada componente principal. Usado para decidir cuántos componentes retener.

### SVD (Singular Value Decomposition)

Descomposición matricial usada internamente por PCA para mayor estabilidad numérica.

$$A = U\Sigma V^T$$

## T

### t-Distribution (Distribución t de Student)

Distribución de probabilidad usada en t-SNE para el espacio de baja dimensión. Tiene colas más pesadas que la gaussiana, lo que ayuda a separar clusters.

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

Técnica no lineal para visualización que preserva estructura local de los datos.

### Trustworthiness

Métrica que evalúa qué tan bien se preservan las vecindades locales en el embedding. Valor entre 0 y 1; más alto es mejor.

## U

### UMAP (Uniform Manifold Approximation and Projection)

Técnica no lineal moderna que combina buena preservación de estructura local y global, con mejor rendimiento que t-SNE.

## V

### Varianza

Medida de dispersión de los datos:

$$\text{Var}(X) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

### Varianza Acumulada

Suma acumulativa de varianza explicada. Usada para determinar cuántos componentes retener (ej: 95% de varianza).

---

## 📊 Comparación Rápida

| Técnica   | Tipo      | Velocidad     | Nuevos Datos | Mejor Para                          |
| --------- | --------- | ------------- | ------------ | ----------------------------------- |
| **PCA**   | Lineal    | ⚡ Muy rápido | ✅ Sí        | Preprocesamiento, interpretabilidad |
| **t-SNE** | No lineal | 🐢 Lento      | ❌ No        | Visualización 2D/3D                 |
| **UMAP**  | No lineal | ⚡ Rápido     | ✅ Sí        | Visualización + pipelines ML        |

---

## 🔗 Navegación

| ⬅️ Anterior                         | 🏠 Semana                 | Siguiente ➡️                         |
| ----------------------------------- | ------------------------- | ------------------------------------ |
| [Recursos](../4-recursos/README.md) | [Semana 17](../README.md) | [Semana 18](../../week-18/README.md) |
