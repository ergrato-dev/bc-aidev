# 📋 Rúbrica de Evaluación - Semana 17

## Reducción de Dimensionalidad

### 📊 Distribución de Puntos

| Componente | Puntos | Porcentaje |
|------------|--------|------------|
| Conocimiento (Teoría) | 30 | 30% |
| Desempeño (Prácticas) | 40 | 40% |
| Producto (Proyecto) | 30 | 30% |
| **Total** | **100** | **100%** |

---

## 🧠 Conocimiento (30 puntos)

### Conceptos Fundamentales (15 pts)

| Criterio | Excelente (15) | Bueno (12) | Suficiente (9) | Insuficiente (0-8) |
|----------|----------------|------------|----------------|-------------------|
| Maldición de dimensionalidad | Explica completamente el problema y sus consecuencias | Explica el concepto con algunos detalles | Comprensión básica | No comprende el concepto |
| Diferencia lineal vs no lineal | Distingue claramente ambos enfoques | Distingue con algunos errores | Distinción parcial | No distingue |

### Comprensión de Algoritmos (15 pts)

| Criterio | Excelente (15) | Bueno (12) | Suficiente (9) | Insuficiente (0-8) |
|----------|----------------|------------|----------------|-------------------|
| PCA | Explica autovalores, varianza explicada, proyección | Explica conceptos principales | Comprensión superficial | No comprende |
| t-SNE/UMAP | Comprende perplexity, neighbors, uso apropiado | Comprende uso básico | Conocimiento limitado | No comprende |

---

## 💪 Desempeño (40 puntos)

### Ejercicio 1: PCA (10 pts)

| Criterio | Puntos |
|----------|--------|
| PCA desde cero con numpy | 4 |
| PCA con sklearn | 2 |
| Selección de componentes (varianza) | 2 |
| Visualización de componentes | 2 |

### Ejercicio 2: t-SNE (10 pts)

| Criterio | Puntos |
|----------|--------|
| Implementación básica | 3 |
| Ajuste de perplexity | 3 |
| Visualización 2D/3D | 2 |
| Interpretación de resultados | 2 |

### Ejercicio 3: UMAP (10 pts)

| Criterio | Puntos |
|----------|--------|
| Implementación con umap-learn | 3 |
| Ajuste de n_neighbors y min_dist | 3 |
| Comparación con t-SNE | 2 |
| Análisis de tiempos | 2 |

### Ejercicio 4: Pipeline (10 pts)

| Criterio | Puntos |
|----------|--------|
| Pipeline reducción + clasificador | 4 |
| Comparación de rendimiento | 3 |
| Selección óptima de componentes | 3 |

---

## 📦 Producto (30 puntos)

### Proyecto: Visualización MNIST

| Criterio | Excelente (30) | Bueno (24) | Suficiente (18) | Insuficiente (0-17) |
|----------|----------------|------------|-----------------|---------------------|
| **Implementación** | PCA + t-SNE + UMAP correctos | Al menos 2 técnicas | Solo 1 técnica | Implementación incorrecta |
| **Visualizaciones** | Gráficos claros, coloreados, con leyendas | Gráficos correctos | Gráficos básicos | Sin visualizaciones |
| **Análisis** | Comparación completa de técnicas | Análisis parcial | Análisis mínimo | Sin análisis |
| **Clasificación** | Pipeline con métricas comparativas | Clasificador básico | Sin clasificador | No funciona |
| **Documentación** | Código comentado, README completo | Documentación parcial | Mínima documentación | Sin documentación |

---

## 📝 Criterios de Aprobación

- ✅ Mínimo **70%** en cada componente
- ✅ Todos los ejercicios ejecutables
- ✅ Proyecto funcional con al menos 2 técnicas
- ✅ Visualizaciones interpretables

---

## 🎯 Rúbrica Detallada del Proyecto

| Aspecto | Puntos | Descripción |
|---------|--------|-------------|
| PCA implementado | 5 | Varianza explicada, scree plot |
| t-SNE visualización | 5 | Perplexity ajustado, clusters visibles |
| UMAP visualización | 5 | Parámetros ajustados, comparación |
| Comparación tiempos | 3 | Benchmark de las 3 técnicas |
| Clasificación pre/post | 5 | Accuracy con/sin reducción |
| Interpretación | 4 | Análisis de resultados |
| Código limpio | 3 | Documentado, modular |
| **Total** | **30** | |

---

## 📅 Fecha de Entrega

- **Ejercicios**: Durante la semana
- **Proyecto**: Fin de semana 17

---

_Rúbrica Semana 17 | Bootcamp IA: Zero to Hero_
