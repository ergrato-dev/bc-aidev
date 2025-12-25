# 📉 Semana 17: Reducción de Dimensionalidad

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender el problema de la maldición de la dimensionalidad
- ✅ Implementar PCA (Análisis de Componentes Principales)
- ✅ Aplicar t-SNE para visualización de datos de alta dimensión
- ✅ Usar UMAP como alternativa moderna a t-SNE
- ✅ Seleccionar el número óptimo de componentes
- ✅ Combinar reducción dimensional con otros algoritmos de ML

---

## 📚 Requisitos Previos

- Álgebra lineal básica (vectores, matrices, autovalores)
- Estadística descriptiva (varianza, covarianza)
- Scikit-learn básico
- Clustering (semana 16)

---

## 🗂️ Estructura de la Semana

```
week-17/
├── README.md
├── rubrica-evaluacion.md
├── 0-assets/
│   ├── 01-curse-dimensionality.svg
│   ├── 02-pca-concept.svg
│   ├── 03-tsne-visualization.svg
│   └── 04-comparison-techniques.svg
├── 1-teoria/
│   ├── 01-intro-reduccion-dimensional.md
│   ├── 02-pca.md
│   ├── 03-tsne.md
│   └── 04-umap-comparacion.md
├── 2-practicas/
│   ├── ejercicio-01-pca/
│   ├── ejercicio-02-tsne/
│   ├── ejercicio-03-umap/
│   └── ejercicio-04-pipeline-completo/
├── 3-proyecto/
│   └── visualizacion-mnist/
├── 4-recursos/
│   └── README.md
└── 5-glosario/
    └── README.md
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| # | Tema | Archivo | Duración |
|---|------|---------|----------|
| 1 | Introducción y Maldición de Dimensionalidad | [01-intro-reduccion-dimensional.md](1-teoria/01-intro-reduccion-dimensional.md) | 20 min |
| 2 | PCA: Análisis de Componentes Principales | [02-pca.md](1-teoria/02-pca.md) | 30 min |
| 3 | t-SNE: Visualización No Lineal | [03-tsne.md](1-teoria/03-tsne.md) | 20 min |
| 4 | UMAP y Comparación de Técnicas | [04-umap-comparacion.md](1-teoria/04-umap-comparacion.md) | 20 min |

### 💻 Prácticas (2.5 horas)

| # | Ejercicio | Carpeta | Duración |
|---|-----------|---------|----------|
| 1 | PCA desde Cero y Sklearn | [ejercicio-01-pca/](2-practicas/ejercicio-01-pca/) | 40 min |
| 2 | t-SNE para Visualización | [ejercicio-02-tsne/](2-practicas/ejercicio-02-tsne/) | 35 min |
| 3 | UMAP y Comparaciones | [ejercicio-03-umap/](2-practicas/ejercicio-03-umap/) | 35 min |
| 4 | Pipeline Completo | [ejercicio-04-pipeline-completo/](2-practicas/ejercicio-04-pipeline-completo/) | 40 min |

### 🎯 Proyecto (2 horas)

| Proyecto | Descripción | Carpeta |
|----------|-------------|---------|
| Visualización MNIST | Visualizar y clasificar dígitos con reducción dimensional | [visualizacion-mnist/](3-proyecto/visualizacion-mnist/) |

---

## ⏱️ Distribución del Tiempo

```
Total: 6 horas

┌─────────────────────────────────────────────────────────────┐
│  📖 Teoría       │████████░░░░░░░░░░░░░░░░│  1.5h (25%)    │
│  💻 Prácticas    │████████████████░░░░░░░░│  2.5h (42%)    │
│  🎯 Proyecto     │████████████░░░░░░░░░░░░│  2.0h (33%)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📌 Entregables

1. **Ejercicios Completados**
   - [ ] PCA implementado desde cero y con sklearn
   - [ ] Visualizaciones t-SNE funcionando
   - [ ] Comparación UMAP vs t-SNE
   - [ ] Pipeline completo de reducción + clasificación

2. **Proyecto Semanal**
   - [ ] Visualización de MNIST con múltiples técnicas
   - [ ] Análisis de varianza explicada
   - [ ] Comparación de rendimiento en clasificación

---

## 🧠 Conceptos Clave

### Técnicas Lineales
- **PCA**: Maximiza varianza, proyección lineal, componentes ortogonales
- **LDA**: Maximiza separabilidad entre clases (supervisado)

### Técnicas No Lineales
- **t-SNE**: Preserva estructura local, bueno para visualización
- **UMAP**: Más rápido que t-SNE, preserva estructura global

### Métricas
- Varianza explicada (PCA)
- Preservación de vecindarios (t-SNE/UMAP)
- Trustworthiness y continuity

---

## 🔗 Navegación

| ⬅️ Anterior | 🏠 Módulo | Siguiente ➡️ |
|-------------|-----------|--------------|
| [Semana 16: Clustering](../week-16/README.md) | [Módulo 2: ML](../README.md) | [Semana 18: ML en Producción](../week-18/README.md) |

---

_Semana 17 de 36 | Módulo: Machine Learning | Bootcamp IA: Zero to Hero_
