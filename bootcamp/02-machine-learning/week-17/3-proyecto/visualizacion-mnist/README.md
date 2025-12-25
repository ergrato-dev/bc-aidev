# 📦 Proyecto: Visualización y Clasificación de MNIST

## 🎯 Objetivo

Crear un sistema completo de visualización y clasificación de dígitos escritos a mano usando técnicas de reducción dimensional.

---

## 📋 Descripción

En este proyecto aplicarás todo lo aprendido sobre reducción de dimensionalidad para:

1. **Visualizar** el dataset MNIST con diferentes técnicas
2. **Comparar** PCA, t-SNE y UMAP en términos de separación de clases
3. **Clasificar** dígitos usando reducción dimensional como preprocesamiento
4. **Optimizar** el pipeline completo

---

## 🎓 Especificaciones Técnicas

### Dataset

- **MNIST** (sklearn.datasets.load_digits): 1,797 imágenes de 8×8 pixels (64 features)
- Alternativamente: MNIST completo de keras (70,000 imágenes de 28×28)

### Requisitos Funcionales

1. **Visualización Comparativa**
   - Aplicar PCA, t-SNE y UMAP al dataset
   - Generar gráficos lado a lado de las 3 técnicas
   - Calcular y mostrar métricas (trustworthiness, tiempo)

2. **Análisis de Hiperparámetros**
   - Para PCA: n_components (varianza acumulada)
   - Para t-SNE: perplexity (5, 15, 30, 50)
   - Para UMAP: n_neighbors y min_dist

3. **Pipeline de Clasificación**
   - Comparar accuracy con/sin reducción
   - Encontrar número óptimo de componentes
   - Evaluar al menos 2 clasificadores diferentes

4. **Reporte Visual**
   - Dashboard con todos los resultados
   - Conclusiones sobre cuándo usar cada técnica

---

## 📁 Estructura del Proyecto

```
visualizacion-mnist/
├── README.md              # Este archivo
├── starter/
│   └── main.py            # Código inicial con TODOs
└── solution/
    └── main.py            # Solución completa
```

---

## ⏱️ Tiempo Estimado

2 horas

---

## 📊 Criterios de Evaluación

| Criterio | Puntos |
|----------|--------|
| Visualizaciones correctas (3 técnicas) | 25% |
| Análisis de hiperparámetros | 25% |
| Pipeline de clasificación funcional | 30% |
| Código limpio y documentado | 10% |
| Conclusiones y análisis | 10% |

---

## 🚀 Entregables

1. Archivo `main.py` completado
2. Al menos 4 visualizaciones generadas
3. Tabla comparativa de métricas
4. Conclusiones escritas como comentarios en el código

---

## 💡 Hints

- Usa `figsize` grande para visualizaciones legibles
- Recuerda escalar los datos antes de aplicar reducción
- t-SNE es lento; usa un subconjunto para experimentación rápida
- UMAP puede transformar nuevos datos (útil para el pipeline)
