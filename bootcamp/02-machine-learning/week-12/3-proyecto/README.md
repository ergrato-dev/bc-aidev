# Proyecto Semana 12: Clasificador de Spam

## 🎯 Objetivo

Construir un sistema de clasificación de spam comparando KNN, SVM y Naive Bayes, seleccionando el mejor modelo basado en métricas de rendimiento.

## 📋 Descripción

Implementarás un clasificador de spam/ham (no-spam) utilizando los tres algoritmos aprendidos esta semana. Compararás su rendimiento y seleccionarás el mejor modelo.

## 📊 Dataset

Usaremos el dataset SMS Spam Collection (incluido en sklearn o descargable).

## 🎯 Requisitos

### Funcionales

1. **Preprocesamiento de texto**

   - Vectorización con TF-IDF
   - Manejo de stop words

2. **Implementar los 3 algoritmos**

   - KNN con búsqueda de k óptimo
   - SVM con comparación de kernels
   - Naive Bayes (MultinomialNB)

3. **Comparación de modelos**

   - Cross-validation para cada modelo
   - Métricas: Accuracy, Precision, Recall, F1
   - Matriz de confusión

4. **Selección del mejor modelo**
   - Justificación basada en métricas
   - Análisis de trade-offs

### Técnicos

- Python 3.10+
- Accuracy mínimo: **≥ 0.90**
- Código documentado
- Visualizaciones de resultados

## 📁 Estructura

```
3-proyecto/
├── README.md
├── starter/
│   └── main.py          # Código con TODOs
└── .solution/
    └── main.py          # Solución (NO subir a git)
```

## 🚀 Instrucciones

1. Abre `starter/main.py`
2. Completa cada función marcada con `TODO`
3. Ejecuta y verifica que accuracy ≥ 0.90
4. Genera las visualizaciones

## ✅ Criterios de Evaluación

| Criterio                     | Puntos  |
| ---------------------------- | ------- |
| Preprocesamiento correcto    | 15      |
| KNN implementado y tuneado   | 20      |
| SVM implementado con kernels | 20      |
| Naive Bayes implementado     | 15      |
| Comparación con métricas     | 15      |
| Visualizaciones              | 10      |
| Código limpio y documentado  | 5       |
| **Total**                    | **100** |

## 📦 Entregables

1. `main.py` completado
2. Gráfico de comparación (`model_comparison.png`)
3. Matriz de confusión del mejor modelo (`best_model_cm.png`)
4. Reporte breve de selección de modelo (en comentarios o print)

## 🔗 Recursos

- [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Model Comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
