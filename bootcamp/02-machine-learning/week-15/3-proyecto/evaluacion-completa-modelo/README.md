# 🎯 Proyecto: Evaluación Completa de Modelo

## 📋 Descripción

Implementa una evaluación completa y rigurosa de un modelo de Machine Learning, aplicando todas las técnicas de validación y métricas aprendidas.

**Dataset**: Breast Cancer Wisconsin (diagnóstico de cáncer de mama)

---

## 🎯 Objetivos

1. Implementar pipeline completo de evaluación
2. Usar Nested CV para selección de hiperparámetros
3. Calcular múltiples métricas de clasificación
4. Generar visualizaciones profesionales
5. Reportar resultados con intervalos de confianza

---

## 📁 Estructura

```
evaluacion-completa-modelo/
├── README.md           # Este archivo
├── starter/
│   └── main.py         # Código inicial con TODOs
└── solution/
    └── main.py         # Solución completa
```

---

## 🔧 Requisitos

- Nested Cross-Validation (5×5)
- Comparación de al menos 3 modelos
- Métricas: Accuracy, Precision, Recall, F1, AUC-ROC, AP
- Curvas ROC y Precision-Recall
- Matriz de confusión del mejor modelo
- Análisis de importancia de features
- Reporte final con conclusiones

---

## 📊 Criterios de Evaluación

| Criterio | Puntos |
|----------|--------|
| Nested CV implementado correctamente | 6 |
| Comparación de múltiples modelos | 4 |
| Métricas de clasificación completas | 4 |
| Curvas ROC y PR generadas | 4 |
| Matriz de confusión e importancia de features | 4 |
| Reporte final con conclusiones | 4 |
| Código limpio y documentado | 4 |
| **Total** | **30** |

---

## 💡 Consejos

1. Empieza configurando el Nested CV correctamente
2. Usa `cross_validate` para obtener múltiples métricas
3. Documenta tus decisiones y hallazgos
4. Los intervalos de confianza dan credibilidad a los resultados

---

## 🔗 Recursos

- [Nested Cross-Validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
