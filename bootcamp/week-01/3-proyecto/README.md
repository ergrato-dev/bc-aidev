# 🚀 Proyecto Semana 01: Calculadora de Métricas ML

## 🎯 Objetivo

Crear una **calculadora de métricas de Machine Learning** que permita evaluar el rendimiento de modelos de clasificación binaria.

---

## 📋 Descripción

Desarrollarás un programa que:

1. Reciba datos de predicciones (valores reales vs predichos)
2. Calcule métricas fundamentales de ML
3. Genere un reporte de evaluación
4. Clasifique el rendimiento del modelo

---

## 📊 Métricas a Implementar

![Matriz de Confusión](../0-assets/03-matriz-confusion.svg)

| Métrica       | Fórmula               | Descripción                                       |
| ------------- | --------------------- | ------------------------------------------------- |
| **Accuracy**  | (TP + TN) / Total     | Proporción de predicciones correctas              |
| **Precision** | TP / (TP + FP)        | De los positivos predichos, cuántos son correctos |
| **Recall**    | TP / (TP + FN)        | De los positivos reales, cuántos se detectaron    |
| **F1-Score**  | 2 × (P × R) / (P + R) | Media armónica de Precision y Recall              |

Donde:

- **TP** (True Positives): Predichos positivos que son realmente positivos
- **TN** (True Negatives): Predichos negativos que son realmente negativos
- **FP** (False Positives): Predichos positivos que son realmente negativos
- **FN** (False Negatives): Predichos negativos que son realmente positivos

---

## 📁 Estructura del Proyecto

```
3-proyecto/
├── README.md           # Este archivo
├── starter/
│   └── main.py         # Plantilla con TODOs
└── .solution/          # Carpeta oculta
    └── main.py         # Solución de referencia
```

---

## 📋 Instrucciones

### 1. Abre `starter/main.py`

El archivo contiene:

- Estructura del programa
- Datos de prueba
- Funciones con TODOs para implementar

### 2. Implementa las funciones

Completa cada función siguiendo las instrucciones en los comentarios.

### 3. Ejecuta y verifica

```bash
python starter/main.py
```

### 4. Compara con la solución

Si tienes dudas, revisa `solution/main.py`.

---

## ✅ Criterios de Evaluación

| Criterio                                          | Puntos  |
| ------------------------------------------------- | ------- |
| `count_confusion_matrix()` funciona correctamente | 20      |
| `calculate_accuracy()` implementado               | 15      |
| `calculate_precision()` implementado              | 15      |
| `calculate_recall()` implementado                 | 15      |
| `calculate_f1_score()` implementado               | 15      |
| `classify_model()` con clasificación correcta     | 10      |
| `generate_report()` genera reporte completo       | 10      |
| **Total**                                         | **100** |

---

## 🎯 Salida Esperada

```
============================================================
🤖 CALCULADORA DE MÉTRICAS ML
============================================================

--- Matriz de Confusión ---
TP (True Positives): 45
TN (True Negatives): 40
FP (False Positives): 5
FN (False Negatives): 10

--- Métricas Calculadas ---
Accuracy:  0.85
Precision: 0.90
Recall:    0.82
F1-Score:  0.86

--- Clasificación del Modelo ---
✅ Bueno

============================================================
📊 REPORTE DE EVALUACIÓN
============================================================
El modelo tiene un accuracy de 85.0%.
Con precision de 90.0% y recall de 82.0%.
F1-Score: 0.86
Clasificación: ✅ Bueno
Recomendación: Modelo apto para uso en producción con monitoreo.
============================================================
```

---

## 💡 Tips

1. **Empieza por la matriz de confusión** - Es la base de todas las métricas
2. **Prueba cada función** por separado antes de continuar
3. **Cuidado con la división por cero** - Considera casos edge
4. **Usa f-strings** para formatear la salida

---

## 🏆 Reto Extra (Opcional)

Si terminas antes, intenta:

1. **Agregar Specificity**: TN / (TN + FP)
2. **Múltiples modelos**: Comparar varios conjuntos de predicciones
3. **Visualización ASCII**: Mostrar matriz de confusión como tabla

---

## 📚 Recursos

- [Matriz de Confusión - Wikipedia](https://es.wikipedia.org/wiki/Matriz_de_confusi%C3%B3n)
- [Métricas de Clasificación - Scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

_Volver a: [Semana 01](../README.md)_
