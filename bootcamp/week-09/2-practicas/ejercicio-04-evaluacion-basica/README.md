# Ejercicio 04: Evaluación Básica de Modelos

## 🎯 Objetivo

Aprender a evaluar modelos de clasificación usando diferentes métricas: accuracy, precision, recall, F1-score y matriz de confusión.

## 📋 Descripción

La accuracy no siempre es suficiente para evaluar un modelo. En este ejercicio aprenderás métricas más completas que te ayudarán a entender el rendimiento real de tu modelo.

## 📚 Conceptos Clave

- **Accuracy**: Proporción de predicciones correctas
- **Precision**: De los que predije positivo, ¿cuántos eran realmente positivos?
- **Recall**: De los positivos reales, ¿cuántos encontré?
- **F1-Score**: Media armónica de precision y recall
- **Matriz de Confusión**: Tabla que muestra TP, TN, FP, FN

## 🛠️ Instrucciones

Abre `starter/main.py` y sigue los pasos descomentando el código indicado.

### Paso 1: Entrenar un Modelo

Preparar datos y entrenar un clasificador.

### Paso 2: Matriz de Confusión

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
```

### Paso 3: Métricas de Clasificación

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
```

### Paso 4: Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

## ✅ Resultado Esperado

- Matriz de confusión interpretable
- Accuracy, Precision, Recall y F1-Score calculados
- Classification report completo por clase

## 🔗 Recursos

- [Sklearn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
