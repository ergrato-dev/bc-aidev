# 📊 Ejercicio 02: Métricas de Clasificación

## 🎯 Objetivo

Comprender y aplicar las principales métricas de clasificación: matriz de confusión, precision, recall, F1 y curvas ROC/PR.

---

## 📋 Descripción

En este ejercicio aprenderás a:
1. Calcular e interpretar la matriz de confusión
2. Entender cuándo usar precision vs recall
3. Generar curvas ROC y Precision-Recall
4. Ajustar umbrales de decisión

---

## 📁 Archivos

- `starter/main.py` - Código inicial para descomentar
- `solution/main.py` - Solución completa

---

## 🔨 Pasos

### Paso 1: Preparar Datos y Modelo

Creamos un dataset de clasificación binaria y entrenamos un modelo.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Matriz de Confusión

Calculamos y visualizamos la matriz de confusión.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
# Extraer TN, FP, FN, TP
tn, fp, fn, tp = cm.ravel()
```

**Abre `starter/main.py`** y descomenta la sección del Paso 2.

---

### Paso 3: Métricas Básicas

Calculamos accuracy, precision, recall y F1.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
```

**Abre `starter/main.py`** y descomenta la sección del Paso 3.

---

### Paso 4: Classification Report

Obtenemos un resumen completo de métricas.

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

**Abre `starter/main.py`** y descomenta la sección del Paso 4.

---

### Paso 5: Curva ROC y AUC

Generamos la curva ROC y calculamos el área bajo la curva.

```python
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 5.

---

### Paso 6: Curva Precision-Recall

Generamos la curva PR para análisis más detallado.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 6.

---

### Paso 7: Ajustar Umbral de Decisión

Exploramos cómo cambian las métricas con diferentes umbrales.

```python
for threshold in [0.3, 0.5, 0.7]:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    # Calcular métricas
```

**Abre `starter/main.py`** y descomenta la sección del Paso 7.

---

### Paso 8: Dataset Desbalanceado

Aplicamos métricas en un escenario con clases desbalanceadas.

```python
X_imb, y_imb = make_classification(weights=[0.9, 0.1], ...)
# Comparar accuracy vs F1 vs PR-AUC
```

**Abre `starter/main.py`** y descomenta la sección del Paso 8.

---

## ✅ Criterios de Evaluación

| Criterio | Puntos |
|----------|--------|
| Matriz de confusión calculada e interpretada | 2 |
| Métricas básicas correctas | 2 |
| Curva ROC generada | 2 |
| Curva PR generada | 2 |
| Análisis de dataset desbalanceado | 2 |
| **Total** | **10** |

---

## 🔗 Recursos

- [Classification metrics scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
