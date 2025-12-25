# Ejercicio 03: Regresión Logística

## 🎯 Objetivo

Implementar regresión logística para clasificación binaria (aprobado/reprobado).

## 📋 Conceptos Cubiertos

- Diferencia entre regresión y clasificación
- Función sigmoide y probabilidades
- Umbral de decisión (threshold)
- Métricas: accuracy, precision, recall, F1

## 🛠️ Requisitos

```bash
pip install numpy pandas matplotlib scikit-learn
```

## 📝 Instrucciones

Sigue los pasos en orden, descomentando el código en `starter/main.py`.

---

### Paso 1: Crear Dataset de Clasificación

Estudiantes con horas de estudio y si aprobaron (1) o no (0):

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

horas_estudio = np.random.uniform(0, 10, n)
# Probabilidad de aprobar aumenta con horas de estudio
prob_aprobar = 1 / (1 + np.exp(-(horas_estudio - 5)))
aprobado = (np.random.random(n) < prob_aprobar).astype(int)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Visualizar la Distribución

```python
import matplotlib.pyplot as plt

plt.scatter(horas_estudio[aprobado==0], aprobado[aprobado==0],
            label='Reprobado', alpha=0.5)
plt.scatter(horas_estudio[aprobado==1], aprobado[aprobado==1],
            label='Aprobado', alpha=0.5)
plt.xlabel('Horas de estudio')
plt.ylabel('Resultado (0/1)')
plt.legend()
plt.show()
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Entrenar Regresión Logística

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = horas_estudio.reshape(-1, 1)
y = aprobado

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Entender Probabilidades

```python
# Predicción de probabilidades
probs = model.predict_proba(X_test)
print("Probabilidades [P(0), P(1)]:")
print(probs[:5])

# Predicción de clases (threshold=0.5 por defecto)
preds = model.predict(X_test)
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Evaluar con Múltiples Métricas

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Matriz de Confusión

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=['Reprobado', 'Aprobado'])
disp.plot(cmap='Blues')
plt.title('Matriz de Confusión')
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
```

**Descomenta** la sección del Paso 6.

---

### Paso 7: Visualizar Curva Sigmoide

```python
# Crear rango continuo para visualizar
X_range = np.linspace(0, 10, 100).reshape(-1, 1)
probs_range = model.predict_proba(X_range)[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.5, label='Datos reales')
plt.plot(X_range, probs_range, color='red', linewidth=2,
         label='P(aprobar)')
plt.axhline(y=0.5, color='gray', linestyle='--', label='Threshold=0.5')
plt.xlabel('Horas de estudio')
plt.ylabel('Probabilidad')
plt.legend()
plt.title('Regresión Logística: Curva Sigmoide')
plt.savefig('curva_sigmoide.png', dpi=150)
plt.show()
```

**Descomenta** la sección del Paso 7.

---

### Paso 8: Cambiar Threshold

```python
# Con threshold=0.5 (default)
preds_50 = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)

# Con threshold=0.7 (más conservador)
preds_70 = (model.predict_proba(X_test)[:, 1] >= 0.7).astype(int)

print(f"Threshold 0.5: {accuracy_score(y_test, preds_50):.4f}")
print(f"Threshold 0.7: {accuracy_score(y_test, preds_70):.4f}")
```

**Descomenta** la sección del Paso 8.

---

## ✅ Resultado Esperado

```
Accuracy: ~0.85
Precision: ~0.85
Recall: ~0.88
F1-Score: ~0.86
```

---

## 📚 Recursos

- [LogisticRegression - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Métricas de clasificación - sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
