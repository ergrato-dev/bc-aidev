# Ejercicio 02: Support Vector Machines

## 🎯 Objetivo

Implementar SVM con diferentes kernels y encontrar los mejores hiperparámetros.

## 📋 Instrucciones

### Paso 1: Cargar Dataset

Usamos Breast Cancer para clasificación binaria.

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

cancer = load_breast_cancer()
print(f"Features: {cancer.feature_names[:5]}...")
print(f"Clases: {cancer.target_names}")
print(f"Shape: {cancer.data.shape}")
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

### Paso 2: Preparar Datos

Split y normalización (esencial para SVM).

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Descomenta** la sección del Paso 2.

### Paso 3: SVM con Kernel Lineal

Probamos SVM lineal como baseline.

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)
y_pred = svm_linear.predict(X_test_scaled)

print(f"Linear SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Vectores de soporte: {svm_linear.n_support_}")
```

**Descomenta** la sección del Paso 3.

### Paso 4: Comparar Kernels

Comparamos linear, RBF y polynomial.

```python
from sklearn.model_selection import cross_val_score

kernels = ['linear', 'rbf', 'poly']

for kernel in kernels:
    svm = SVC(kernel=kernel)
    scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
    print(f"{kernel:8s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Descomenta** la sección del Paso 4.

### Paso 5: GridSearch para RBF

Buscamos C y gamma óptimos para kernel RBF.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_scaled, y_train)

print(f"Mejores parámetros: {grid.best_params_}")
print(f"Mejor CV score: {grid.best_score_:.4f}")
```

**Descomenta** la sección del Paso 5.

### Paso 6: Evaluación Final

Evaluamos el mejor modelo en test.

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = grid.predict(X_test_scaled)
print(f"Test Accuracy: {grid.score(X_test_scaled, y_test):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names, yticklabels=cancer.target_names)
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión - SVM')
plt.savefig('svm_confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.show()
```

**Descomenta** la sección del Paso 6.

## ✅ Resultado Esperado

- Comparación de kernels
- Mejores hiperparámetros encontrados
- Test accuracy ≥ 0.95
- Matriz de confusión guardada

## 🔗 Recursos

- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Kernel Functions](https://scikit-learn.org/stable/modules/svm.html#kernel-functions)
