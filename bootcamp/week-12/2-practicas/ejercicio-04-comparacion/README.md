# Ejercicio 04: Comparación de Algoritmos

## 🎯 Objetivo

Comparar KNN, SVM y Naive Bayes en el mismo dataset para elegir el mejor modelo.

## 📋 Instrucciones

### Paso 1: Cargar Dataset

Usamos Wine dataset para multiclase.

```python
from sklearn.datasets import load_wine

wine = load_wine()
print(f"Features: {wine.feature_names}")
print(f"Clases: {wine.target_names}")
print(f"Shape: {wine.data.shape}")
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

### Paso 2: Preparar Datos

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
)
```

**Descomenta** la sección del Paso 2.

### Paso 3: Definir Modelos

Creamos pipelines para cada algoritmo.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

models = {
    'KNN': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=5))]),
    'SVM-RBF': Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf'))]),
    'SVM-Linear': Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear'))]),
    'Naive Bayes': GaussianNB()
}
```

**Descomenta** la sección del Paso 3.

### Paso 4: Comparar con Cross-Validation

```python
from sklearn.model_selection import cross_val_score
import time

results = {}
for name, model in models.items():
    start = time.time()
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    train_time = time.time() - start
    results[name] = {'cv_mean': scores.mean(), 'cv_std': scores.std(), 'time': train_time}
    print(f"{name:12s}: {scores.mean():.4f} ± {scores.std():.4f} ({train_time:.3f}s)")
```

**Descomenta** la sección del Paso 4.

### Paso 5: Evaluación en Test

```python
from sklearn.metrics import accuracy_score

print("\nResultados en Test:")
for name, model in models.items():
    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    results[name]['test_acc'] = test_acc
    print(f"{name:12s}: {test_acc:.4f}")
```

**Descomenta** la sección del Paso 5.

### Paso 6: Visualizar Comparación

```python
import matplotlib.pyplot as plt

names = list(results.keys())
cv_scores = [results[n]['cv_mean'] for n in names]
test_scores = [results[n]['test_acc'] for n in names]

x = range(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar([i - width/2 for i in x], cv_scores, width, label='CV Score')
bars2 = ax.bar([i + width/2 for i in x], test_scores, width, label='Test Score')

ax.set_ylabel('Accuracy')
ax.set_title('Comparación de Algoritmos')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()
ax.set_ylim(0.8, 1.0)

plt.tight_layout()
plt.savefig('algorithm_comparison.png', dpi=100)
plt.show()
```

**Descomenta** la sección del Paso 6.

## ✅ Resultado Esperado

- Tabla comparativa de los 4 modelos
- Gráfico de barras comparativo
- Identificación del mejor modelo para este dataset

## 🔗 Recursos

- [Model Selection](https://scikit-learn.org/stable/model_selection.html)
- [Comparing Classifiers](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
