# Comparación de Algoritmos: KNN vs SVM vs Naive Bayes

## 🎯 Objetivos

- Comparar características de los tres algoritmos
- Saber cuándo usar cada uno
- Implementar comparación práctica
- Tomar decisiones informadas de selección de modelo

## 📋 Contenido

### 1. Tabla Comparativa

![Comparación Algoritmos](../0-assets/05-comparacion-algoritmos.svg)

| Criterio           | KNN            | SVM            | Naive Bayes       |
| ------------------ | -------------- | -------------- | ----------------- |
| **Tipo**           | Instance-based | Margin-based   | Probabilístico    |
| **Entrenamiento**  | Ninguno (lazy) | Lento O(n²-n³) | Muy rápido O(n)   |
| **Predicción**     | Lento O(n)     | Rápido         | Muy rápido        |
| **Memoria**        | Guarda todo    | Solo SVs       | Solo parámetros   |
| **No lineal**      | Natural        | Con kernels    | No (asume lineal) |
| **Interpretable**  | Moderado       | Bajo           | Alto              |
| **Probabilidades** | No nativo      | Con overhead   | Nativo            |

### 2. Cuándo Usar Cada Algoritmo

#### KNN es mejor cuando:

- Dataset pequeño (< 10k muestras)
- Necesitas un baseline rápido
- Los datos tienen estructura local
- No hay muchas features

```python
# Buen caso para KNN: datos pequeños, pocas features
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
```

#### SVM es mejor cuando:

- Alta dimensionalidad (muchas features)
- Hay margen claro entre clases
- Necesitas alto rendimiento
- Datos no son linealmente separables (usar RBF)

```python
# Buen caso para SVM: alta dimensionalidad
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
```

#### Naive Bayes es mejor cuando:

- Clasificación de texto
- Necesitas probabilidades
- Dataset muy grande
- Entrenamiento/predicción debe ser rápido

```python
# Buen caso para NB: clasificación de texto
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=0.1)
```

### 3. Comparación Práctica

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import time
import numpy as np

# Cargar datos
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Definir modelos
models = {
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ]),
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', gamma='scale'))
    ]),
    'SVM (Linear)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='linear'))
    ]),
    'Naive Bayes': GaussianNB()
}

# Comparar
results = []

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    # Tiempo de entrenamiento
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Tiempo de predicción
    start = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start

    # Test accuracy
    test_acc = model.score(X_test, y_test)

    results.append({
        'Model': name,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Test Acc': test_acc,
        'Train Time': train_time,
        'Pred Time': pred_time
    })

# Mostrar resultados
import pandas as pd
df = pd.DataFrame(results)
print(df.to_string(index=False))
```

### 4. Análisis por Tipo de Problema

#### Clasificación Binaria

| Problema           | Recomendación               |
| ------------------ | --------------------------- |
| Spam detection     | Naive Bayes (MultinomialNB) |
| Medical diagnosis  | SVM (RBF) con GridSearch    |
| Credit scoring     | SVM o ensemble              |
| Sentiment analysis | Naive Bayes o SVM linear    |

#### Clasificación Multiclase

| Problema                | Recomendación                  |
| ----------------------- | ------------------------------ |
| Image classification    | SVM (RBF)                      |
| Document categorization | Naive Bayes                    |
| Species classification  | KNN (si dataset pequeño) o SVM |

### 5. Consideraciones de Escalabilidad

```python
# Datos de ejemplo para testing de escalabilidad
from sklearn.datasets import make_classification

# Dataset pequeño (1k samples)
X_small, y_small = make_classification(n_samples=1000, n_features=20, random_state=42)

# Dataset grande (100k samples)
X_large, y_large = make_classification(n_samples=100000, n_features=20, random_state=42)
```

| Dataset       | KNN                    | SVM            | Naive Bayes     |
| ------------- | ---------------------- | -------------- | --------------- |
| 1k samples    | ✅ Rápido              | ✅ Rápido      | ✅ Instantáneo  |
| 10k samples   | ⚠️ Lento en predicción | ⚠️ Lento train | ✅ Rápido       |
| 100k+ samples | ❌ Muy lento           | ❌ Muy lento   | ✅ Sigue rápido |

### 6. Pipeline de Selección de Modelo

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

def compare_models(X_train, X_test, y_train, y_test):
    """
    Compara KNN, SVM y Naive Bayes con tuning básico.
    """

    # KNN con búsqueda de k
    knn_params = {'clf__n_neighbors': [3, 5, 7, 9]}
    knn_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier())
    ])
    knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=-1)

    # SVM con búsqueda de C y kernel
    svm_params = {
        'clf__C': [0.1, 1, 10],
        'clf__kernel': ['rbf', 'linear']
    }
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC())
    ])
    svm_grid = GridSearchCV(svm_pipe, svm_params, cv=5, n_jobs=-1)

    # Naive Bayes (no necesita mucho tuning)
    nb = GaussianNB()

    # Entrenar y evaluar
    results = {}

    for name, model in [('KNN', knn_grid), ('SVM', svm_grid), ('NB', nb)]:
        model.fit(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        if hasattr(model, 'best_params_'):
            print(f"{name} best params: {model.best_params_}")

        results[name] = test_acc
        print(f"{name} Test Accuracy: {test_acc:.4f}")

    # Mejor modelo
    best = max(results, key=results.get)
    print(f"\n🏆 Mejor modelo: {best} ({results[best]:.4f})")

    return results

# Usar la función
results = compare_models(X_train, X_test, y_train, y_test)
```

### 7. Guía de Decisión Rápida

```
¿Es clasificación de texto?
├── SÍ → Naive Bayes (MultinomialNB)
└── NO → ¿Dataset > 10k muestras?
    ├── SÍ → ¿Necesitas probabilidades?
    │   ├── SÍ → Naive Bayes (si features independientes) o SVM + probability=True
    │   └── NO → SVM con kernel RBF
    └── NO → ¿Features > 100?
        ├── SÍ → SVM (linear o RBF)
        └── NO → KNN o cualquiera de los 3
```

### 8. Resumen de Hiperparámetros

| Algoritmo | Hiperparámetros Clave        | Valores Típicos             |
| --------- | ---------------------------- | --------------------------- |
| **KNN**   | n_neighbors, weights, metric | k=5, 'uniform', 'euclidean' |
| **SVM**   | C, kernel, gamma             | C=1, 'rbf', 'scale'         |
| **NB**    | alpha (suavizado)            | 1.0 (Laplace)               |

---

## ✅ Checklist de Verificación

- [ ] Conozco las fortalezas y debilidades de cada algoritmo
- [ ] Sé cuándo usar KNN, SVM o Naive Bayes
- [ ] Puedo implementar comparación práctica
- [ ] Entiendo consideraciones de escalabilidad
- [ ] Puedo tomar decisiones informadas de selección

---

## 📚 Recursos

- [Choosing the right estimator - sklearn](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
- [Model Selection - sklearn](https://scikit-learn.org/stable/model_selection.html)
- [Comparing Classifiers](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
