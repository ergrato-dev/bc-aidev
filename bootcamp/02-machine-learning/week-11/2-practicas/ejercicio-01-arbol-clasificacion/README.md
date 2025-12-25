# Ejercicio 01: Árbol de Decisión para Clasificación

## 🎯 Objetivo

Construir un árbol de decisión para clasificar flores del dataset Iris, visualizar el árbol y analizar las predicciones.

## 📋 Conceptos Clave

- `DecisionTreeClassifier` de scikit-learn
- Parámetros `max_depth` y `criterion`
- Visualización con `plot_tree`
- Predicción de clases y probabilidades

## ⏱️ Tiempo Estimado

30 minutos

---

## 📝 Instrucciones

### Paso 1: Importar Librerías

Comenzamos importando las librerías necesarias para árboles de decisión.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Cargar el Dataset Iris

Iris es un dataset clásico con 150 muestras de 3 especies de flores.

```python
iris = load_iris()
X, y = iris.data, iris.target

print(f"Features: {iris.feature_names}")
print(f"Clases: {iris.target_names}")
print(f"Shape X: {X.shape}, Shape y: {y.shape}")
```

**Descomenta** la sección del Paso 2 en `starter/main.py`.

---

### Paso 3: Dividir los Datos

Separamos 80% para entrenamiento y 20% para test.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Mantiene proporciones de clases
)

print(f"Train: {X_train.shape[0]} muestras")
print(f"Test: {X_test.shape[0]} muestras")
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Crear y Entrenar el Árbol

Creamos un árbol con profundidad limitada para evitar overfitting.

```python
tree = DecisionTreeClassifier(
    max_depth=3,           # Limitar profundidad
    criterion='gini',      # Criterio de división
    random_state=42
)

tree.fit(X_train, y_train)
print("Árbol entrenado correctamente")
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Evaluar el Modelo

Calculamos accuracy y mostramos el reporte de clasificación.

```python
y_pred = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Visualizar el Árbol

Scikit-learn permite visualizar el árbol de decisión.

```python
plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,           # Colorear nodos
    rounded=True,          # Bordes redondeados
    fontsize=10
)
plt.title('Árbol de Decisión - Iris Dataset')
plt.tight_layout()
plt.savefig('arbol_iris.png', dpi=150)
plt.show()
```

**Descomenta** la sección del Paso 6.

---

### Paso 7: Predecir con Probabilidades

Podemos obtener la probabilidad de cada clase.

```python
# Tomar algunas muestras de test
sample = X_test[:3]

# Predicción de clase
predictions = tree.predict(sample)

# Probabilidades
probabilities = tree.predict_proba(sample)

for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
    print(f"\nMuestra {i+1}:")
    print(f"  Predicción: {iris.target_names[pred]}")
    print(f"  Probabilidades: {dict(zip(iris.target_names, proba.round(3)))}")
```

**Descomenta** la sección del Paso 7.

---

### Paso 8: Experimentar con max_depth

Observa cómo cambia el rendimiento con diferentes profundidades.

```python
print("\n--- Impacto de max_depth ---")
for depth in [1, 2, 3, 5, 10, None]:
    tree_exp = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_exp.fit(X_train, y_train)

    train_acc = tree_exp.score(X_train, y_train)
    test_acc = tree_exp.score(X_test, y_test)

    depth_str = str(depth) if depth else "None"
    print(f"max_depth={depth_str:4s}: Train={train_acc:.4f}, Test={test_acc:.4f}")
```

**Descomenta** la sección del Paso 8.

---

## ✅ Resultado Esperado

Al ejecutar el código completo deberías ver:

1. Información del dataset (150 muestras, 4 features, 3 clases)
2. Accuracy del modelo (~96-100%)
3. Visualización del árbol guardada como `arbol_iris.png`
4. Predicciones con probabilidades
5. Comparación de diferentes profundidades

---

## 🔬 Experimenta

1. Cambia `criterion` de `'gini'` a `'entropy'` y compara
2. Prueba con `min_samples_split=5` o `min_samples_leaf=3`
3. ¿Qué pasa si usas `max_depth=None`?

---

## 📚 Recursos

- [DecisionTreeClassifier - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [plot_tree - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)
