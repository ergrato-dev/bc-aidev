# Árboles de Decisión: Fundamentos

## 🎯 Objetivos

- Entender la estructura de un árbol de decisión
- Comprender cómo se realizan las predicciones
- Conocer el algoritmo CART
- Identificar ventajas y limitaciones

## 📋 Contenido

### 1. ¿Qué es un Árbol de Decisión?

Un árbol de decisión es un modelo de ML que toma decisiones secuenciales basadas en preguntas sobre las features, similar a un diagrama de flujo.

![Estructura del Árbol](../0-assets/01-arbol-decision-estructura.svg)

### 2. Componentes del Árbol

| Componente         | Descripción                                     |
| ------------------ | ----------------------------------------------- |
| **Nodo Raíz**      | Primera división, usa la feature más importante |
| **Nodos Internos** | Divisiones intermedias                          |
| **Nodos Hoja**     | Predicción final (clase o valor)                |
| **Rama**           | Conexión entre nodos (condición)                |
| **Profundidad**    | Número de niveles del árbol                     |

### 3. Algoritmo CART

**CART** (Classification And Regression Trees) es el algoritmo que usa scikit-learn:

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Clasificación
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Regresión
reg = DecisionTreeRegressor(max_depth=5, random_state=42)
reg.fit(X_train, y_train)
```

### 4. ¿Cómo Decide Dónde Dividir?

En cada nodo, el algoritmo:

1. **Evalúa todas las features** disponibles
2. **Prueba todos los posibles umbrales** (thresholds)
3. **Selecciona la división** que maximiza la "pureza" de los nodos hijos
4. **Repite recursivamente** hasta cumplir criterio de parada

```python
# Ejemplo de división
# Si feature "edad" con threshold 30:
# - Izquierda: muestras donde edad <= 30
# - Derecha: muestras donde edad > 30
```

### 5. Predicción en Árboles

#### Clasificación

```python
# Recorre el árbol hasta llegar a una hoja
# Retorna la clase mayoritaria en esa hoja
y_pred = clf.predict(X_test)

# También puede dar probabilidades
y_proba = clf.predict_proba(X_test)
# Devuelve proporción de cada clase en la hoja
```

#### Regresión

```python
# Retorna el valor promedio de las muestras en la hoja
y_pred = reg.predict(X_test)
```

### 6. Ventajas de los Árboles

| Ventaja                     | Descripción                         |
| --------------------------- | ----------------------------------- |
| ✅ **Interpretabilidad**    | Fácil de visualizar y explicar      |
| ✅ **No requiere escalado** | Funciona con datos sin normalizar   |
| ✅ **Maneja mixtos**        | Features numéricas y categóricas    |
| ✅ **No lineal**            | Captura relaciones complejas        |
| ✅ **Feature importance**   | Indica qué features son importantes |

### 7. Limitaciones

| Limitación                                     | Solución                                |
| ---------------------------------------------- | --------------------------------------- |
| ⚠️ **Overfitting**                             | Limitar profundidad, poda               |
| ⚠️ **Inestabilidad**                           | Pequeños cambios → árbol diferente      |
| ⚠️ **Sesgo hacia features con muchos valores** | Usar Gini en lugar de Entropy           |
| ⚠️ **No extrapola**                            | Solo predice valores vistos en training |

### 8. Visualización del Árbol

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Visualizar árbol
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.savefig('arbol_decision.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9. Ejemplo Completo

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar datos
iris = load_iris()
X, y = iris.data, iris.target

# Dividir
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar árbol (limitando profundidad)
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Evaluar
y_pred = tree.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Ver importancia de features
for name, imp in zip(iris.feature_names, tree.feature_importances_):
    print(f"{name}: {imp:.4f}")
```

---

## ✅ Checklist de Verificación

- [ ] Entiendo la estructura: raíz, nodos internos, hojas
- [ ] Sé cómo el árbol hace predicciones
- [ ] Conozco las ventajas de interpretabilidad
- [ ] Identifico el riesgo de overfitting
- [ ] Puedo visualizar un árbol con sklearn

---

## 📚 Recursos

- [Decision Trees - sklearn](https://scikit-learn.org/stable/modules/tree.html)
- [Visualizing Decision Trees](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)
