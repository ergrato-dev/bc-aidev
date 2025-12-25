# 📖 Glosario - Semana 11: Árboles de Decisión y Random Forest

## A

### Accuracy (Exactitud)

Proporción de predicciones correctas sobre el total de predicciones.
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

### Agregación

Proceso de combinar predicciones de múltiples modelos. En Random Forest: **votación** (clasificación) o **promedio** (regresión).

## B

### Bagging (Bootstrap Aggregating)

Técnica de ensemble que:

1. Crea múltiples subconjuntos de datos mediante bootstrap sampling
2. Entrena un modelo en cada subconjunto
3. Agrega las predicciones

```python
from sklearn.ensemble import BaggingClassifier
```

### Bootstrap Sampling

Muestreo **con reemplazo** del dataset original. Cada muestra bootstrap tiene el mismo tamaño que el original, pero algunas observaciones se repiten y otras no aparecen (~37% quedan fuera).

## C

### CART (Classification and Regression Trees)

Algoritmo usado por sklearn para construir árboles de decisión. Características:

- Divisiones binarias
- Usa Gini o Entropy para clasificación
- Usa MSE o MAE para regresión

### ccp_alpha (Cost-Complexity Pruning)

Parámetro de poda que penaliza la complejidad del árbol. Valores mayores = árboles más simples.

```python
tree = DecisionTreeClassifier(ccp_alpha=0.01)
```

### Criterion (Criterio)

Métrica usada para evaluar la calidad de una división:

- **Clasificación**: `'gini'` (default), `'entropy'`
- **Regresión**: `'squared_error'` (default), `'absolute_error'`

## D

### Decision Boundary (Frontera de Decisión)

Límite que separa las clases en el espacio de features. En árboles, son fronteras **paralelas a los ejes** (rectangulares).

### Decision Tree (Árbol de Decisión)

Modelo que aprende reglas de decisión en forma de árbol:

- **Nodos internos**: condiciones (feature ≤ threshold)
- **Hojas**: predicciones

## E

### Ensemble (Conjunto)

Combinación de múltiples modelos para mejorar predicciones. Tipos principales:

- **Bagging**: Random Forest
- **Boosting**: XGBoost, AdaBoost
- **Stacking**: Meta-aprendizaje

### Entropy (Entropía)

Medida de desorden o incertidumbre basada en teoría de información.
$$Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

| Valor | Significado                  |
| ----- | ---------------------------- |
| 0     | Nodo puro                    |
| 1     | Máxima incertidumbre (50/50) |

## F

### Feature Importance (Importancia de Features)

Medida de cuánto contribuye cada feature a las predicciones. En Random Forest: suma ponderada de las reducciones de impureza.

```python
importance = model.feature_importances_
```

### Feature Subsampling

En Random Forest, solo se consideran `max_features` features aleatorias en cada división. Reduce correlación entre árboles.

## G

### Gini Impurity (Impureza de Gini)

Medida de impureza que calcula la probabilidad de clasificar incorrectamente.
$$Gini = 1 - \sum_{i=1}^{C} p_i^2$$

| Valor | Significado                     |
| ----- | ------------------------------- |
| 0     | Nodo puro                       |
| 0.5   | Máxima impureza (50/50 binario) |

### Greedy Algorithm (Algoritmo Codicioso)

Estrategia de CART que elige la **mejor división local** en cada paso, sin considerar divisiones futuras.

## H

### Hiperparámetro

Parámetro que se configura antes del entrenamiento (no se aprende de los datos).

**Principales en Random Forest:**

- `n_estimators`: número de árboles
- `max_depth`: profundidad máxima
- `min_samples_split`: mínimo para dividir
- `max_features`: features por división

## I

### Information Gain (Ganancia de Información)

Reducción de entropía después de una división.
$$IG = Entropy(padre) - \sum \frac{n_j}{n} Entropy(hijo_j)$$

### Internal Node (Nodo Interno)

Nodo que contiene una condición de división (no es hoja).

## L

### Leaf Node (Hoja)

Nodo terminal que contiene la predicción final (clase mayoritaria o valor promedio).

## M

### max_depth

Profundidad máxima del árbol. Limitar previene overfitting.

```python
tree = DecisionTreeClassifier(max_depth=5)
```

### max_features

Número de features a considerar en cada división:

- `'sqrt'`: √n_features (default clasificación)
- `'log2'`: log₂(n_features)
- `None`: todos los features

### min_samples_leaf

Número mínimo de muestras requeridas en un nodo hoja.

### min_samples_split

Número mínimo de muestras para dividir un nodo interno.

## N

### n_estimators

Número de árboles en Random Forest. Más árboles = mejor rendimiento (hasta cierto punto) pero más lento.

```python
rf = RandomForestClassifier(n_estimators=100)
```

### n_jobs

Número de cores para paralelización. `-1` usa todos los disponibles.

## O

### OOB (Out-of-Bag)

Muestras no usadas en un bootstrap particular (~37%). Permiten validación interna.

### OOB Score

Estimación del error de generalización usando muestras OOB. Similar a cross-validation pero "gratis".

```python
rf = RandomForestClassifier(oob_score=True)
rf.fit(X, y)
print(rf.oob_score_)
```

### Overfitting (Sobreajuste)

Cuando el modelo memoriza los datos de entrenamiento y no generaliza. Señales:

- Train accuracy >> Test accuracy
- Árbol muy profundo

## P

### Pruning (Poda)

Técnica para simplificar árboles y evitar overfitting:

- **Pre-pruning**: limitar durante construcción (`max_depth`, `min_samples_split`)
- **Post-pruning**: podar después de construir (`ccp_alpha`)

## R

### Random Forest

Ensemble de árboles de decisión que usa:

1. **Bagging**: bootstrap sampling
2. **Feature randomness**: subconjunto aleatorio de features por split
3. **Agregación**: votación o promedio

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
```

### random_state

Semilla para reproducibilidad. Mismo valor = mismos resultados.

## S

### Split (División)

Partición de un nodo en dos hijos basada en una condición (feature ≤ threshold).

## T

### Threshold (Umbral)

Valor de corte para una feature en una división. Ejemplo: "edad ≤ 30".

### Tree Depth (Profundidad)

Número de niveles desde la raíz hasta la hoja más profunda.

## V

### Variance (Varianza)

Sensibilidad del modelo a cambios en los datos de entrenamiento. Árboles individuales tienen **alta varianza**; Random Forest la reduce.

### Voting (Votación)

Método de agregación en clasificación:

- **Hard voting**: clase más votada
- **Soft voting**: promedio de probabilidades

---

## 📊 Tabla Resumen: Hiperparámetros

| Parámetro           | Efecto al aumentar      | Default |
| ------------------- | ----------------------- | ------- |
| `max_depth`         | ↑ Overfitting           | None    |
| `min_samples_split` | ↓ Overfitting           | 2       |
| `min_samples_leaf`  | ↓ Overfitting           | 1       |
| `n_estimators`      | ↑ Performance, ↑ Tiempo | 100     |
| `max_features`      | ↑ Correlación árboles   | 'sqrt'  |
| `ccp_alpha`         | ↓ Complejidad           | 0.0     |

---

## 🔗 Referencias

- [sklearn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [sklearn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [Random Forests - Breiman 2001](https://link.springer.com/article/10.1023/A:1010933404324)
