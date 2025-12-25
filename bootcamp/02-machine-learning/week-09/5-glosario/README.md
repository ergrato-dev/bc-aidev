# 📖 Glosario - Semana 09: Fundamentos de Machine Learning

Términos técnicos ordenados alfabéticamente.

---

## A

### Accuracy (Exactitud)

Métrica que mide la proporción de predicciones correctas sobre el total.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
```

### Aprendizaje No Supervisado (Unsupervised Learning)

Tipo de ML donde el modelo aprende patrones de datos sin etiquetas. Ejemplos: clustering, reducción de dimensionalidad.

### Aprendizaje por Refuerzo (Reinforcement Learning)

Tipo de ML donde un agente aprende a tomar decisiones mediante recompensas y penalizaciones.

### Aprendizaje Supervisado (Supervised Learning)

Tipo de ML donde el modelo aprende de datos etiquetados (features + target conocido). Ejemplos: clasificación, regresión.

---

## B

### Bias (Sesgo)

Error introducido por suposiciones simplificadas en el modelo. Alto sesgo causa **underfitting**.

### Bias-Variance Tradeoff

Balance entre sesgo y varianza. Modelos simples tienen alto sesgo; modelos complejos tienen alta varianza.

---

## C

### Clasificación (Classification)

Tarea de ML que predice categorías discretas (clases). Ejemplo: spam/no spam.

### Clustering

Técnica de aprendizaje no supervisado que agrupa datos similares sin etiquetas previas.

### Confusion Matrix (Matriz de Confusión)

Tabla que muestra predicciones vs valores reales en clasificación:

|               | Pred: Neg | Pred: Pos |
| ------------- | --------- | --------- |
| **Real: Neg** | TN        | FP        |
| **Real: Pos** | FN        | TP        |

### Cross-Validation (Validación Cruzada)

Técnica que divide datos en K partes (folds), entrena en K-1 y valida en 1, rotando K veces.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
```

---

## D

### Dataset

Conjunto de datos organizado en filas (samples) y columnas (features).

### Data Leakage (Fuga de Datos)

Error donde información del test set "se filtra" al entrenamiento, causando métricas engañosamente altas.

---

## E

### EDA (Exploratory Data Analysis)

Análisis exploratorio de datos antes de modelar: estadísticas, visualizaciones, detección de patrones.

### Epoch

Una pasada completa por todos los datos de entrenamiento.

---

## F

### F1-Score

Media armónica de precision y recall. Útil cuando las clases están desbalanceadas.

$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

### False Negative (FN)

Predicción negativa cuando el valor real es positivo. "Miss" o "error tipo II".

### False Positive (FP)

Predicción positiva cuando el valor real es negativo. "Falsa alarma" o "error tipo I".

### Feature (Característica)

Variable de entrada usada para hacer predicciones. Columnas de X.

### Feature Engineering

Proceso de crear, seleccionar o transformar features para mejorar el modelo.

### fit()

Método de sklearn que entrena un modelo con datos.

```python
model.fit(X_train, y_train)
```

---

## G

### Generalization (Generalización)

Capacidad del modelo de funcionar bien con datos nuevos no vistos durante el entrenamiento.

---

## H

### Hiperparámetro

Parámetro del modelo configurado antes del entrenamiento (ej: n_neighbors en KNN).

### Holdout

Técnica de validación que separa un conjunto de datos para prueba final.

---

## I

### Imputación

Técnica para rellenar valores faltantes (nulos) en un dataset.

```python
df['col'].fillna(df['col'].median(), inplace=True)
```

---

## K

### K-Fold Cross-Validation

Validación cruzada dividiendo datos en K partes iguales.

### KNN (K-Nearest Neighbors)

Algoritmo que clasifica basándose en los K vecinos más cercanos.

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
```

---

## L

### Label (Etiqueta)

Valor de la variable target en datos supervisados. También llamado "y".

### Learning Rate

Hiperparámetro que controla cuánto ajusta el modelo en cada iteración.

---

## M

### Machine Learning (ML)

Campo de la IA donde los sistemas aprenden patrones de datos sin ser explícitamente programados.

### Model (Modelo)

Representación matemática aprendida de los datos que puede hacer predicciones.

---

## O

### Overfitting (Sobreajuste)

Modelo que memoriza datos de entrenamiento pero falla en datos nuevos. Alta varianza.

**Síntomas**: Alta accuracy en train, baja en test.

---

## P

### Precision (Precisión)

De las predicciones positivas, ¿qué proporción era realmente positiva?

$$\text{Precision} = \frac{TP}{TP + FP}$$

### predict()

Método de sklearn que genera predicciones con un modelo entrenado.

```python
y_pred = model.predict(X_test)
```

---

## R

### Random State

Semilla para el generador aleatorio. Garantiza reproducibilidad.

```python
train_test_split(X, y, random_state=42)
```

### Recall (Sensibilidad)

De los positivos reales, ¿qué proporción encontró el modelo?

$$\text{Recall} = \frac{TP}{TP + FN}$$

### Regresión (Regression)

Tarea de ML que predice valores continuos. Ejemplo: precio de una casa.

---

## S

### Sample (Muestra)

Una fila del dataset. Un ejemplo individual.

### score()

Método de sklearn que calcula la métrica por defecto (accuracy para clasificación).

```python
accuracy = model.score(X_test, y_test)
```

### Stratify

Parámetro que mantiene las proporciones de clases al dividir datos.

```python
train_test_split(X, y, stratify=y)
```

---

## T

### Target (Variable Objetivo)

Variable que queremos predecir. También llamada "y" o "label".

### Test Set (Conjunto de Prueba)

Datos reservados para evaluar el modelo final. No se usa para entrenar.

### Train Set (Conjunto de Entrenamiento)

Datos usados para entrenar el modelo.

### train_test_split

Función de sklearn para dividir datos en train y test.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### True Negative (TN)

Predicción negativa correcta.

### True Positive (TP)

Predicción positiva correcta.

---

## U

### Underfitting (Subajuste)

Modelo demasiado simple que no captura patrones. Alto sesgo.

**Síntomas**: Baja accuracy tanto en train como en test.

---

## V

### Validation Set (Conjunto de Validación)

Datos usados para ajustar hiperparámetros durante el desarrollo.

### Variance (Varianza)

Sensibilidad del modelo a pequeñas fluctuaciones en datos. Alta varianza causa **overfitting**.

---

## Fórmulas Resumen

| Métrica   | Fórmula                                         |
| --------- | ----------------------------------------------- |
| Accuracy  | (TP + TN) / (TP + TN + FP + FN)                 |
| Precision | TP / (TP + FP)                                  |
| Recall    | TP / (TP + FN)                                  |
| F1-Score  | 2 × (Precision × Recall) / (Precision + Recall) |

---

_Glosario actualizado: Semana 09 - Fundamentos de Machine Learning_
