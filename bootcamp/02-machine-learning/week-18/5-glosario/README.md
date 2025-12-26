# 📖 Glosario - Semana 18

## A

### Accuracy
Métrica de clasificación que mide la proporción de predicciones correctas.
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

---

## B

### Baseline
Modelo simple usado como referencia para comparar modelos más complejos. Generalmente usa estrategias triviales como predecir siempre la clase más frecuente.

---

## C

### ColumnTransformer
Clase de sklearn que permite aplicar diferentes transformaciones a diferentes columnas del dataset.
```python
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'fare']),
    ('cat', OneHotEncoder(), ['sex', 'embarked'])
])
```

### CRISP-DM
Cross-Industry Standard Process for Data Mining. Metodología de 6 fases para proyectos de ciencia de datos:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

### Cross-Validation
Técnica de validación que divide los datos en K folds, entrena en K-1 y valida en el restante, rotando el proceso.

---

## D

### Data Leakage
Error metodológico donde información del conjunto de test "se filtra" al entrenamiento, causando métricas infladas artificialmente.

### DummyClassifier
Clasificador de sklearn que usa estrategias simples (clase más frecuente, aleatorio) para establecer baselines.

---

## E

### EDA (Exploratory Data Analysis)
Proceso de análisis inicial de datos para entender distribuciones, correlaciones, outliers y patrones.

### End-to-End
Proceso completo desde datos crudos hasta predicciones finales, incluyendo todas las etapas intermedias.

---

## F

### Feature Engineering
Proceso de crear, transformar o seleccionar features para mejorar el rendimiento del modelo.

### Fold
Cada una de las particiones de datos en cross-validation.

---

## G

### GridSearchCV
Búsqueda exhaustiva de hiperparámetros probando todas las combinaciones posibles en una grilla definida.
```python
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(model, param_grid, cv=5)
```

---

## H

### Hyperparameter Tuning
Proceso de encontrar los mejores hiperparámetros para un modelo mediante búsqueda sistemática.

---

## I

### Imputation
Proceso de rellenar valores faltantes (missing values) con valores estimados como media, mediana o moda.

---

## K

### K-Fold
Variante de cross-validation que divide los datos en K particiones iguales.

---

## L

### Leakage
Ver Data Leakage.

---

## M

### Missing Values
Valores faltantes o nulos en un dataset que requieren tratamiento especial.

---

## O

### OneHotEncoder
Técnica de encoding que convierte variables categóricas en vectores binarios.

### Overfitting
Cuando un modelo memoriza el training set y no generaliza bien a nuevos datos.

---

## P

### Pipeline
Secuencia de transformaciones y modelo encadenados que garantizan reproducibilidad y evitan data leakage.
```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
```

---

## R

### Reproducibility
Capacidad de obtener los mismos resultados al ejecutar el mismo código con los mismos datos.

---

## S

### Submission
Archivo CSV con predicciones formateado según los requisitos de una competencia (ej. Kaggle).

---

## T

### Train-Test Split
División de datos en conjunto de entrenamiento y conjunto de prueba para evaluar modelos.

### Transformer
En sklearn, cualquier objeto que implementa `fit()` y `transform()` para modificar datos.

---

## V

### Validation Set
Conjunto de datos separado del training para evaluar modelos durante el desarrollo (diferente del test final).

---

## 🔗 Referencias

- [Scikit-learn Glossary](https://scikit-learn.org/stable/glossary.html)
- [CRISP-DM Wikipedia](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
- [Kaggle Learn](https://www.kaggle.com/learn)
