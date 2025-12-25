# Ejercicio 01: Exploración de Datos para ML

## 🎯 Objetivo

Aprender a explorar y analizar un dataset antes de aplicar Machine Learning, identificando características relevantes para el modelado.

## 📋 Descripción

En este ejercicio explorarás el dataset Iris, uno de los más clásicos en ML, aplicando técnicas de EDA (Exploratory Data Analysis) orientadas a preparar los datos para un modelo de clasificación.

## 📚 Conceptos Clave

- **Dataset**: Conjunto de datos con features y target
- **Features (X)**: Variables de entrada (características)
- **Target (y)**: Variable a predecir (etiqueta)
- **EDA**: Análisis exploratorio antes de modelar

## 🛠️ Instrucciones

Abre `starter/main.py` y sigue los pasos descomentando el código indicado.

### Paso 1: Cargar el Dataset

Scikit-learn incluye datasets de ejemplo listos para usar:

```python
from sklearn.datasets import load_iris

iris = load_iris()
print(type(iris))  # Es un objeto Bunch (similar a diccionario)
```

### Paso 2: Explorar la Estructura

El dataset tiene atributos importantes:

- `data`: matriz de features
- `target`: array de etiquetas
- `feature_names`: nombres de las columnas
- `target_names`: nombres de las clases

### Paso 3: Convertir a DataFrame

Para mejor manipulación, convertimos a pandas DataFrame:

```python
import pandas as pd

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
```

### Paso 4: Análisis Estadístico

Usa `describe()` para ver estadísticas de cada feature.

### Paso 5: Distribución del Target

Verifica si las clases están balanceadas con `value_counts()`.

### Paso 6: Visualización

Crea visualizaciones para entender las relaciones entre features.

## ✅ Resultado Esperado

Al ejecutar el script completo deberías ver:

- Shape del dataset (150 samples, 4 features)
- Estadísticas descriptivas de cada feature
- Distribución balanceada de las 3 clases (50 cada una)
- Visualizaciones de las distribuciones

## 🔗 Recursos

- [Sklearn Iris Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset)
- [Pandas describe()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)
