# Ejercicio 03: Tu Primer Modelo de ML

## 🎯 Objetivo

Entrenar tu primer modelo de Machine Learning usando scikit-learn y entender el flujo básico de entrenamiento y predicción.

## 📋 Descripción

En este ejercicio crearás un modelo de clasificación usando el algoritmo K-Nearest Neighbors (KNN) para clasificar flores del dataset Iris.

## 📚 Conceptos Clave

- **fit()**: Entrenar el modelo con datos
- **predict()**: Hacer predicciones con el modelo entrenado
- **score()**: Evaluar la precisión del modelo
- **KNN**: Algoritmo que clasifica basándose en los K vecinos más cercanos

## 🛠️ Instrucciones

Abre `starter/main.py` y sigue los pasos descomentando el código indicado.

### Paso 1: Preparar los Datos

Cargar y dividir el dataset.

### Paso 2: Crear el Modelo

```python
from sklearn.neighbors import KNeighborsClassifier

modelo = KNeighborsClassifier(n_neighbors=3)
```

### Paso 3: Entrenar (fit)

```python
modelo.fit(X_train, y_train)
```

### Paso 4: Predecir (predict)

```python
predicciones = modelo.predict(X_test)
```

### Paso 5: Evaluar (score)

```python
accuracy = modelo.score(X_test, y_test)
```

## ✅ Resultado Esperado

- Modelo entrenado exitosamente
- Accuracy > 90% en el dataset Iris
- Predicciones correctas para nuevas muestras

## 🔗 Recursos

- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Scikit-learn Getting Started](https://scikit-learn.org/stable/getting_started.html)
