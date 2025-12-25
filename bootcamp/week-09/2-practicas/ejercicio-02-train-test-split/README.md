# Ejercicio 02: Train/Test Split

## 🎯 Objetivo

Aprender a dividir correctamente un dataset en conjuntos de entrenamiento y prueba para evaluar modelos de ML.

## 📋 Descripción

La división train/test es fundamental en ML. Entrenar y evaluar con los mismos datos no nos dice si el modelo generaliza. En este ejercicio aprenderás diferentes estrategias de división.

## 📚 Conceptos Clave

- **Train Set**: Datos para entrenar el modelo (típicamente 70-80%)
- **Test Set**: Datos para evaluar el modelo (típicamente 20-30%)
- **Stratify**: Mantener la proporción de clases en ambos conjuntos
- **Random State**: Semilla para reproducibilidad

## 🛠️ Instrucciones

Abre `starter/main.py` y sigue los pasos descomentando el código indicado.

### Paso 1: Cargar Datos

Usaremos el dataset Iris para practicar la división.

### Paso 2: División Básica

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Paso 3: División Estratificada

Cuando hay clases desbalanceadas, es importante mantener las proporciones:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Paso 4: Verificar la División

Comprueba que las proporciones sean correctas en ambos conjuntos.

### Paso 5: División Train/Val/Test

Para proyectos más robustos, usamos tres conjuntos.

## ✅ Resultado Esperado

- Train set: ~120 samples (80%)
- Test set: ~30 samples (20%)
- Proporciones de clases mantenidas en ambos conjuntos

## 🔗 Recursos

- [train_test_split Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
