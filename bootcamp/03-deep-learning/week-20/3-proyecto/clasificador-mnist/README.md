# 🔢 Proyecto: Clasificador MNIST con Keras

## 🎯 Objetivo

Construir un clasificador de dígitos escritos a mano utilizando TensorFlow/Keras, aplicando todos los conceptos aprendidos durante la semana: API Sequential, capas, compilación, callbacks y guardado de modelos.

## ⏱️ Duración

2 horas

## 📋 Descripción del Proyecto

El dataset MNIST contiene 70,000 imágenes de dígitos escritos a mano (0-9), cada una de 28x28 píxeles en escala de grises. Tu tarea es construir una red neuronal que clasifique estos dígitos con al menos **97% de accuracy** en el conjunto de test.

---

## 📊 El Dataset

```
MNIST Dataset:
├── Entrenamiento: 60,000 imágenes
├── Test: 10,000 imágenes
├── Tamaño: 28x28 píxeles
├── Canales: 1 (grayscale)
├── Clases: 10 (dígitos 0-9)
└── Formato: numpy arrays
```

---

## 🎯 Requisitos del Proyecto

### Requisitos Mínimos (70%)

1. **Preprocesamiento correcto**
   - Normalización de píxeles (0-255 → 0-1)
   - Reshape apropiado para el modelo
   - Split de validación

2. **Arquitectura del modelo**
   - Mínimo 2 capas ocultas
   - Activaciones apropiadas
   - Capa de salida con softmax

3. **Entrenamiento**
   - Compilación con optimizer, loss y metrics
   - Al menos 1 callback implementado
   - Accuracy ≥ 95% en test

### Requisitos Intermedios (85%)

4. **Regularización**
   - Dropout en capas ocultas
   - BatchNormalization opcional

5. **Callbacks completos**
   - EarlyStopping
   - ModelCheckpoint

6. **Métricas**
   - Accuracy ≥ 97% en test
   - Visualización de curvas de entrenamiento

### Requisitos Avanzados (100%)

7. **Optimización**
   - ReduceLROnPlateau
   - Experimentación con hiperparámetros

8. **Evaluación completa**
   - Matriz de confusión
   - Visualización de predicciones incorrectas
   - Accuracy ≥ 98% en test

9. **Documentación**
   - Código comentado
   - Decisiones justificadas
   - Modelo exportado

---

## 🗂️ Estructura del Proyecto

```
clasificador-mnist/
├── README.md              # Este archivo
├── starter/
│   └── main.py           # Plantilla con TODOs
├── solution/
│   └── main.py           # Solución de referencia
└── outputs/              # Generado durante ejecución
    ├── best_model.keras
    ├── training_history.png
    └── confusion_matrix.png
```

---

## 📝 Instrucciones

### Paso 1: Cargar y Explorar Datos

```python
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Explorar shapes y valores
print(f"Train: {X_train.shape}")  # (60000, 28, 28)
print(f"Valores: min={X_train.min()}, max={X_train.max()}")
```

### Paso 2: Preprocesar

```python
# Aplanar: (60000, 28, 28) → (60000, 784)
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

# Separar validación
X_val, y_val = X_train[:6000], y_train[:6000]
X_train, y_train = X_train[6000:], y_train[6000:]
```

### Paso 3: Diseñar Arquitectura

Experimenta con diferentes configuraciones:

```python
model = Sequential([
    # TODO: Diseñar arquitectura
    # - Input: 784 features
    # - Hidden layers con ReLU
    # - Regularización (Dropout/BatchNorm)
    # - Output: 10 clases con softmax
])
```

### Paso 4: Compilar

```python
model.compile(
    optimizer='adam',  # Experimentar con learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Paso 5: Configurar Callbacks

```python
callbacks = [
    # TODO: EarlyStopping
    # TODO: ModelCheckpoint
    # TODO: ReduceLROnPlateau (opcional)
]
```

### Paso 6: Entrenar

```python
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)
```

### Paso 7: Evaluar

```python
# Evaluación en test
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Matriz de confusión
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)
```

### Paso 8: Visualizar

```python
# Curvas de entrenamiento
# Matriz de confusión
# Predicciones incorrectas
```

---

## 🎨 Arquitectura Sugerida

Punto de partida recomendado:

```
Input (784)
    ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(64) → BatchNorm → ReLU → Dropout(0.2)
    ↓
Dense(10) → Softmax
    ↓
Output (10 probabilidades)
```

---

## 📈 Métricas Objetivo

| Métrica      | Mínimo | Esperado | Excelente |
| ------------ | ------ | -------- | --------- |
| Test Accuracy | 95%   | 97%      | >98%      |
| Val Loss     | <0.15  | <0.10    | <0.08     |

---

## 💡 Tips

1. **Empieza simple**: Un modelo básico primero, luego añade complejidad
2. **Monitorea overfitting**: Si train_acc >> val_acc, añade regularización
3. **Learning rate**: Si el loss oscila mucho, reduce el learning rate
4. **Batch size**: 32-128 suelen funcionar bien
5. **Paciencia**: Usa EarlyStopping con patience suficiente (5-10)

---

## ✅ Checklist de Entrega

- [ ] Código ejecutable sin errores
- [ ] Preprocesamiento correcto
- [ ] Modelo con arquitectura justificada
- [ ] Callbacks implementados
- [ ] Test accuracy ≥ 97%
- [ ] Visualizaciones generadas
- [ ] Modelo guardado en .keras
- [ ] Código comentado

---

## 📚 Recursos

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Keras Sequential API](https://keras.io/guides/sequential_model/)
- [Keras Callbacks](https://keras.io/api/callbacks/)

---

_Proyecto Semana 20 | TensorFlow y Keras | Bootcamp IA: Zero to Hero_
