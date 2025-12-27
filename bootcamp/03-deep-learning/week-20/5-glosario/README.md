# 📖 Glosario - Semana 20

Términos clave de TensorFlow y Keras.

---

## A

### Activation Function (Función de Activación)

Función matemática aplicada a la salida de cada neurona. Introduce no-linealidad permitiendo que la red aprenda patrones complejos. Ejemplos: ReLU, sigmoid, softmax.

```python
# En Keras
layers.Dense(64, activation='relu')
```

### Adam (Adaptive Moment Estimation)

Optimizador que combina las ventajas de AdaGrad y RMSprop. Calcula learning rates adaptativos para cada parámetro. Es el optimizador por defecto más recomendado.

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
```

---

## B

### Batch

Subconjunto de datos de entrenamiento procesados juntos antes de actualizar los pesos. Un batch_size de 32 significa que 32 muestras se procesan antes de cada actualización de gradiente.

### Batch Normalization

Técnica que normaliza las activaciones de cada capa durante el entrenamiento. Acelera el entrenamiento y actúa como regularización.

```python
layers.Dense(64)
layers.BatchNormalization()
layers.Activation('relu')
```

---

## C

### Callback

Objeto que ejecuta acciones en puntos específicos del entrenamiento (inicio/fin de época, batch, etc.). Permiten control avanzado del proceso de entrenamiento.

```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]
```

### Categorical Crossentropy

Función de pérdida para clasificación multiclase. Mide la diferencia entre la distribución predicha y la real.

$$L = -\sum_{i} y_i \log(\hat{y}_i)$$

### Compile

Método que configura el modelo para entrenamiento, especificando optimizer, loss y métricas.

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## D

### Dense Layer (Capa Densa)

Capa completamente conectada donde cada neurona recibe entrada de todas las neuronas de la capa anterior. También llamada "fully connected".

```python
layers.Dense(units=64, activation='relu')
```

### Dropout

Técnica de regularización que desactiva aleatoriamente un porcentaje de neuronas durante el entrenamiento para prevenir overfitting.

```python
layers.Dropout(0.3)  # 30% de neuronas desactivadas
```

---

## E

### EarlyStopping

Callback que detiene el entrenamiento cuando una métrica monitoreada deja de mejorar.

```python
EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

### Epoch (Época)

Una pasada completa por todo el conjunto de entrenamiento. 10 epochs significa que cada muestra fue vista 10 veces.

### Eager Execution

Modo de ejecución de TensorFlow 2.x donde las operaciones se evalúan inmediatamente, sin necesidad de sesiones.

---

## F

### Fit

Método que entrena el modelo con los datos proporcionados.

```python
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val)
)
```

### Flatten

Capa que convierte un tensor multidimensional a 1D. Típicamente usado antes de capas Dense.

```python
layers.Flatten(input_shape=(28, 28, 1))  # (28, 28, 1) → (784,)
```

---

## G

### Glorot Initialization (Xavier)

Método de inicialización de pesos que mantiene la varianza de las activaciones. Recomendado para tanh y sigmoid.

```python
kernel_initializer='glorot_uniform'
```

### GradientTape

Contexto de TensorFlow para calcular gradientes automáticamente durante operaciones.

```python
with tf.GradientTape() as tape:
    y = model(x)
    loss = loss_fn(y, y_true)
gradients = tape.gradient(loss, model.trainable_variables)
```

---

## H

### He Initialization

Método de inicialización de pesos optimizado para funciones de activación ReLU.

```python
kernel_initializer='he_normal'
```

### History

Objeto retornado por `model.fit()` que contiene las métricas de entrenamiento por época.

```python
history.history['loss']      # Lista de loss por época
history.history['val_accuracy']  # Accuracy de validación
```

---

## I

### Input Shape

Forma de los datos de entrada que el modelo espera. Se especifica en la primera capa.

```python
layers.Dense(64, input_shape=(784,))  # 784 features de entrada
```

---

## K

### Keras

API de alto nivel para construir y entrenar modelos de deep learning. Integrada en TensorFlow desde la versión 2.0.

### Kernel

En el contexto de capas Dense, los "kernels" son los pesos (weights) de las conexiones.

```python
kernel_initializer='he_normal'
kernel_regularizer=tf.keras.regularizers.l2(0.01)
```

---

## L

### Learning Rate

Hiperparámetro que controla cuánto se ajustan los pesos en cada paso de optimización. Valores típicos: 0.001, 0.01.

### Loss Function

Función que mide qué tan lejos están las predicciones del modelo de los valores reales. El objetivo del entrenamiento es minimizarla.

---

## M

### Metrics

Medidas de rendimiento del modelo que se calculan durante entrenamiento y evaluación, pero no afectan el proceso de optimización.

```python
metrics=['accuracy', 'precision', 'recall']
```

### ModelCheckpoint

Callback que guarda el modelo durante el entrenamiento.

```python
ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True
)
```

---

## O

### Optimizer

Algoritmo que actualiza los pesos del modelo usando los gradientes calculados. Ejemplos: Adam, SGD, RMSprop.

### Overfitting

Cuando el modelo aprende demasiado bien los datos de entrenamiento pero no generaliza a datos nuevos. Se detecta cuando train_accuracy >> val_accuracy.

---

## P

### Predict

Método que genera predicciones para datos de entrada.

```python
predictions = model.predict(X_test)
```

---

## R

### ReduceLROnPlateau

Callback que reduce el learning rate cuando una métrica deja de mejorar.

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3
)
```

### ReLU (Rectified Linear Unit)

Función de activación que retorna max(0, x). La más usada en capas ocultas.

$$f(x) = \max(0, x)$$

---

## S

### Sequential

API de Keras para crear modelos como una pila lineal de capas.

```python
model = Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### Softmax

Función de activación que convierte logits en probabilidades (suman 1). Usada en la capa de salida para clasificación multiclase.

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

### Sparse Categorical Crossentropy

Versión de categorical crossentropy para etiquetas como enteros (no one-hot encoded).

```python
loss='sparse_categorical_crossentropy'
```

---

## T

### TensorBoard

Herramienta de visualización de TensorFlow para monitorear el entrenamiento.

```bash
tensorboard --logdir logs/fit
```

### TensorFlow

Framework de código abierto de Google para computación numérica y machine learning.

### Tensor

Array n-dimensional, estructura de datos fundamental en TensorFlow.

```python
tensor = tf.constant([[1, 2], [3, 4]])
```

### Trainable

Propiedad que indica si los pesos de una capa se actualizan durante el entrenamiento.

---

## V

### Validation Split

Porcentaje de datos de entrenamiento reservados para validación.

```python
model.fit(X, y, validation_split=0.2)  # 20% para validación
```

### Variable

Tensor mutable en TensorFlow, usado para almacenar pesos entrenables.

```python
weights = tf.Variable(tf.random.normal([3, 2]))
```

---

_Glosario Semana 20 | TensorFlow y Keras | Bootcamp IA: Zero to Hero_
