# ⚙️ Ejercicio 3: Callbacks y Checkpoints

## 🎯 Objetivo

Dominar el uso de callbacks en Keras para controlar el proceso de entrenamiento, incluyendo early stopping, guardado de modelos, y visualización con TensorBoard.

## ⏱️ Duración

55 minutos

## 📋 Instrucciones

Sigue cada paso en orden. Este ejercicio entrena un modelo real con el dataset MNIST.

---

## Paso 1: Preparación de Datos

Cargamos y preparamos el dataset MNIST:

```python
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizar y aplanar
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0
```

---

## Paso 2: Crear Modelo Base

Un modelo simple para experimentar:

```python
model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Paso 3: EarlyStopping Básico

Detener cuando la validación deja de mejorar:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_split=0.1,
    callbacks=[early_stop]
)
```

---

## Paso 4: ModelCheckpoint

Guardar el mejor modelo durante entrenamiento:

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

model.fit(
    X_train, y_train,
    epochs=20,
    validation_split=0.1,
    callbacks=[checkpoint]
)
```

---

## Paso 5: Múltiples Checkpoints

Guardar modelos con información de la época:

```python
checkpoint_epochs = ModelCheckpoint(
    filepath='models/model_epoch_{epoch:02d}_acc_{val_accuracy:.3f}.keras',
    save_best_only=False,
    save_freq='epoch'
)
```

---

## Paso 6: ReduceLROnPlateau

Reducir learning rate cuando el entrenamiento se estanca:

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)
```

---

## Paso 7: TensorBoard

Visualización en tiempo real del entrenamiento:

```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

# Para visualizar: tensorboard --logdir logs/fit
```

---

## Paso 8: Custom Callback

Crear tu propio callback:

```python
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > 0.98:
            print(f"\n¡Accuracy > 98% alcanzada en época {epoch}!")
            self.model.stop_training = True
```

---

## Paso 9: Combinar Todos los Callbacks

Entrenamiento completo con todas las herramientas:

```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    TensorBoard(log_dir='./logs')
]

history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_split=0.1,
    callbacks=callbacks
)
```

---

## Paso 10: Cargar y Evaluar Modelo Guardado

Recuperar el mejor modelo:

```python
# Cargar modelo guardado
loaded_model = tf.keras.models.load_model('best_model.keras')

# Evaluar
test_loss, test_acc = loaded_model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Predecir
predictions = loaded_model.predict(X_test[:5])
```

---

## Paso 11: Visualizar History

Graficar el progreso del entrenamiento:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(history.history['loss'], label='Train')
axes[0].plot(history.history['val_loss'], label='Val')
axes[0].set_title('Loss')
axes[0].legend()

# Accuracy
axes[1].plot(history.history['accuracy'], label='Train')
axes[1].plot(history.history['val_accuracy'], label='Val')
axes[1].set_title('Accuracy')
axes[1].legend()

plt.savefig('training_history.png')
plt.show()
```

---

## ✅ Checklist de Completado

- [ ] Dataset MNIST cargado y preprocesado
- [ ] Modelo base creado y compilado
- [ ] EarlyStopping configurado
- [ ] ModelCheckpoint guardando mejor modelo
- [ ] ReduceLROnPlateau implementado
- [ ] TensorBoard configurado
- [ ] Custom callback creado
- [ ] Todos los callbacks combinados
- [ ] Modelo cargado y evaluado
- [ ] Gráficas de entrenamiento generadas

---

## 🎯 Resultado Esperado

```
Epoch 15/50
...
EarlyStopping: stopped at epoch 15
Restoring model weights from the best epoch.

Test accuracy: 0.9756

Modelo guardado en: best_model.keras
```
