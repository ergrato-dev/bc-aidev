"""
Ejercicio 3: Callbacks y Checkpoints
====================================

Este ejercicio entrena un modelo real con MNIST.
Descomenta cada sección según avances en el README.
"""

# ============================================
# PASO 1: Preparación de Datos
# ============================================
print("--- Paso 1: Preparación de Datos ---")

import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.datasets import mnist

# Reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

# Cargar MNIST
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Normalizar y aplanar
X_train_full = X_train_full.reshape(-1, 784).astype("float32") / 255.0
X_test = X_test.reshape(-1, 784).astype("float32") / 255.0

# Separar validación
n_val = 6000
X_val = X_train_full[:n_val]
y_val = y_train_full[:n_val]
X_train = X_train_full[n_val:]
y_train = y_train_full[n_val:]

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Val: {X_val.shape}, {y_val.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

print()

# ============================================
# PASO 2: Crear Modelo Base
# ============================================
print("--- Paso 2: Crear Modelo Base ---")


def create_model():
    """Crea un modelo nuevo (para reiniciar experimentos)."""
    model = Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=(784,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ],
        name="mnist_model",
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


model = create_model()
model.summary()

print()

# ============================================
# PASO 3: EarlyStopping Básico
# ============================================
print("--- Paso 3: EarlyStopping ---")

# Descomenta las siguientes líneas:

# # Crear modelo nuevo
# model_es = create_model()

# # EarlyStopping callback
# early_stop = EarlyStopping(
#     monitor='val_loss',        # Métrica a monitorear
#     patience=3,                # Épocas sin mejora antes de parar
#     restore_best_weights=True, # Restaurar mejores pesos
#     min_delta=0.001,           # Mejora mínima considerada
#     mode='min',                # 'min' para loss, 'max' para accuracy
#     verbose=1
# )

# print("Entrenando con EarlyStopping (patience=3)...")
# history_es = model_es.fit(
#     X_train, y_train,
#     epochs=30,
#     batch_size=128,
#     validation_data=(X_val, y_val),
#     callbacks=[early_stop],
#     verbose=1
# )

# print(f"\nEntrenamiento terminó en época {len(history_es.history['loss'])}")
# print(f"Mejor val_loss: {min(history_es.history['val_loss']):.4f}")

print()

# ============================================
# PASO 4: ModelCheckpoint Básico
# ============================================
print("--- Paso 4: ModelCheckpoint ---")

# Descomenta las siguientes líneas:

# # Crear modelo nuevo
# model_ckpt = create_model()

# # Crear directorio para modelos
# os.makedirs('saved_models', exist_ok=True)

# # ModelCheckpoint callback
# checkpoint = ModelCheckpoint(
#     filepath='saved_models/best_model.keras',
#     monitor='val_accuracy',
#     save_best_only=True,       # Solo guardar si mejora
#     save_weights_only=False,   # Guardar modelo completo
#     mode='max',
#     verbose=1
# )

# print("Entrenando con ModelCheckpoint...")
# history_ckpt = model_ckpt.fit(
#     X_train, y_train,
#     epochs=10,
#     batch_size=128,
#     validation_data=(X_val, y_val),
#     callbacks=[checkpoint],
#     verbose=1
# )

# print(f"\nMejor modelo guardado en: saved_models/best_model.keras")

print()

# ============================================
# PASO 5: Múltiples Checkpoints
# ============================================
print("--- Paso 5: Checkpoints por Época ---")

# Descomenta las siguientes líneas:

# # Crear modelo nuevo
# model_multi = create_model()

# # Crear directorio para checkpoints
# os.makedirs('checkpoints', exist_ok=True)

# # Checkpoint que guarda cada época con info en el nombre
# checkpoint_epochs = ModelCheckpoint(
#     filepath='checkpoints/model_epoch_{epoch:02d}_acc_{val_accuracy:.3f}.keras',
#     save_best_only=False,  # Guardar todos
#     save_freq='epoch',     # Cada época
#     verbose=0
# )

# print("Entrenando guardando cada época...")
# model_multi.fit(
#     X_train, y_train,
#     epochs=5,
#     batch_size=128,
#     validation_data=(X_val, y_val),
#     callbacks=[checkpoint_epochs],
#     verbose=1
# )

# # Listar archivos guardados
# print("\nCheckpoints guardados:")
# for f in sorted(os.listdir('checkpoints')):
#     print(f"  {f}")

print()

# ============================================
# PASO 6: ReduceLROnPlateau
# ============================================
print("--- Paso 6: ReduceLROnPlateau ---")

# Descomenta las siguientes líneas:

# # Crear modelo nuevo
# model_lr = create_model()

# # ReduceLROnPlateau callback
# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,         # Nuevo LR = LR * 0.5
#     patience=2,         # Épocas sin mejora antes de reducir
#     min_lr=1e-6,        # LR mínimo
#     verbose=1
# )

# print("Entrenando con ReduceLROnPlateau...")
# history_lr = model_lr.fit(
#     X_train, y_train,
#     epochs=15,
#     batch_size=128,
#     validation_data=(X_val, y_val),
#     callbacks=[reduce_lr],
#     verbose=1
# )

print()

# ============================================
# PASO 7: TensorBoard
# ============================================
print("--- Paso 7: TensorBoard ---")

# Descomenta las siguientes líneas:

# # Crear modelo nuevo
# model_tb = create_model()

# # Crear directorio de logs con timestamp
# log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
# os.makedirs(log_dir, exist_ok=True)

# # TensorBoard callback
# tensorboard = TensorBoard(
#     log_dir=log_dir,
#     histogram_freq=1,      # Histogramas de pesos cada época
#     write_graph=True,      # Guardar grafo del modelo
#     update_freq='epoch'
# )

# print(f"Logs de TensorBoard en: {log_dir}")
# print("Para visualizar ejecutar: tensorboard --logdir logs/fit")

# history_tb = model_tb.fit(
#     X_train, y_train,
#     epochs=5,
#     batch_size=128,
#     validation_data=(X_val, y_val),
#     callbacks=[tensorboard],
#     verbose=1
# )

print()

# ============================================
# PASO 8: Custom Callback
# ============================================
print("--- Paso 8: Custom Callback ---")

# Descomenta las siguientes líneas:

# class AccuracyThresholdCallback(tf.keras.callbacks.Callback):
#     """Callback personalizado que detiene cuando se alcanza un umbral."""

#     def __init__(self, threshold=0.97):
#         super().__init__()
#         self.threshold = threshold

#     def on_epoch_end(self, epoch, logs=None):
#         val_acc = logs.get('val_accuracy', 0)
#         if val_acc > self.threshold:
#             print(f"\n¡Val accuracy {val_acc:.4f} > {self.threshold}!")
#             print(f"Deteniendo entrenamiento en época {epoch + 1}")
#             self.model.stop_training = True


# class LearningRateLogger(tf.keras.callbacks.Callback):
#     """Callback que muestra el learning rate actual."""

#     def on_epoch_begin(self, epoch, logs=None):
#         lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
#         print(f"  LR actual: {lr:.6f}")


# # Probar callbacks personalizados
# model_custom = create_model()

# custom_callbacks = [
#     AccuracyThresholdCallback(threshold=0.97),
#     LearningRateLogger()
# ]

# print("Entrenando con callbacks personalizados...")
# history_custom = model_custom.fit(
#     X_train, y_train,
#     epochs=20,
#     batch_size=128,
#     validation_data=(X_val, y_val),
#     callbacks=custom_callbacks,
#     verbose=1
# )

print()

# ============================================
# PASO 9: Combinar Todos los Callbacks
# ============================================
print("--- Paso 9: Todos los Callbacks Combinados ---")

# Descomenta las siguientes líneas:

# # Crear modelo nuevo
# model_full = create_model()

# # Directorio para este experimento
# os.makedirs('experiment', exist_ok=True)
# log_dir = f"experiment/logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

# # Todos los callbacks combinados
# all_callbacks = [
#     EarlyStopping(
#         monitor='val_loss',
#         patience=5,
#         restore_best_weights=True,
#         verbose=1
#     ),
#     ModelCheckpoint(
#         filepath='experiment/best_model.keras',
#         monitor='val_accuracy',
#         save_best_only=True,
#         verbose=1
#     ),
#     ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=3,
#         min_lr=1e-6,
#         verbose=1
#     ),
#     TensorBoard(
#         log_dir=log_dir,
#         histogram_freq=1
#     )
# ]

# print("Entrenamiento completo con todos los callbacks...")
# print("-" * 50)

# history_full = model_full.fit(
#     X_train, y_train,
#     epochs=50,
#     batch_size=128,
#     validation_data=(X_val, y_val),
#     callbacks=all_callbacks,
#     verbose=1
# )

# print("\nEntrenamiento finalizado")
# print(f"Épocas completadas: {len(history_full.history['loss'])}")
# print(f"Mejor val_accuracy: {max(history_full.history['val_accuracy']):.4f}")

print()

# ============================================
# PASO 10: Cargar y Evaluar Modelo Guardado
# ============================================
print("--- Paso 10: Cargar y Evaluar ---")

# Descomenta las siguientes líneas:

# # Verificar que existe el modelo guardado
# model_path = 'experiment/best_model.keras'
# if os.path.exists(model_path):
#     print(f"Cargando modelo desde: {model_path}")

#     # Cargar modelo completo
#     loaded_model = tf.keras.models.load_model(model_path)

#     # Evaluar en test set
#     test_loss, test_acc = loaded_model.evaluate(X_test, y_test, verbose=0)
#     print(f"\nEvaluación en Test Set:")
#     print(f"  Loss: {test_loss:.4f}")
#     print(f"  Accuracy: {test_acc:.4f}")

#     # Hacer predicciones
#     predictions = loaded_model.predict(X_test[:5], verbose=0)
#     print(f"\nPredicciones para primeras 5 muestras:")
#     print(f"  Predichas: {predictions.argmax(axis=1)}")
#     print(f"  Reales:    {y_test[:5]}")
# else:
#     print(f"No se encontró modelo en {model_path}")
#     print("Ejecuta el paso 9 primero para entrenar y guardar un modelo")

print()

# ============================================
# PASO 11: Visualizar History
# ============================================
print("--- Paso 11: Visualizar History ---")

# Descomenta las siguientes líneas:

# import matplotlib.pyplot as plt

# # Verificar que hay history disponible
# if 'history_full' in dir() and history_full is not None:
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#     # Loss
#     axes[0].plot(history_full.history['loss'], label='Train', linewidth=2)
#     axes[0].plot(history_full.history['val_loss'], label='Validation', linewidth=2)
#     axes[0].set_title('Loss over Epochs', fontsize=12)
#     axes[0].set_xlabel('Epoch')
#     axes[0].set_ylabel('Loss')
#     axes[0].legend()
#     axes[0].grid(True, alpha=0.3)

#     # Accuracy
#     axes[1].plot(history_full.history['accuracy'], label='Train', linewidth=2)
#     axes[1].plot(history_full.history['val_accuracy'], label='Validation', linewidth=2)
#     axes[1].set_title('Accuracy over Epochs', fontsize=12)
#     axes[1].set_xlabel('Epoch')
#     axes[1].set_ylabel('Accuracy')
#     axes[1].legend()
#     axes[1].grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig('experiment/training_history.png', dpi=150)
#     print("Gráfica guardada en: experiment/training_history.png")
#     plt.show()
# else:
#     print("Ejecuta el paso 9 primero para generar history")

print()

# ============================================
# RESUMEN FINAL
# ============================================
print("=" * 50)
print("RESUMEN DE CONCEPTOS APRENDIDOS")
print("=" * 50)
print(
    """
✅ EarlyStopping - Detener cuando no mejora
✅ ModelCheckpoint - Guardar mejor modelo
✅ ReduceLROnPlateau - Ajustar learning rate
✅ TensorBoard - Visualización en tiempo real
✅ Custom Callbacks - Lógica personalizada
✅ Combinar múltiples callbacks
✅ Cargar y evaluar modelos guardados
✅ Visualizar métricas de entrenamiento
"""
)
