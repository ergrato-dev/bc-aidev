"""
Proyecto: Clasificador MNIST con Keras
======================================

Objetivo: Construir un clasificador de dígitos con ≥97% accuracy

Implementa cada sección marcada con TODO.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist

# Reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

# Crear directorio para outputs
os.makedirs("outputs", exist_ok=True)

# ============================================
# PASO 1: CARGAR Y EXPLORAR DATOS
# ============================================
print("=" * 50)
print("PASO 1: Cargar y Explorar Datos")
print("=" * 50)

# Cargar MNIST
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

print(f"Train shape: {X_train_full.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Valores: min={X_train_full.min()}, max={X_train_full.max()}")
print(f"Clases únicas: {np.unique(y_train_full)}")

# Visualizar algunas muestras
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train_full[i], cmap="gray")
    ax.set_title(f"Label: {y_train_full[i]}")
    ax.axis("off")
plt.suptitle("Muestras del Dataset MNIST")
plt.tight_layout()
plt.savefig("outputs/mnist_samples.png", dpi=150)
plt.close()
print("Muestras guardadas en: outputs/mnist_samples.png")

print()

# ============================================
# PASO 2: PREPROCESAMIENTO
# ============================================
print("=" * 50)
print("PASO 2: Preprocesamiento")
print("=" * 50)

# TODO: Implementar preprocesamiento
# 1. Aplanar imágenes de (28, 28) a (784,)
# 2. Normalizar valores de 0-255 a 0-1
# 3. Convertir a float32

# X_train_full = ...
# X_test = ...

# TODO: Separar conjunto de validación (10% del train)
# n_val = ...
# X_val, y_val = ...
# X_train, y_train = ...

# Placeholder - Reemplazar con tu implementación
X_train_full = X_train_full.reshape(-1, 784).astype("float32") / 255.0
X_test = X_test.reshape(-1, 784).astype("float32") / 255.0

n_val = 6000
X_val, y_val = X_train_full[:n_val], y_train_full[:n_val]
X_train, y_train = X_train_full[n_val:], y_train_full[n_val:]

print(f"Train: {X_train.shape}")
print(f"Val: {X_val.shape}")
print(f"Test: {X_test.shape}")
print(f"Valores normalizados: min={X_train.min():.2f}, max={X_train.max():.2f}")

print()

# ============================================
# PASO 3: DISEÑAR ARQUITECTURA
# ============================================
print("=" * 50)
print("PASO 3: Diseñar Arquitectura")
print("=" * 50)

# TODO: Crear modelo Sequential
# Requisitos:
# - Input shape: (784,)
# - Mínimo 2 capas ocultas
# - Usar ReLU en capas ocultas
# - Añadir Dropout para regularización
# - BatchNormalization opcional
# - Capa de salida: 10 neuronas con softmax


def create_model():
    """
    Crea el modelo de clasificación MNIST.

    Returns:
        model: Modelo Sequential compilado
    """
    # TODO: Implementar arquitectura
    model = Sequential(
        [
            # Capa de entrada implícita
            # TODO: Añadir capas ocultas
            # TODO: Añadir regularización
            # TODO: Capa de salida
            # Placeholder - Reemplazar con tu arquitectura
            layers.Dense(10, activation="softmax", input_shape=(784,))
        ],
        name="mnist_classifier",
    )

    return model


model = create_model()
model.summary()

print()

# ============================================
# PASO 4: COMPILAR MODELO
# ============================================
print("=" * 50)
print("PASO 4: Compilar Modelo")
print("=" * 50)

# TODO: Compilar el modelo
# - optimizer: 'adam' o Adam con learning_rate específico
# - loss: 'sparse_categorical_crossentropy'
# - metrics: ['accuracy']

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print("Modelo compilado")
print(f"Optimizer: {model.optimizer.__class__.__name__}")
print(f"Loss: {model.loss}")

print()

# ============================================
# PASO 5: CONFIGURAR CALLBACKS
# ============================================
print("=" * 50)
print("PASO 5: Configurar Callbacks")
print("=" * 50)

# TODO: Implementar callbacks
# 1. EarlyStopping: monitor='val_loss', patience=5, restore_best_weights=True
# 2. ModelCheckpoint: guardar mejor modelo en 'outputs/best_model.keras'
# 3. (Opcional) ReduceLROnPlateau

callbacks = [
    # TODO: Añadir EarlyStopping
    # TODO: Añadir ModelCheckpoint
    # TODO: (Opcional) Añadir ReduceLROnPlateau
]

# Placeholder - Reemplazar con tus callbacks
if not callbacks:
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            "outputs/best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

print(f"Callbacks configurados: {len(callbacks)}")

print()

# ============================================
# PASO 6: ENTRENAR MODELO
# ============================================
print("=" * 50)
print("PASO 6: Entrenar Modelo")
print("=" * 50)

# TODO: Entrenar el modelo
# - epochs: suficientes para que EarlyStopping actúe (~30-50)
# - batch_size: 64-128
# - validation_data: (X_val, y_val)
# - callbacks: los configurados arriba

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1,
)

print(f"\nEntrenamiento completado en {len(history.history['loss'])} épocas")
print(f"Mejor val_accuracy: {max(history.history['val_accuracy']):.4f}")

print()

# ============================================
# PASO 7: EVALUAR EN TEST
# ============================================
print("=" * 50)
print("PASO 7: Evaluar en Test")
print("=" * 50)

# Cargar mejor modelo
best_model = tf.keras.models.load_model("outputs/best_model.keras")

# Evaluar
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Verificar objetivo
if test_acc >= 0.97:
    print("✅ ¡Objetivo alcanzado! (≥97%)")
elif test_acc >= 0.95:
    print("⚠️ Cerca del objetivo (≥95%, <97%)")
else:
    print("❌ Por debajo del mínimo (<95%)")

print()

# ============================================
# PASO 8: VISUALIZACIONES
# ============================================
print("=" * 50)
print("PASO 8: Visualizaciones")
print("=" * 50)

# --- Curvas de entrenamiento ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(history.history["loss"], label="Train", linewidth=2)
axes[0].plot(history.history["val_loss"], label="Validation", linewidth=2)
axes[0].set_title("Loss over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history.history["accuracy"], label="Train", linewidth=2)
axes[1].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
axes[1].set_title("Accuracy over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/training_history.png", dpi=150)
plt.close()
print("Curvas guardadas en: outputs/training_history.png")

# --- Matriz de confusión ---
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

y_pred = best_model.predict(X_test, verbose=0).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Confusion Matrix - MNIST Classifier")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()
print("Matriz de confusión guardada en: outputs/confusion_matrix.png")

# --- Predicciones incorrectas ---
incorrect_mask = y_pred != y_test
incorrect_indices = np.where(incorrect_mask)[0]

if len(incorrect_indices) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        if i < len(incorrect_indices):
            idx = incorrect_indices[i]
            img = X_test[idx].reshape(28, 28)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Pred: {y_pred[idx]}, Real: {y_test[idx]}", color="red")
        ax.axis("off")
    plt.suptitle("Predicciones Incorrectas")
    plt.tight_layout()
    plt.savefig("outputs/incorrect_predictions.png", dpi=150)
    plt.close()
    print("Predicciones incorrectas guardadas en: outputs/incorrect_predictions.png")

print()

# ============================================
# RESUMEN FINAL
# ============================================
print("=" * 50)
print("RESUMEN DEL PROYECTO")
print("=" * 50)
print(
    f"""
📊 Dataset MNIST:
   - Train: {X_train.shape[0]:,} muestras
   - Validation: {X_val.shape[0]:,} muestras  
   - Test: {X_test.shape[0]:,} muestras

🏗️ Arquitectura:
   - Parámetros totales: {model.count_params():,}
   - Capas: {len(model.layers)}

📈 Resultados:
   - Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)
   - Test Loss: {test_loss:.4f}
   - Épocas entrenadas: {len(history.history['loss'])}
   - Predicciones incorrectas: {incorrect_mask.sum()} / {len(y_test)}

📁 Archivos generados:
   - outputs/best_model.keras
   - outputs/training_history.png
   - outputs/confusion_matrix.png
   - outputs/incorrect_predictions.png
   - outputs/mnist_samples.png
"""
)

# Verificación final
if test_acc >= 0.97:
    print("✅ PROYECTO COMPLETADO EXITOSAMENTE")
else:
    print("⚠️ Revisa tu arquitectura para alcanzar ≥97% accuracy")
