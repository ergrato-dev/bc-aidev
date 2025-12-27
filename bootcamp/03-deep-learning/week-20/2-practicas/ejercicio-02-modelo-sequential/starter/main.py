"""
Ejercicio 2: Modelo Sequential en Keras
=======================================

Descomenta cada sección según avances en el README.
"""

# ============================================
# PASO 1: Importaciones y Configuración
# ============================================
print("--- Paso 1: Importaciones y Configuración ---")

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers

# Reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

print(f"TensorFlow version: {tf.__version__}")
print()

# ============================================
# PASO 2: Crear Modelo con Lista de Capas
# ============================================
print("--- Paso 2: Modelo con Lista de Capas ---")

# Descomenta las siguientes líneas:

# # Forma más común de crear un modelo Sequential
# model_list = Sequential([
#     layers.Dense(64, activation='relu', input_shape=(784,)),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])

# print("Modelo creado con lista de capas:")
# model_list.summary()

print()

# ============================================
# PASO 3: Crear Modelo con .add()
# ============================================
print("--- Paso 3: Modelo con .add() ---")

# Descomenta las siguientes líneas:

# # Forma alternativa usando .add()
# model_add = Sequential()
# model_add.add(layers.InputLayer(input_shape=(784,)))
# model_add.add(layers.Dense(64, activation='relu'))
# model_add.add(layers.Dense(32, activation='relu'))
# model_add.add(layers.Dense(10, activation='softmax'))

# print("Modelo creado con .add():")
# model_add.summary()

print()

# ============================================
# PASO 4: Nombrar Modelos y Capas
# ============================================
print("--- Paso 4: Nombrar Modelos y Capas ---")

# Descomenta las siguientes líneas:

# model_named = Sequential([
#     layers.Dense(64, activation='relu', name='hidden_1', input_shape=(784,)),
#     layers.Dense(32, activation='relu', name='hidden_2'),
#     layers.Dense(10, activation='softmax', name='output')
# ], name='mnist_classifier')

# print("Modelo con nombres personalizados:")
# model_named.summary()

# # Acceder a capas por nombre
# hidden_layer = model_named.get_layer('hidden_1')
# print(f"\nCapa 'hidden_1':")
# print(f"  Nombre: {hidden_layer.name}")
# print(f"  Units: {hidden_layer.units}")
# print(f"  Activation: {hidden_layer.activation.__name__}")

print()

# ============================================
# PASO 5: Inspección Detallada
# ============================================
print("--- Paso 5: Inspección Detallada ---")

# Descomenta las siguientes líneas:

# # Crear modelo para inspeccionar
# model_inspect = Sequential([
#     layers.Dense(128, activation='relu', input_shape=(784,)),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ], name='model_to_inspect')

# print("Inspección detallada de capas:")
# print("-" * 50)

# for i, layer in enumerate(model_inspect.layers):
#     print(f"\nCapa {i}: {layer.name}")
#     print(f"  Tipo: {type(layer).__name__}")
#     print(f"  Input shape: {layer.input_shape}")
#     print(f"  Output shape: {layer.output_shape}")
#     print(f"  Parámetros: {layer.count_params():,}")

#     # Obtener pesos si la capa tiene
#     if layer.weights:
#         weights, biases = layer.get_weights()
#         print(f"  Weights shape: {weights.shape}")
#         print(f"  Biases shape: {biases.shape}")

# print(f"\nTotal de parámetros: {model_inspect.count_params():,}")

print()

# ============================================
# PASO 6: Modelo para Clasificación Binaria
# ============================================
print("--- Paso 6: Clasificación Binaria ---")

# Descomenta las siguientes líneas:

# # Arquitectura para clasificación binaria (2 clases)
# model_binary = Sequential([
#     layers.Dense(64, activation='relu', input_shape=(20,)),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(16, activation='relu'),
#     layers.Dense(1, activation='sigmoid')  # UNA neurona con sigmoid
# ], name='binary_classifier')

# print("Modelo para clasificación binaria:")
# model_binary.summary()

# # Probar con datos sintéticos
# X_binary = np.random.randn(3, 20).astype('float32')
# pred_binary = model_binary.predict(X_binary, verbose=0)
# print(f"\nPredicciones (probabilidades entre 0 y 1):")
# print(pred_binary.flatten())

print()

# ============================================
# PASO 7: Modelo para Regresión
# ============================================
print("--- Paso 7: Modelo para Regresión ---")

# Descomenta las siguientes líneas:

# # Arquitectura para regresión (predicción de valores continuos)
# model_regression = Sequential([
#     layers.Dense(64, activation='relu', input_shape=(13,)),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(16, activation='relu'),
#     layers.Dense(1)  # Sin activación para valores continuos
# ], name='regression_model')

# print("Modelo para regresión:")
# model_regression.summary()

# # Probar con datos sintéticos
# X_reg = np.random.randn(3, 13).astype('float32')
# pred_reg = model_regression.predict(X_reg, verbose=0)
# print(f"\nPredicciones (valores continuos):")
# print(pred_reg.flatten())

print()

# ============================================
# PASO 8: Modelo con Regularización
# ============================================
print("--- Paso 8: Modelo con Regularización ---")

# Descomenta las siguientes líneas:

# # Modelo con Dropout y BatchNormalization
# model_regularized = Sequential([
#     # Bloque 1
#     layers.Dense(128, input_shape=(784,)),
#     layers.BatchNormalization(),
#     layers.Activation('relu'),
#     layers.Dropout(0.3),

#     # Bloque 2
#     layers.Dense(64),
#     layers.BatchNormalization(),
#     layers.Activation('relu'),
#     layers.Dropout(0.3),

#     # Bloque 3
#     layers.Dense(32),
#     layers.BatchNormalization(),
#     layers.Activation('relu'),
#     layers.Dropout(0.2),

#     # Salida
#     layers.Dense(10, activation='softmax')
# ], name='regularized_classifier')

# print("Modelo con regularización:")
# model_regularized.summary()

# # Contar tipos de capas
# layer_types = {}
# for layer in model_regularized.layers:
#     layer_type = type(layer).__name__
#     layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
# print(f"\nTipos de capas: {layer_types}")

print()

# ============================================
# PASO 9: Modelo con Inicializadores Personalizados
# ============================================
print("--- Paso 9: Inicializadores Personalizados ---")

# Descomenta las siguientes líneas:

# # Modelo con inicializadores específicos
# model_custom_init = Sequential([
#     # He initialization (mejor para ReLU)
#     layers.Dense(64, activation='relu',
#                  kernel_initializer='he_normal',
#                  bias_initializer='zeros',
#                  input_shape=(784,)),

#     # He uniform
#     layers.Dense(32, activation='relu',
#                  kernel_initializer='he_uniform'),

#     # Glorot (Xavier) para softmax
#     layers.Dense(10, activation='softmax',
#                  kernel_initializer='glorot_uniform')
# ], name='custom_init_model')

# print("Modelo con inicializadores personalizados:")
# model_custom_init.summary()

# # Ver inicializadores de cada capa
# print("\nInicializadores usados:")
# for layer in model_custom_init.layers:
#     if hasattr(layer, 'kernel_initializer'):
#         print(f"  {layer.name}: {layer.kernel_initializer.__class__.__name__}")

print()

# ============================================
# PASO 10: Probar con Datos de Prueba
# ============================================
print("--- Paso 10: Probar con Datos ---")

# Descomenta las siguientes líneas:

# # Crear modelo final para pruebas
# model_final = Sequential([
#     layers.Dense(128, activation='relu', input_shape=(784,)),
#     layers.Dropout(0.2),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(10, activation='softmax')
# ], name='final_test_model')

# # Datos sintéticos (simular imágenes MNIST aplanadas)
# n_samples = 5
# X_dummy = np.random.randn(n_samples, 784).astype('float32')

# # Forward pass (predicción sin entrenar)
# predictions = model_final.predict(X_dummy, verbose=0)

# print(f"Datos de entrada shape: {X_dummy.shape}")
# print(f"Predicciones shape: {predictions.shape}")
# print(f"\nPredicciones (probabilidades por clase):")
# print(predictions.round(3))

# print(f"\nSuma de probabilidades por muestra:")
# print(predictions.sum(axis=1).round(3))  # Debe ser ~1 para cada muestra

# print(f"\nClase predicha por muestra:")
# print(predictions.argmax(axis=1))

print()

# ============================================
# RESUMEN FINAL
# ============================================
print("=" * 50)
print("RESUMEN DE CONCEPTOS APRENDIDOS")
print("=" * 50)
print(
    """
✅ Sequential con lista de capas
✅ Sequential con .add()
✅ Nombrar modelos y capas
✅ Inspección con summary() y get_layer()
✅ Arquitectura para clasificación binaria (sigmoid)
✅ Arquitectura para regresión (sin activación)
✅ Regularización: Dropout + BatchNormalization
✅ Inicializadores personalizados (he_normal, glorot)
✅ Forward pass con predict()
"""
)
