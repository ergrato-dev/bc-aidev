"""
Ejercicio 1: Tensores Básicos en TensorFlow
===========================================

Descomenta cada sección según avances en el README.
Ejecuta después de cada sección para verificar resultados.
"""

# ============================================
# PASO 1: Configuración y Verificación
# ============================================
print("--- Paso 1: Configuración y Verificación ---")

import numpy as np
import tensorflow as tf

# Verificar instalación
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Configurar seed para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

print()

# ============================================
# PASO 2: Creación de Tensores
# ============================================
print("--- Paso 2: Creación de Tensores ---")

# Descomenta las siguientes líneas:

# # Escalar (0-D): un solo valor
# scalar = tf.constant(42)
# print(f"Escalar: {scalar}")
# print(f"  Shape: {scalar.shape}")
# print(f"  dtype: {scalar.dtype}")

# # Vector (1-D): array de valores
# vector = tf.constant([1.0, 2.0, 3.0, 4.0])
# print(f"\nVector: {vector}")
# print(f"  Shape: {vector.shape}")

# # Matriz (2-D): filas y columnas
# matrix = tf.constant([[1, 2, 3],
#                       [4, 5, 6]])
# print(f"\nMatriz:\n{matrix}")
# print(f"  Shape: {matrix.shape}")

# # Tensor 3-D
# tensor_3d = tf.random.normal([2, 3, 4])
# print(f"\nTensor 3D shape: {tensor_3d.shape}")

print()

# ============================================
# PASO 3: Tipos de Datos (dtypes)
# ============================================
print("--- Paso 3: Tipos de Datos ---")

# Descomenta las siguientes líneas:

# # Float32: más común para redes neuronales
# weights = tf.constant([0.1, 0.2, 0.3], dtype=tf.float32)
# print(f"Weights dtype: {weights.dtype}")

# # Int32: para índices y etiquetas
# labels = tf.constant([0, 1, 2], dtype=tf.int32)
# print(f"Labels dtype: {labels.dtype}")

# # Conversión de tipos
# converted = tf.cast(labels, tf.float32)
# print(f"Convertido a float32: {converted}")

# # Verificar tipos disponibles
# print(f"\nTipos comunes: tf.float32, tf.float64, tf.int32, tf.int64, tf.bool")

print()

# ============================================
# PASO 4: Operaciones Matemáticas
# ============================================
print("--- Paso 4: Operaciones Matemáticas ---")

# Descomenta las siguientes líneas:

# a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
# b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# # Operaciones elemento a elemento
# suma = a + b
# resta = a - b
# producto = a * b
# division = a / b

# print(f"a:\n{a.numpy()}")
# print(f"b:\n{b.numpy()}")
# print(f"\nSuma (a+b):\n{suma.numpy()}")
# print(f"\nProducto elemento a elemento (a*b):\n{producto.numpy()}")

# # Multiplicación matricial
# matmul = a @ b  # equivalente a tf.matmul(a, b)
# print(f"\nMultiplicación matricial (a @ b):\n{matmul.numpy()}")

# # Funciones matemáticas
# print(f"\nRaíz cuadrada de a:\n{tf.sqrt(a).numpy()}")
# print(f"\nExponencial de a:\n{tf.exp(a).numpy()}")

print()

# ============================================
# PASO 5: Reducción y Agregación
# ============================================
print("--- Paso 5: Reducción y Agregación ---")

# Descomenta las siguientes líneas:

# data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
# print(f"Data:\n{data.numpy()}")

# # Reducción sobre todo el tensor
# print(f"\nSuma total: {tf.reduce_sum(data).numpy()}")
# print(f"Media: {tf.reduce_mean(data).numpy()}")
# print(f"Máximo: {tf.reduce_max(data).numpy()}")
# print(f"Mínimo: {tf.reduce_min(data).numpy()}")

# # Reducción sobre ejes específicos
# print(f"\nSuma por filas (axis=1): {tf.reduce_sum(data, axis=1).numpy()}")
# print(f"Suma por columnas (axis=0): {tf.reduce_sum(data, axis=0).numpy()}")

# # Argmax/Argmin
# print(f"\nÍndice del máximo por fila: {tf.argmax(data, axis=1).numpy()}")

print()

# ============================================
# PASO 6: Reshape y Manipulación
# ============================================
print("--- Paso 6: Reshape y Manipulación ---")

# Descomenta las siguientes líneas:

# original = tf.range(12)
# print(f"Original: {original.numpy()}")
# print(f"Shape: {original.shape}")

# # Reshape a matriz
# matriz = tf.reshape(original, [3, 4])
# print(f"\nReshape a (3, 4):\n{matriz.numpy()}")

# # Reshape con -1 (inferir dimensión)
# matriz_auto = tf.reshape(original, [4, -1])  # -1 se calcula automáticamente
# print(f"\nReshape a (4, -1):\n{matriz_auto.numpy()}")

# # Aplanar
# aplanado = tf.reshape(matriz, [-1])
# print(f"\nAplanado: {aplanado.numpy()}")

# # Expand dims (añadir dimensión para batch)
# con_batch = tf.expand_dims(matriz, axis=0)
# print(f"\nCon batch dimension: {con_batch.shape}")

# # Squeeze (eliminar dimensiones de tamaño 1)
# squeezed = tf.squeeze(con_batch)
# print(f"Después de squeeze: {squeezed.shape}")

# # Transponer
# transpuesta = tf.transpose(matriz)
# print(f"\nTranspuesta shape: {matriz.shape} -> {transpuesta.shape}")

print()

# ============================================
# PASO 7: Broadcasting
# ============================================
print("--- Paso 7: Broadcasting ---")

# Descomenta las siguientes líneas:

# # Escalar + tensor
# tensor = tf.constant([[1, 2], [3, 4]])
# resultado_escalar = tensor + 10
# print(f"Tensor:\n{tensor.numpy()}")
# print(f"\nTensor + 10:\n{resultado_escalar.numpy()}")

# # Vector + matriz (broadcasting por filas)
# vector = tf.constant([10, 20])
# resultado_vector = tensor + vector
# print(f"\nVector: {vector.numpy()}")
# print(f"\nTensor + vector (broadcast):\n{resultado_vector.numpy()}")

# # Vector columna + matriz
# vector_col = tf.constant([[100], [200]])
# resultado_col = tensor + vector_col
# print(f"\nVector columna:\n{vector_col.numpy()}")
# print(f"\nTensor + vector columna:\n{resultado_col.numpy()}")

print()

# ============================================
# PASO 8: Variables (Pesos Entrenables)
# ============================================
print("--- Paso 8: Variables ---")

# Descomenta las siguientes líneas:

# # Crear variable (mutable, para pesos de red)
# weights = tf.Variable(
#     tf.random.normal([3, 2]),
#     trainable=True,
#     name='layer_weights'
# )
# print(f"Variable inicial:\n{weights.numpy()}")
# print(f"Trainable: {weights.trainable}")

# # Modificar valor
# weights.assign(weights * 2)
# print(f"\nDespués de multiplicar por 2:\n{weights.numpy()}")

# # Sumar
# weights.assign_add(tf.ones([3, 2]))
# print(f"\nDespués de sumar 1:\n{weights.numpy()}")

# # Diferencia con tf.constant
# constante = tf.constant([1, 2, 3])
# # constante[0] = 5  # ERROR: los tensores constantes son inmutables
# print(f"\nConstantes son inmutables, Variables son mutables")

print()

# ============================================
# PASO 9: GradientTape
# ============================================
print("--- Paso 9: GradientTape ---")

# Descomenta las siguientes líneas:

# # Ejemplo simple: derivada de x²
# x = tf.Variable(3.0)

# with tf.GradientTape() as tape:
#     y = x ** 2  # y = x²

# # dy/dx = 2x, cuando x=3 → gradiente = 6
# grad = tape.gradient(y, x)
# print(f"x = {x.numpy()}")
# print(f"y = x² = {y.numpy()}")
# print(f"dy/dx = 2x = {grad.numpy()}")

# # Ejemplo con múltiples variables (simulando red neuronal)
# w = tf.Variable([2.0, 3.0])
# b = tf.Variable(1.0)
# x_input = tf.constant([1.0, 2.0])

# with tf.GradientTape() as tape:
#     # Forward: y = w·x + b
#     y_pred = tf.reduce_sum(w * x_input) + b
#     # Loss: (y_pred - y_true)²
#     y_true = 10.0
#     loss = (y_pred - y_true) ** 2

# # Calcular gradientes
# gradients = tape.gradient(loss, [w, b])
# print(f"\nGradiente respecto a w: {gradients[0].numpy()}")
# print(f"Gradiente respecto a b: {gradients[1].numpy()}")

print()

# ============================================
# PASO 10: Interoperabilidad con NumPy
# ============================================
print("--- Paso 10: Interoperabilidad con NumPy ---")

# Descomenta las siguientes líneas:

# # NumPy a TensorFlow
# np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
# tensor_from_np = tf.convert_to_tensor(np_array)
# print(f"NumPy array:\n{np_array}")
# print(f"Tensor de NumPy:\n{tensor_from_np}")

# # TensorFlow a NumPy
# tf_tensor = tf.constant([[5, 6], [7, 8]])
# numpy_from_tf = tf_tensor.numpy()
# print(f"\nTensor TF:\n{tf_tensor}")
# print(f"Como NumPy:\n{numpy_from_tf}")
# print(f"Tipo: {type(numpy_from_tf)}")

# # Operaciones mixtas (automáticas)
# resultado_mixto = tf_tensor + np_array
# print(f"\nOperación mixta (TF + NumPy):\n{resultado_mixto.numpy()}")

print()

# ============================================
# RESUMEN FINAL
# ============================================
print("=" * 50)
print("RESUMEN DE CONCEPTOS APRENDIDOS")
print("=" * 50)
print(
    """
✅ tf.constant() - Tensores inmutables
✅ tf.Variable() - Tensores mutables (pesos)
✅ Operaciones matemáticas vectorizadas
✅ tf.reduce_* - Agregaciones
✅ tf.reshape() - Cambiar forma
✅ Broadcasting automático
✅ GradientTape - Diferenciación automática
✅ Interoperabilidad con NumPy
"""
)
