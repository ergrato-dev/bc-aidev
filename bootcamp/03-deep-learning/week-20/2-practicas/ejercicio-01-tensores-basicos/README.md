# 🔷 Ejercicio 1: Tensores Básicos en TensorFlow

## 🎯 Objetivo

Dominar las operaciones fundamentales con tensores de TensorFlow, comprendiendo su creación, manipulación y operaciones matemáticas.

## ⏱️ Duración

45 minutos

## 📋 Instrucciones

Sigue cada paso en orden, descomentando el código en `starter/main.py` según avances. Ejecuta después de cada sección para verificar los resultados.

---

## Paso 1: Configuración y Verificación

Primero verificamos la instalación de TensorFlow:

```python
import tensorflow as tf
import numpy as np

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")
```

**Abre `starter/main.py`** y ejecuta la sección de configuración.

---

## Paso 2: Creación de Tensores

TensorFlow soporta tensores de múltiples dimensiones:

```python
# Escalar (0-D): un solo valor
scalar = tf.constant(42)

# Vector (1-D): array de valores
vector = tf.constant([1.0, 2.0, 3.0, 4.0])

# Matriz (2-D): filas y columnas
matrix = tf.constant([[1, 2, 3],
                      [4, 5, 6]])

# Tensor 3-D: típico para batches de imágenes
tensor_3d = tf.random.normal([2, 3, 4])  # (batch, height, width)
```

Descomenta la sección correspondiente y observa los shapes y dtypes.

---

## Paso 3: Tipos de Datos (dtypes)

Es importante controlar el tipo de datos:

```python
# Float32: más común para redes neuronales
weights = tf.constant([0.1, 0.2, 0.3], dtype=tf.float32)

# Int32: para índices y etiquetas
labels = tf.constant([0, 1, 2], dtype=tf.int32)

# Conversión de tipos
converted = tf.cast(labels, tf.float32)
```

---

## Paso 4: Operaciones Matemáticas

TensorFlow soporta operaciones vectorizadas:

```python
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# Elemento a elemento
suma = a + b
producto = a * b

# Multiplicación matricial
matmul = a @ b  # o tf.matmul(a, b)

# Funciones matemáticas
raiz = tf.sqrt(a)
exp = tf.exp(a)
```

---

## Paso 5: Reducción y Agregación

Calcular estadísticas sobre tensores:

```python
data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

# Sobre todo el tensor
total = tf.reduce_sum(data)
media = tf.reduce_mean(data)

# Sobre ejes específicos
suma_filas = tf.reduce_sum(data, axis=1)     # [6, 15]
suma_columnas = tf.reduce_sum(data, axis=0)  # [5, 7, 9]
```

---

## Paso 6: Reshape y Manipulación

Cambiar la forma de los tensores:

```python
original = tf.range(12)  # [0, 1, 2, ..., 11]

# Reshape a matriz
matriz = tf.reshape(original, [3, 4])

# Aplanar
aplanado = tf.reshape(matriz, [-1])  # -1 infiere la dimensión

# Añadir dimensión (para batch)
con_batch = tf.expand_dims(matriz, axis=0)  # (1, 3, 4)

# Transponer
transpuesta = tf.transpose(matriz)
```

---

## Paso 7: Broadcasting

TensorFlow expande automáticamente dimensiones compatibles:

```python
# Escalar + tensor
tensor = tf.constant([[1, 2], [3, 4]])
resultado = tensor + 10  # suma 10 a cada elemento

# Vector + matriz
vector = tf.constant([10, 20])
resultado = tensor + vector  # suma a cada fila
```

---

## Paso 8: Variables (Pesos Entrenables)

Las Variables son tensores mutables:

```python
# Crear variable
weights = tf.Variable(tf.random.normal([3, 2]))
print(f"Inicial: {weights.numpy()}")

# Modificar
weights.assign(weights * 2)  # Multiplicar por 2
weights.assign_add(tf.ones([3, 2]))  # Sumar 1 a todo
print(f"Modificado: {weights.numpy()}")
```

---

## Paso 9: GradientTape

Diferenciación automática para backpropagation:

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2  # y = x²

# dy/dx = 2x, cuando x=3 → gradiente = 6
grad = tape.gradient(y, x)
print(f"Gradiente: {grad.numpy()}")
```

---

## Paso 10: Interoperabilidad con NumPy

TensorFlow y NumPy trabajan juntos:

```python
# NumPy a TensorFlow
np_array = np.array([1, 2, 3])
tensor = tf.convert_to_tensor(np_array)

# TensorFlow a NumPy
back_to_numpy = tensor.numpy()

# Operaciones mixtas
resultado = tensor + np_array  # Funciona directamente
```

---

## ✅ Checklist de Completado

- [ ] Verificación de TensorFlow instalado
- [ ] Creación de tensores de diferentes dimensiones
- [ ] Especificación de dtypes
- [ ] Operaciones matemáticas básicas
- [ ] Reducción y agregación
- [ ] Reshape y manipulación de shapes
- [ ] Broadcasting funcionando
- [ ] Variables creadas y modificadas
- [ ] GradientTape calculando derivadas
- [ ] Conversión NumPy ↔ TensorFlow

---

## 🎯 Resultado Esperado

Al completar este ejercicio, deberías ver outputs similares a:

```
TensorFlow version: 2.15.0
Escalar shape: (), valor: 42
Vector shape: (4,)
Matriz shape: (2, 3)
Suma total: 21.0
Gradiente de x² en x=3: 6.0
```
