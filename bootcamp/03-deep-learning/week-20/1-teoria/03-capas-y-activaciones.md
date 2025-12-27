# 🧩 Capas y Funciones de Activación

## 🎯 Objetivos

- Conocer los tipos de capas más utilizados en Keras
- Entender las funciones de activación y cuándo usar cada una
- Aprender a combinar capas para diferentes tareas
- Dominar capas de regularización (Dropout, BatchNorm)

---

## 📚 Contenido

### 1. Tipos de Capas en Keras

Keras ofrece una amplia variedad de capas predefinidas:

```python
from tensorflow.keras import layers

# Categorías principales
# - Core layers: Dense, Activation, Embedding, etc.
# - Convolutional layers: Conv1D, Conv2D, Conv3D (semanas siguientes)
# - Pooling layers: MaxPooling, AveragePooling
# - Recurrent layers: LSTM, GRU (semana 24)
# - Normalization layers: BatchNormalization, LayerNormalization
# - Regularization layers: Dropout, SpatialDropout
# - Reshaping layers: Flatten, Reshape, Permute
```

---

### 2. Capas Core

#### 2.1 Dense (Fully Connected)

La capa más fundamental - cada neurona conectada a todas las anteriores:

```python
# Dense layer
# Parámetros = (input_features * units) + units(bias)
# Para input=784, units=64: 784*64 + 64 = 50,240 parámetros

dense_layer = layers.Dense(
    units=64,
    activation='relu',
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros'
)

# Ejemplo en modelo
model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

#### 2.2 Activation

Aplica una función de activación independiente:

```python
# Forma 1: Activación incluida en Dense
layers.Dense(64, activation='relu')

# Forma 2: Capa de activación separada (útil para BatchNorm)
model = Sequential([
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),  # Activación después de BatchNorm
    layers.Dense(10, activation='softmax')
])
```

#### 2.3 Flatten

Convierte tensores multidimensionales a 1D:

```python
# Para imágenes: (batch, 28, 28, 1) -> (batch, 784)
model = Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Cálculo: 28 * 28 * 1 = 784
```

#### 2.4 Reshape

Cambia la forma del tensor:

```python
# Reshape para diferentes propósitos
layers.Reshape((7, 7, 64))  # Para decoder en autoencoder
layers.Reshape((-1,))  # Aplanar (equivalente a Flatten)
layers.Reshape((28, 28, 1))  # Reconstruir imagen
```

---

### 3. Funciones de Activación

Las activaciones introducen no-linealidad, permitiendo aprender patrones complejos.

#### 3.1 ReLU y Variantes (Capas Ocultas)

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

# ReLU: Rectified Linear Unit
# f(x) = max(0, x)
relu = tf.nn.relu(x).numpy()

# Leaky ReLU: Permite pequeño gradiente para x < 0
# f(x) = x if x > 0 else alpha * x
leaky_relu = tf.nn.leaky_relu(x, alpha=0.1).numpy()

# ELU: Exponential Linear Unit
# f(x) = x if x > 0 else alpha * (exp(x) - 1)
elu = tf.nn.elu(x).numpy()

# SELU: Scaled ELU (auto-normaliza)
selu = tf.nn.selu(x).numpy()
```

**¿Cuál usar?**

| Activación   | Ventajas                        | Desventajas              |
| ------------ | ------------------------------- | ------------------------ |
| **ReLU**     | Simple, eficiente, estándar     | "Dying ReLU" problem     |
| **Leaky ReLU** | Evita dying ReLU              | Hiperparámetro alpha     |
| **ELU**      | Media cercana a cero            | Más costoso que ReLU     |
| **SELU**     | Auto-normalización              | Requiere inicialización especial |

#### 3.2 Sigmoid y Tanh

```python
# Sigmoid: Salida entre 0 y 1
# f(x) = 1 / (1 + exp(-x))
# Uso: Capa de salida para clasificación binaria
sigmoid = tf.nn.sigmoid(x).numpy()

# Tanh: Salida entre -1 y 1
# f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
# Uso: Menos común en capas ocultas (históricamente en RNNs)
tanh = tf.nn.tanh(x).numpy()
```

#### 3.3 Softmax (Capa de Salida Multiclase)

```python
# Softmax: Convierte logits en probabilidades
# f(x_i) = exp(x_i) / sum(exp(x_j))
# La suma de todas las salidas = 1

logits = tf.constant([[2.0, 1.0, 0.5]])
probabilities = tf.nn.softmax(logits)
print(f"Probabilidades: {probabilities}")
print(f"Suma: {tf.reduce_sum(probabilities)}")  # 1.0

# En modelo de clasificación multiclase
model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')  # 10 clases
])
```

#### 3.4 Tabla de Referencia

| Activación | Rango Output | Uso Típico                    | En Keras            |
| ---------- | ------------ | ----------------------------- | ------------------- |
| ReLU       | [0, ∞)       | Capas ocultas                 | `'relu'`            |
| Leaky ReLU | (-∞, ∞)      | Capas ocultas                 | `layers.LeakyReLU()`|
| Sigmoid    | (0, 1)       | Output binario                | `'sigmoid'`         |
| Tanh       | (-1, 1)      | RNNs, normalizado             | `'tanh'`            |
| Softmax    | (0, 1), Σ=1  | Output multiclase             | `'softmax'`         |
| Linear     | (-∞, ∞)      | Regresión                     | `None` o `'linear'` |

---

### 4. Capas de Regularización

#### 4.1 Dropout

Desactiva aleatoriamente neuronas durante el entrenamiento:

```python
# Dropout: p = probabilidad de desactivar una neurona
model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),  # 30% de neuronas desactivadas
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# IMPORTANTE: Dropout solo actúa durante training
# Durante inference/prediction, todas las neuronas están activas
```

**Tasas de Dropout típicas:**
- 0.2-0.3: Capas iniciales
- 0.4-0.5: Capas intermedias/profundas
- 0.5: Valor clásico (paper original)

#### 4.2 BatchNormalization

Normaliza las activaciones de la capa anterior:

```python
# BatchNorm: normaliza, escala y desplaza
model = Sequential([
    layers.Dense(128, input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),  # Activación DESPUÉS de BatchNorm
    
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    layers.Dense(10, activation='softmax')
])
```

**Beneficios de BatchNorm:**
- Entrenamiento más rápido
- Permite learning rates más altos
- Actúa como regularización
- Reduce la dependencia de inicialización

#### 4.3 Orden Recomendado

```python
# Patrón estándar para cada bloque:
# Dense -> BatchNorm -> Activation -> Dropout

model = Sequential([
    # Bloque 1
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    
    # Bloque 2
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    
    # Salida (sin Dropout ni BatchNorm)
    layers.Dense(10, activation='softmax')
], name='regularized_network')
```

---

### 5. Capas de Normalización de Input

#### 5.1 Normalization Layer

```python
# Normalizar inputs automáticamente
normalizer = layers.Normalization(axis=-1)

# Adaptar a los datos de entrenamiento
normalizer.adapt(X_train)

model = Sequential([
    normalizer,  # Primera capa
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
```

---

### 6. Inicializadores de Pesos

La inicialización correcta es crucial para el entrenamiento:

```python
from tensorflow.keras import initializers

# Glorot/Xavier (default) - Bueno para tanh, sigmoid
layers.Dense(64, kernel_initializer='glorot_uniform')
layers.Dense(64, kernel_initializer='glorot_normal')

# He - Mejor para ReLU
layers.Dense(64, activation='relu', kernel_initializer='he_uniform')
layers.Dense(64, activation='relu', kernel_initializer='he_normal')

# LeCun - Para SELU
layers.Dense(64, activation='selu', kernel_initializer='lecun_normal')

# Custom
custom_init = initializers.RandomNormal(mean=0., stddev=0.05)
layers.Dense(64, kernel_initializer=custom_init)
```

#### Regla General

| Activación | Inicializador Recomendado |
| ---------- | ------------------------- |
| tanh       | Glorot (Xavier)           |
| sigmoid    | Glorot (Xavier)           |
| ReLU       | He                        |
| Leaky ReLU | He                        |
| SELU       | LeCun                     |

---

### 7. Arquitectura Completa de Ejemplo

```python
from tensorflow.keras import Sequential, layers

def create_classifier(
    input_dim: int,
    hidden_layers: list[int],
    num_classes: int,
    dropout_rate: float = 0.3,
    use_batch_norm: bool = True
) -> Sequential:
    """
    Crea un clasificador MLP con regularización.
    
    Args:
        input_dim: Dimensión de entrada
        hidden_layers: Lista con neuronas por capa oculta
        num_classes: Número de clases de salida
        dropout_rate: Tasa de dropout
        use_batch_norm: Usar BatchNormalization
    
    Returns:
        Modelo Sequential compilado
    """
    model_layers = [layers.InputLayer(input_shape=(input_dim,))]
    
    for units in hidden_layers:
        model_layers.append(layers.Dense(units, kernel_initializer='he_normal'))
        
        if use_batch_norm:
            model_layers.append(layers.BatchNormalization())
        
        model_layers.append(layers.Activation('relu'))
        model_layers.append(layers.Dropout(dropout_rate))
    
    # Capa de salida
    if num_classes == 2:
        model_layers.append(layers.Dense(1, activation='sigmoid'))
    else:
        model_layers.append(layers.Dense(num_classes, activation='softmax'))
    
    return Sequential(model_layers)


# Uso
model = create_classifier(
    input_dim=784,
    hidden_layers=[256, 128, 64],
    num_classes=10,
    dropout_rate=0.3
)
model.summary()
```

---

## 💡 Resumen

| Capa             | Uso                                | Parámetros Clave                |
| ---------------- | ---------------------------------- | ------------------------------- |
| **Dense**        | Conexión completa                  | units, activation               |
| **Dropout**      | Regularización                     | rate (0.2-0.5)                  |
| **BatchNorm**    | Normalización, estabilidad         | momentum, epsilon               |
| **Flatten**      | Multidim → 1D                      | input_shape                     |
| **Activation**   | Aplicar activación separada        | activation function             |

---

## ✅ Verificación de Aprendizaje

- [ ] Conozco los tipos de capas principales en Keras
- [ ] Entiendo cuándo usar cada función de activación
- [ ] Sé aplicar Dropout y BatchNormalization correctamente
- [ ] Comprendo los inicializadores y cuándo usar cada uno
- [ ] Puedo diseñar arquitecturas con regularización apropiada

---

_Siguiente: [04-compilacion-entrenamiento.md](04-compilacion-entrenamiento.md)_
