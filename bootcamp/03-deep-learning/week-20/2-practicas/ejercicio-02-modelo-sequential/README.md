# 🧱 Ejercicio 2: Modelo Sequential en Keras

## 🎯 Objetivo

Construir y configurar modelos de redes neuronales usando la API Sequential de Keras, explorando diferentes arquitecturas y configuraciones.

## ⏱️ Duración

50 minutos

## 📋 Instrucciones

Sigue cada paso en orden, descomentando el código en `starter/main.py` según avances.

---

## Paso 1: Importaciones y Configuración

Configuramos el entorno y las importaciones necesarias:

```python
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.utils import plot_model
import numpy as np

tf.random.set_seed(42)
```

**Abre `starter/main.py`** y verifica que las importaciones funcionan.

---

## Paso 2: Crear Modelo con Lista de Capas

La forma más común de crear un modelo Sequential:

```python
model = Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()
```

Observa:
- El número de parámetros de cada capa
- Cómo se propagan los shapes

---

## Paso 3: Crear Modelo con .add()

Forma alternativa, útil para construcción dinámica:

```python
model = Sequential()
model.add(layers.InputLayer(input_shape=(784,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

---

## Paso 4: Nombrar Modelos y Capas

Los nombres descriptivos facilitan el debugging:

```python
model = Sequential([
    layers.Dense(64, activation='relu', name='hidden_1', input_shape=(784,)),
    layers.Dense(32, activation='relu', name='hidden_2'),
    layers.Dense(10, activation='softmax', name='output')
], name='my_classifier')

# Acceder a capas por nombre
layer = model.get_layer('hidden_1')
```

---

## Paso 5: Inspección Detallada

Analiza la estructura del modelo:

```python
# Iterar sobre capas
for i, layer in enumerate(model.layers):
    print(f"Capa {i}: {layer.name}")
    print(f"  Output shape: {layer.output_shape}")
    print(f"  Parámetros: {layer.count_params()}")
    
# Obtener pesos
weights, biases = model.layers[0].get_weights()
```

---

## Paso 6: Modelo para Clasificación Binaria

Arquitectura para problemas de 2 clases:

```python
model_binary = Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # UNA neurona, sigmoid
])
```

---

## Paso 7: Modelo para Regresión

Sin activación en la capa de salida:

```python
model_regression = Sequential([
    layers.Dense(64, activation='relu', input_shape=(13,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Sin activación para valores continuos
])
```

---

## Paso 8: Modelo con Regularización

Añadiendo Dropout y BatchNormalization:

```python
model_regularized = Sequential([
    layers.Dense(128, input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    
    layers.Dense(10, activation='softmax')
])
```

---

## Paso 9: Modelo con Diferentes Inicializadores

Configura la inicialización de pesos:

```python
model_custom_init = Sequential([
    layers.Dense(64, activation='relu', 
                 kernel_initializer='he_normal',
                 bias_initializer='zeros',
                 input_shape=(784,)),
    layers.Dense(10, activation='softmax',
                 kernel_initializer='glorot_uniform')
])
```

---

## Paso 10: Probar con Datos de Prueba

Verifica que el modelo funciona con datos:

```python
# Datos sintéticos
X_dummy = np.random.randn(5, 784).astype('float32')

# Forward pass (sin entrenar)
predictions = model.predict(X_dummy)
print(f"Shape de predicciones: {predictions.shape}")
print(f"Suma de probabilidades: {predictions.sum(axis=1)}")  # Debe ser ~1
```

---

## ✅ Checklist de Completado

- [ ] Modelo creado con lista de capas
- [ ] Modelo creado con `.add()`
- [ ] Modelos y capas nombrados
- [ ] Inspección de arquitectura realizada
- [ ] Modelo para clasificación binaria
- [ ] Modelo para regresión
- [ ] Modelo con regularización (Dropout, BatchNorm)
- [ ] Inicializadores personalizados aplicados
- [ ] Forward pass con datos de prueba

---

## 🎯 Resultado Esperado

Al completar, deberías poder crear modelos como:

```
Model: "my_classifier"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 hidden_1 (Dense)            (None, 64)                50240     
 hidden_2 (Dense)            (None, 32)                2080      
 output (Dense)              (None, 10)                330       
=================================================================
Total params: 52,650
```
