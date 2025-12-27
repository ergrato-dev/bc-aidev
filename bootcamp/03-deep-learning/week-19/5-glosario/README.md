# 📖 Glosario - Semana 19

Términos clave de Fundamentos de Redes Neuronales.

---

## A

### Activation Function (Función de Activación)

Función no lineal aplicada a la salida de una neurona. Introduce no-linealidad en la red, permitiendo aprender patrones complejos.

```python
# Ejemplo: ReLU
def relu(z):
    return np.maximum(0, z)
```

---

## B

### Backpropagation (Retropropagación)

Algoritmo para calcular gradientes de la función de pérdida respecto a los pesos. Usa la regla de la cadena para propagar el error desde la salida hacia las capas anteriores.

### Bias (Sesgo)

Parámetro adicional en cada neurona que permite desplazar la función de activación. Análogo al término independiente en una ecuación lineal.

$$z = w \cdot x + b$$

---

## C

### Chain Rule (Regla de la Cadena)

Regla de cálculo para derivar funciones compuestas. Fundamento matemático de backpropagation.

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### Computation Graph (Grafo Computacional)

Representación visual de las operaciones matemáticas como un grafo dirigido. Facilita el cálculo de gradientes.

---

## D

### Deep Learning (Aprendizaje Profundo)

Subcampo del Machine Learning que usa redes neuronales con múltiples capas (profundas) para aprender representaciones jerárquicas.

### Dense Layer (Capa Densa)

Capa donde cada neurona está conectada a todas las neuronas de la capa anterior. También llamada "fully connected".

### Derivative (Derivada)

Medida de la tasa de cambio de una función. En redes neuronales, usada para determinar cómo ajustar pesos.

---

## E

### Epoch (Época)

Una pasada completa por todo el dataset de entrenamiento durante el proceso de aprendizaje.

### Exploding Gradient

Problema donde los gradientes crecen exponencialmente durante backpropagation, causando inestabilidad numérica.

---

## F

### Forward Propagation (Propagación Hacia Adelante)

Proceso de calcular la salida de la red pasando la entrada a través de todas las capas.

---

## G

### Gradient (Gradiente)

Vector de derivadas parciales. Indica la dirección de máximo crecimiento de una función.

### Gradient Descent (Descenso de Gradiente)

Algoritmo de optimización que actualiza parámetros en dirección opuesta al gradiente para minimizar la función de pérdida.

$$w_{nuevo} = w_{viejo} - \eta \cdot \nabla L$$

---

## H

### He Initialization

Técnica de inicialización de pesos diseñada para redes con ReLU.

$$W \sim \mathcal{N}(0, \sqrt{2/n_{in}})$$

### Hidden Layer (Capa Oculta)

Capas entre la entrada y la salida. Extraen representaciones intermedias de los datos.

---

## L

### Learning Rate (Tasa de Aprendizaje)

Hiperparámetro que controla el tamaño de los pasos en gradient descent. Notación: η (eta) o α (alpha).

### Leaky ReLU

Variante de ReLU que permite un pequeño gradiente para valores negativos, evitando "neuronas muertas".

$$f(x) = \max(\alpha x, x), \quad \alpha \approx 0.01$$

### Loss Function (Función de Pérdida)

Mide qué tan lejos están las predicciones de los valores reales. El objetivo del entrenamiento es minimizarla.

---

## M

### MLP (Multi-Layer Perceptron)

Red neuronal feedforward con una o más capas ocultas. Arquitectura más básica de Deep Learning.

---

## N

### Neural Network (Red Neuronal)

Modelo computacional inspirado en el cerebro, compuesto por capas de neuronas artificiales interconectadas.

### Neuron (Neurona)

Unidad básica de una red neuronal. Recibe entradas, aplica pesos, suma, añade bias y pasa por activación.

---

## P

### Perceptron (Perceptrón)

Red neuronal más simple, con una sola capa. Solo puede clasificar datos linealmente separables.

---

## R

### ReLU (Rectified Linear Unit)

Función de activación más usada en capas ocultas: $f(x) = \max(0, x)$

---

## S

### Sigmoid

Función de activación que mapea valores al rango (0, 1). Usada en salidas binarias.

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### Softmax

Función que convierte un vector de valores en probabilidades que suman 1. Usada en clasificación multiclase.

---

## T

### Tanh (Tangente Hiperbólica)

Función de activación que mapea valores al rango (-1, 1). Centrada en cero.

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

---

## V

### Vanishing Gradient

Problema donde los gradientes se vuelven muy pequeños en capas profundas, impidiendo el aprendizaje efectivo.

---

## W

### Weight (Peso)

Parámetro que determina la importancia de una conexión entre neuronas. Se ajusta durante el entrenamiento.

---

## X

### Xavier Initialization

Técnica de inicialización de pesos para mantener varianza estable entre capas.

$$W \sim \mathcal{N}(0, \sqrt{1/n_{in}})$$

### XOR Problem

Problema clásico que un perceptrón simple no puede resolver, pero un MLP con capa oculta sí.

---

## Símbolos Comunes

| Símbolo  | Significado                                     |
| -------- | ----------------------------------------------- |
| $W$      | Matriz de pesos                                 |
| $b$      | Vector de bias                                  |
| $z$      | Pre-activación (antes de función de activación) |
| $a$      | Activación (después de función de activación)   |
| $L$      | Pérdida (loss)                                  |
| $\eta$   | Learning rate                                   |
| $\nabla$ | Gradiente                                       |
| $\sigma$ | Sigmoid                                         |
