# 🧠 Proyecto: Red Neuronal desde Cero

## 🎯 Objetivo

Implementar una red neuronal multicapa (MLP) completa usando **solo NumPy**, incluyendo forward propagation, backpropagation y entrenamiento.

---

## 📋 Descripción

Construirás una red neuronal que pueda:

1. Resolver el problema XOR (imposible para un perceptrón simple)
2. Clasificar datos sintéticos no linealmente separables
3. Aprender mediante gradient descent con backpropagation

**⚠️ IMPORTANTE**: No uses TensorFlow, PyTorch, Keras ni ningún framework de Deep Learning. Solo NumPy.

---

## 🏗️ Arquitectura a Implementar

```
Input Layer    Hidden Layer    Output Layer
    (2)            (4)             (1)
    
   [x₁]──┐     ┌──[h₁]──┐
         ├─────┤        ├─────[ŷ]
   [x₂]──┘     └──[h₄]──┘
```

---

## 📁 Estructura

```
red-neuronal-desde-cero/
├── README.md           # Este archivo
├── starter/
│   └── main.py         # Código inicial con TODOs
└── solution/
    └── main.py         # Solución completa
```

---

## ✅ Requisitos

### Funcionalidades

- [ ] Inicialización de pesos (He initialization)
- [ ] Forward propagation con cache
- [ ] Función de pérdida (Binary Cross-Entropy)
- [ ] Backward propagation (gradientes)
- [ ] Actualización de pesos (Gradient Descent)
- [ ] Loop de entrenamiento completo
- [ ] Visualización de pérdida y frontera de decisión

### Métricas Mínimas

- Accuracy en XOR: **100%** (es posible)
- Accuracy en datos sintéticos: **> 90%**

---

## 🔧 Funciones a Implementar

```python
# En starter/main.py encontrarás:

def sigmoid(z): ...
def sigmoid_derivative(z): ...
def relu(z): ...
def relu_derivative(z): ...

def initialize_parameters(layer_dims): ...
def forward_propagation(X, parameters): ...
def compute_loss(Y_hat, Y): ...
def backward_propagation(Y_hat, Y, cache, parameters): ...
def update_parameters(parameters, gradients, learning_rate): ...
def train(X, Y, layer_dims, epochs, learning_rate): ...
```

---

## 📊 Dataset

### XOR (Principal)

```python
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])
```

### Moons (Opcional)

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.2)
```

---

## ⏱️ Tiempo Estimado

2 horas

---

## 💡 Tips

1. **Shapes**: Presta atención a las dimensiones de las matrices
2. **Debugging**: Imprime shapes en cada paso
3. **Gradientes**: Usa gradient checking para verificar
4. **Learning Rate**: Empieza con 0.1, ajusta si no converge
5. **Épocas**: XOR converge en ~1000-5000 épocas

---

## 📚 Recursos

- [Neural Networks and Deep Learning (Nielsen)](http://neuralnetworksanddeeplearning.com/)
- [CS231n Backpropagation](https://cs231n.github.io/optimization-2/)
- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

---

## 🎯 Entregables

1. **Código funcional** que entrene la red en XOR
2. **Gráfica de pérdida** vs épocas
3. **Gráfica de frontera de decisión** mostrando clasificación correcta
4. **Accuracy final** impreso en consola

---

## ✅ Criterios de Evaluación

| Criterio | Puntos |
|----------|--------|
| Forward propagation correcto | 20 |
| Backpropagation correcto | 25 |
| Loop de entrenamiento funcional | 15 |
| XOR resuelto (100% accuracy) | 20 |
| Visualizaciones | 10 |
| Código limpio y documentado | 10 |
| **Total** | **100** |

---

## 🚀 Bonus

- Implementar momentum
- Agregar regularización L2
- Probar con dataset make_moons
- Arquitectura configurable (más capas)
