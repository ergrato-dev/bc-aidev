# 🧠 Semana 19: Fundamentos de Redes Neuronales

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender la inspiración biológica de las redes neuronales
- ✅ Implementar un perceptrón desde cero
- ✅ Entender el algoritmo de backpropagation matemáticamente
- ✅ Conocer las funciones de activación y sus propiedades
- ✅ Construir una red neuronal multicapa con NumPy

---

## 📚 Requisitos Previos

- Módulo 2: Machine Learning completado
- Álgebra lineal (matrices, vectores, multiplicación)
- Cálculo básico (derivadas, regla de la cadena)
- NumPy dominado

---

## 🗂️ Estructura de la Semana

```
week-19/
├── README.md
├── rubrica-evaluacion.md
├── 0-assets/
│   ├── 01-neurona-biologica-vs-artificial.svg
│   ├── 02-perceptron-arquitectura.svg
│   ├── 03-funciones-activacion.svg
│   └── 04-backpropagation-flow.svg
├── 1-teoria/
│   ├── 01-introduccion-redes-neuronales.md
│   ├── 02-perceptron.md
│   ├── 03-funciones-activacion.md
│   └── 04-backpropagation.md
├── 2-practicas/
│   ├── ejercicio-01-perceptron/
│   ├── ejercicio-02-funciones-activacion/
│   └── ejercicio-03-mlp-numpy/
├── 3-proyecto/
│   └── red-neuronal-desde-cero/
├── 4-recursos/
└── 5-glosario/
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| #   | Tema                            | Archivo                                                                             | Duración |
| --- | ------------------------------- | ----------------------------------------------------------------------------------- | -------- |
| 1   | Introducción a Redes Neuronales | [01-introduccion-redes-neuronales.md](1-teoria/01-introduccion-redes-neuronales.md) | 20 min   |
| 2   | El Perceptrón                   | [02-perceptron.md](1-teoria/02-perceptron.md)                                       | 25 min   |
| 3   | Funciones de Activación         | [03-funciones-activacion.md](1-teoria/03-funciones-activacion.md)                   | 20 min   |
| 4   | Backpropagation                 | [04-backpropagation.md](1-teoria/04-backpropagation.md)                             | 25 min   |

### 💻 Prácticas (2.5 horas)

| #   | Ejercicio               | Carpeta                                                                              | Duración |
| --- | ----------------------- | ------------------------------------------------------------------------------------ | -------- |
| 1   | Perceptrón Simple       | [ejercicio-01-perceptron/](2-practicas/ejercicio-01-perceptron/)                     | 45 min   |
| 2   | Funciones de Activación | [ejercicio-02-funciones-activacion/](2-practicas/ejercicio-02-funciones-activacion/) | 45 min   |
| 3   | MLP con NumPy           | [ejercicio-03-mlp-numpy/](2-practicas/ejercicio-03-mlp-numpy/)                       | 60 min   |

### 📦 Proyecto (2 horas)

| Proyecto                | Descripción                                 | Carpeta                                                         |
| ----------------------- | ------------------------------------------- | --------------------------------------------------------------- |
| Red Neuronal desde Cero | Clasificador binario implementado con NumPy | [red-neuronal-desde-cero/](3-proyecto/red-neuronal-desde-cero/) |

---

## ⏱️ Distribución del Tiempo

```
Total: 6 horas

┌─────────────────────────────────────────────────────────┐
│  📖 Teoría      │████████░░░░░░░░░░░░░░░░│  1.5h (25%)  │
│  💻 Prácticas   │████████████████░░░░░░░░│  2.5h (42%)  │
│  📦 Proyecto    │████████████░░░░░░░░░░░░│  2.0h (33%)  │
└─────────────────────────────────────────────────────────┘
```

---

## 📌 Entregables

1. **Ejercicios completados** (2-practicas/)

   - [ ] Perceptrón clasificando AND y OR
   - [ ] Visualización de funciones de activación
   - [ ] MLP forward pass implementado

2. **Proyecto semanal** (3-proyecto/)
   - [ ] Red neuronal multicapa funcional
   - [ ] Backpropagation implementado correctamente
   - [ ] Entrenamiento en dataset sintético
   - [ ] Documentación del proceso

---

## 🔑 Conceptos Clave

- **Neurona artificial**: Unidad básica que recibe inputs, aplica pesos y una función de activación
- **Perceptrón**: Red neuronal de una sola capa para clasificación lineal
- **Función de activación**: Introduce no-linealidad (sigmoid, tanh, ReLU)
- **Forward propagation**: Flujo de datos desde input hasta output
- **Backpropagation**: Algoritmo para calcular gradientes usando regla de la cadena
- **Gradient descent**: Optimización de pesos usando los gradientes calculados

---

## 🔗 Navegación

| ⬅️ Anterior                                              | 🏠 Módulo                     | Siguiente ➡️                      |
| -------------------------------------------------------- | ----------------------------- | --------------------------------- |
| [Semana 18](../../02-machine-learning/week-18/README.md) | [Deep Learning](../README.md) | [Semana 20](../week-20/README.md) |

---

## 💡 Tips para esta Semana

> 🎯 **Consejo**: Entender backpropagation es FUNDAMENTAL. No te saltes las matemáticas - dibuja los grafos de computación y calcula las derivadas a mano al menos una vez.

- **Dibuja**: Visualiza las redes y el flujo de gradientes
- **Deriva**: Calcula las derivadas de las funciones de activación a mano
- **Implementa**: No uses frameworks esta semana, todo con NumPy
- **Verifica**: Usa gradient checking para validar tu implementación

---

## 📚 Recursos Rápidos

- 📖 [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- 📖 [Neural Networks and Deep Learning (Michael Nielsen)](http://neuralnetworksanddeeplearning.com/)
- 🔬 [Backpropagation Calculus - 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8)

---

_Semana 19 de 36 | Módulo: Deep Learning | Bootcamp IA: Zero to Hero_
