# 🖼️ Semana 22: Redes Neuronales Convolucionales I

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender la operación de convolución y su rol en visión por computadora
- ✅ Entender el funcionamiento de pooling y su importancia
- ✅ Conocer arquitecturas clásicas: LeNet-5 y VGG
- ✅ Implementar CNNs desde cero en PyTorch
- ✅ Aplicar CNNs para clasificación de imágenes

---

## 📚 Requisitos Previos

- ✅ Semana 19: Redes Neuronales (perceptrón, backpropagation)
- ✅ Semana 20: TensorFlow/Keras fundamentals
- ✅ Semana 21: PyTorch (tensores, autograd, nn.Module)
- 📐 Álgebra lineal (matrices, operaciones)

---

## 🗂️ Estructura de la Semana

```
week-22/
├── README.md                    # Este archivo
├── rubrica-evaluacion.md        # Criterios de evaluación
├── 0-assets/                    # Diagramas SVG
│   ├── 01-convolucion-operacion.svg
│   ├── 02-pooling-tipos.svg
│   ├── 03-lenet5-arquitectura.svg
│   ├── 04-vgg16-arquitectura.svg
│   └── 05-feature-maps.svg
├── 1-teoria/                    # Material teórico
│   ├── 01-introduccion-cnns.md
│   ├── 02-operacion-convolucion.md
│   ├── 03-pooling-padding-stride.md
│   └── 04-arquitecturas-clasicas.md
├── 2-practicas/                 # Ejercicios guiados
│   ├── ejercicio-01-convolucion-manual/
│   ├── ejercicio-02-cnn-pytorch/
│   └── ejercicio-03-lenet5-mnist/
├── 3-proyecto/                  # Proyecto semanal
│   └── clasificador-cifar10/
├── 4-recursos/                  # Material adicional
└── 5-glosario/                  # Términos clave
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| #   | Tema                        | Archivo                                                               | Duración |
| --- | --------------------------- | --------------------------------------------------------------------- | -------- |
| 1   | Introducción a CNNs         | [01-introduccion-cnns.md](1-teoria/01-introduccion-cnns.md)           | 20 min   |
| 2   | Operación de Convolución    | [02-operacion-convolucion.md](1-teoria/02-operacion-convolucion.md)   | 25 min   |
| 3   | Pooling, Padding y Stride   | [03-pooling-padding-stride.md](1-teoria/03-pooling-padding-stride.md) | 25 min   |
| 4   | Arquitecturas Clásicas      | [04-arquitecturas-clasicas.md](1-teoria/04-arquitecturas-clasicas.md) | 20 min   |

### 💻 Prácticas (2.5 horas)

| #   | Ejercicio                  | Carpeta                                                                | Duración |
| --- | -------------------------- | ---------------------------------------------------------------------- | -------- |
| 1   | Convolución Manual         | [ejercicio-01-convolucion-manual/](2-practicas/ejercicio-01-convolucion-manual/) | 40 min   |
| 2   | CNN en PyTorch             | [ejercicio-02-cnn-pytorch/](2-practicas/ejercicio-02-cnn-pytorch/)     | 50 min   |
| 3   | LeNet-5 con MNIST          | [ejercicio-03-lenet5-mnist/](2-practicas/ejercicio-03-lenet5-mnist/)   | 60 min   |

### 📦 Proyecto (2 horas)

| Proyecto                | Descripción                                    | Carpeta                                                   |
| ----------------------- | ---------------------------------------------- | --------------------------------------------------------- |
| Clasificador CIFAR-10   | CNN para clasificar imágenes a color (10 clases) | [clasificador-cifar10/](3-proyecto/clasificador-cifar10/) |

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

## 🧠 Conceptos Clave

### Convolución

La convolución es la operación fundamental de las CNNs:
- **Kernel/Filtro**: Matriz pequeña que se desliza sobre la imagen
- **Feature Map**: Resultado de aplicar el kernel
- **Parámetros compartidos**: Reduce dramáticamente el número de parámetros

### Pooling

Reduce dimensionalidad manteniendo información importante:
- **Max Pooling**: Toma el valor máximo de cada región
- **Average Pooling**: Promedia los valores de cada región
- **Global Pooling**: Reduce cada canal a un solo valor

### Arquitecturas

- **LeNet-5 (1998)**: Primera CNN exitosa, diseñada para dígitos
- **VGG (2014)**: Arquitectura profunda con filtros 3×3
- **Principio**: Aumentar canales mientras se reduce espacialidad

---

## 📌 Entregables

Al finalizar la semana debes entregar:

1. **Ejercicios completados** (2-practicas/)
   - [ ] Convolución manual implementada
   - [ ] CNN básica funcionando
   - [ ] LeNet-5 entrenada en MNIST

2. **Proyecto semanal** (3-proyecto/)
   - [ ] Clasificador CIFAR-10 con accuracy ≥ 70%
   - [ ] Arquitectura documentada
   - [ ] Visualización de feature maps

3. **Autoevaluación**
   - [ ] Explicar la operación de convolución
   - [ ] Describir la diferencia entre tipos de pooling

---

## 🔗 Navegación

| ⬅️ Anterior                     | 🏠 Módulo                             | Siguiente ➡️                    |
| ------------------------------- | ------------------------------------- | ------------------------------- |
| [Semana 21](../week-21/README.md) | [Deep Learning](../README.md) | [Semana 23](../week-23/README.md) |

---

## 💡 Tips para esta Semana

> 🎯 **Visualiza**: Las CNNs son intuitivas cuando visualizas qué detecta cada filtro. Usa herramientas de visualización de feature maps.

- **Entiende la convolución**: Es la base de todo, asegúrate de entender la operación matemática
- **Dibuja las dimensiones**: Calcula manualmente el tamaño de salida de cada capa
- **Experimenta con filtros**: Los filtros de detección de bordes son un buen inicio
- **Revisa arquitecturas**: Estudia LeNet y VGG antes de diseñar tus propias redes

---

## 📚 Fórmulas Importantes

### Tamaño de Salida de Convolución

$$O = \frac{W - K + 2P}{S} + 1$$

Donde:
- $O$: Tamaño de salida
- $W$: Tamaño de entrada
- $K$: Tamaño del kernel
- $P$: Padding
- $S$: Stride

### Número de Parámetros por Capa Convolucional

$$\text{Params} = (K \times K \times C_{in} + 1) \times C_{out}$$

Donde:
- $K$: Tamaño del kernel
- $C_{in}$: Canales de entrada
- $C_{out}$: Canales de salida (número de filtros)
- $+1$: Bias por filtro

---

_Semana 22 de 36 | Módulo: Deep Learning | Bootcamp IA: Zero to Hero_
