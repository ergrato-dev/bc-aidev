# 📋 Rúbrica de Evaluación - Semana 19

## 🧠 Fundamentos de Redes Neuronales

---

## 📊 Distribución de Puntos

| Componente               | Porcentaje | Puntos  |
| ------------------------ | ---------- | ------- |
| 🧠 Conocimiento          | 30%        | 30      |
| 💪 Desempeño (Prácticas) | 35%        | 35      |
| 📦 Producto (Proyecto)   | 35%        | 35      |
| **Total**                | **100%**   | **100** |

---

## 🧠 Conocimiento (30 puntos)

### Conceptos Teóricos

| Criterio                                            | Puntos |
| --------------------------------------------------- | ------ |
| Explica la analogía neurona biológica vs artificial | 5      |
| Describe el algoritmo del perceptrón                | 5      |
| Conoce propiedades de funciones de activación       | 8      |
| Entiende backpropagation y regla de la cadena       | 12     |

### Niveles de Desempeño - Conocimiento

| Nivel        | Rango | Descripción                              |
| ------------ | ----- | ---------------------------------------- |
| Insuficiente | 0-17  | No comprende los conceptos fundamentales |
| Suficiente   | 18-21 | Comprensión básica de redes neuronales   |
| Bueno        | 22-26 | Entiende matemáticas de backpropagation  |
| Excelente    | 27-30 | Domina teoría y puede derivar ecuaciones |

---

## 💪 Desempeño - Prácticas (35 puntos)

### Ejercicio 1: Perceptrón (10 puntos)

| Criterio                              | Puntos |
| ------------------------------------- | ------ |
| Implementa forward pass correctamente | 3      |
| Implementa regla de aprendizaje       | 4      |
| Clasifica correctamente AND y OR      | 3      |

### Ejercicio 2: Funciones de Activación (10 puntos)

| Criterio                              | Puntos |
| ------------------------------------- | ------ |
| Implementa sigmoid y su derivada      | 3      |
| Implementa tanh y su derivada         | 3      |
| Implementa ReLU y variantes           | 2      |
| Visualiza correctamente las funciones | 2      |

### Ejercicio 3: MLP con NumPy (15 puntos)

| Criterio                                    | Puntos |
| ------------------------------------------- | ------ |
| Forward propagation correcto                | 5      |
| Arquitectura configurable (capas/neuronas)  | 4      |
| Código vectorizado (sin loops innecesarios) | 4      |
| Documentación clara                         | 2      |

---

## 📦 Producto - Proyecto (35 puntos)

### Red Neuronal desde Cero

| Criterio                           | Puntos |
| ---------------------------------- | ------ |
| **Arquitectura** (10 pts)          |        |
| - Inicialización de pesos correcta | 3      |
| - Forward pass multicapa           | 4      |
| - Estructura modular y extensible  | 3      |
| **Backpropagation** (15 pts)       |        |
| - Cálculo de gradientes correcto   | 8      |
| - Actualización de pesos y biases  | 4      |
| - Gradient checking implementado   | 3      |
| **Entrenamiento** (10 pts)         |        |
| - Loop de entrenamiento funcional  | 3      |
| - Tracking de loss por época       | 2      |
| - Convergencia demostrada          | 3      |
| - Visualización del proceso        | 2      |

---

## 🎯 Criterios de Aprobación

- ✅ Mínimo **70%** en cada componente
- ✅ Proyecto funcional con backpropagation correcto
- ✅ Implementación sin usar frameworks de DL
- ✅ Gradient checking con error < 1e-5

---

## ⭐ Puntos Extra (hasta +10)

| Criterio                                    | Puntos |
| ------------------------------------------- | ------ |
| Implementa momentum o Adam optimizer        | +3     |
| Añade regularización L2                     | +2     |
| Implementa mini-batch gradient descent      | +3     |
| Visualización interactiva del entrenamiento | +2     |

---

## ⚠️ Penalizaciones

| Criterio                                   | Puntos |
| ------------------------------------------ | ------ |
| Usar TensorFlow/PyTorch/Keras              | -20    |
| Código copiado sin entender                | -15    |
| No implementar backpropagation manualmente | -20    |
| Gradientes incorrectos (error > 1e-3)      | -10    |

---

## 📝 Notas

- Esta semana es FUNDAMENTAL para entender deep learning
- La implementación manual construye intuición profunda
- Frameworks vendrán en semanas siguientes
- Los gradientes deben verificarse numéricamente
