# 🔷 Semana 20: TensorFlow y Keras

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender la arquitectura de TensorFlow y sus componentes
- ✅ Dominar la API Sequential de Keras para construir modelos
- ✅ Conocer los diferentes tipos de capas y cuándo usarlas
- ✅ Compilar modelos con optimizadores, pérdidas y métricas
- ✅ Entrenar modelos con callbacks y visualizar el progreso
- ✅ Guardar, cargar y exportar modelos entrenados

---

## 📚 Requisitos Previos

- Semana 19: Fundamentos de Redes Neuronales completada
- Entendimiento de backpropagation
- NumPy y Matplotlib dominados
- Python orientado a objetos básico

---

## 🗂️ Estructura de la Semana

```
week-20/
├── README.md
├── rubrica-evaluacion.md
├── 0-assets/
│   ├── 01-tensorflow-arquitectura.svg
│   ├── 02-keras-api-levels.svg
│   ├── 03-sequential-vs-functional.svg
│   ├── 04-training-loop.svg
│   └── 05-callbacks-workflow.svg
├── 1-teoria/
│   ├── 01-introduccion-tensorflow.md
│   ├── 02-keras-api-sequential.md
│   ├── 03-capas-y-activaciones.md
│   └── 04-compilacion-entrenamiento.md
├── 2-practicas/
│   ├── ejercicio-01-tensores-basicos/
│   ├── ejercicio-02-modelo-sequential/
│   └── ejercicio-03-callbacks-checkpoints/
├── 3-proyecto/
│   └── clasificador-mnist/
├── 4-recursos/
│   └── README.md
└── 5-glosario/
    └── README.md
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| #   | Tema                        | Archivo                                                               | Duración |
| --- | --------------------------- | --------------------------------------------------------------------- | -------- |
| 1   | Introducción a TensorFlow   | [01-introduccion-tensorflow.md](1-teoria/01-introduccion-tensorflow.md) | 20 min   |
| 2   | Keras API Sequential        | [02-keras-api-sequential.md](1-teoria/02-keras-api-sequential.md)       | 25 min   |
| 3   | Capas y Activaciones        | [03-capas-y-activaciones.md](1-teoria/03-capas-y-activaciones.md)       | 25 min   |
| 4   | Compilación y Entrenamiento | [04-compilacion-entrenamiento.md](1-teoria/04-compilacion-entrenamiento.md) | 20 min   |

### 💻 Prácticas (2.5 horas)

| #   | Ejercicio               | Carpeta                                                                        | Duración |
| --- | ----------------------- | ------------------------------------------------------------------------------ | -------- |
| 1   | Tensores Básicos        | [ejercicio-01-tensores-basicos/](2-practicas/ejercicio-01-tensores-basicos/)     | 45 min   |
| 2   | Modelo Sequential       | [ejercicio-02-modelo-sequential/](2-practicas/ejercicio-02-modelo-sequential/)   | 50 min   |
| 3   | Callbacks y Checkpoints | [ejercicio-03-callbacks-checkpoints/](2-practicas/ejercicio-03-callbacks-checkpoints/) | 55 min   |

### 📦 Proyecto (2 horas)

| Proyecto           | Descripción                                        | Carpeta                                            |
| ------------------ | -------------------------------------------------- | -------------------------------------------------- |
| Clasificador MNIST | Red neuronal completa para clasificar dígitos escritos a mano | [clasificador-mnist/](3-proyecto/clasificador-mnist/) |

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

   - [ ] Operaciones con tensores de TensorFlow
   - [ ] Modelo Sequential con múltiples capas
   - [ ] Sistema de callbacks y guardado de modelos

2. **Proyecto semanal** (3-proyecto/)
   - [ ] Clasificador MNIST con >97% accuracy
   - [ ] Uso correcto de callbacks
   - [ ] Visualización del entrenamiento
   - [ ] Modelo guardado y exportable

---

## 🔑 Conceptos Clave

- **TensorFlow**: Framework de Google para computación numérica y deep learning
- **Keras**: API de alto nivel para construir y entrenar modelos de forma intuitiva
- **Tensor**: Array multidimensional, estructura de datos fundamental
- **Sequential API**: Forma más simple de construir modelos lineales capa por capa
- **Dense Layer**: Capa completamente conectada (fully connected)
- **Callback**: Función que se ejecuta en puntos específicos del entrenamiento
- **Checkpoint**: Guardado periódico del modelo durante entrenamiento

---

## 🔗 Navegación

| ⬅️ Anterior                              | 🏠 Módulo                     | Siguiente ➡️                      |
| ---------------------------------------- | ----------------------------- | --------------------------------- |
| [Semana 19](../week-19/README.md)        | [Deep Learning](../README.md) | [Semana 21](../week-21/README.md) |

---

## 💡 Tips para esta Semana

> 🎯 **Consejo**: TensorFlow puede parecer complejo al principio, pero Keras lo hace accesible. Enfócate primero en la API Sequential - es todo lo que necesitas para el 90% de los casos.

- **Instala correctamente**: Verifica que TensorFlow funcione con `tf.config.list_physical_devices('GPU')`
- **Usa tf.keras**: Siempre importa desde `tensorflow.keras`, no `keras` standalone
- **Experimenta**: Modifica hiperparámetros y observa cómo cambia el entrenamiento
- **Visualiza**: TensorBoard es tu mejor amigo para debugging

---

## 📚 Recursos Rápidos

- 📖 [TensorFlow Official Tutorials](https://www.tensorflow.org/tutorials)
- 📖 [Keras Documentation](https://keras.io/guides/)
- 🔬 [TensorFlow Playground](https://playground.tensorflow.org/)
- 📺 [MIT Deep Learning Course](https://www.youtube.com/watch?v=5tvmMX8r_OM)

---

_Semana 20 de 36 | Módulo: Deep Learning | Bootcamp IA: Zero to Hero_
