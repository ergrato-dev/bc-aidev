# 🔥 Semana 21: PyTorch Fundamentals

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender la filosofía y arquitectura de PyTorch
- ✅ Manipular tensores de PyTorch con fluidez
- ✅ Utilizar autograd para diferenciación automática
- ✅ Construir redes neuronales con `nn.Module`
- ✅ Implementar el training loop completo manualmente
- ✅ Comparar PyTorch vs TensorFlow y elegir según el caso

---

## 📚 Requisitos Previos

- Semana 19: Fundamentos de Redes Neuronales
- Semana 20: TensorFlow y Keras (para comparación)
- NumPy dominado (PyTorch tiene sintaxis similar)
- Python orientado a objetos sólido

---

## 🗂️ Estructura de la Semana

```
week-21/
├── README.md
├── rubrica-evaluacion.md
├── 0-assets/
│   ├── 01-pytorch-arquitectura.svg
│   ├── 02-tensor-operations.svg
│   ├── 03-autograd-computational-graph.svg
│   ├── 04-nn-module-anatomy.svg
│   └── 05-training-loop-pytorch.svg
├── 1-teoria/
│   ├── 01-introduccion-pytorch.md
│   ├── 02-tensores-pytorch.md
│   ├── 03-autograd-diferenciacion.md
│   └── 04-nn-module-training.md
├── 2-practicas/
│   ├── ejercicio-01-tensores-pytorch/
│   ├── ejercicio-02-autograd-gradientes/
│   └── ejercicio-03-red-neuronal-manual/
├── 3-proyecto/
│   └── clasificador-fashion-mnist/
├── 4-recursos/
│   └── README.md
└── 5-glosario/
    └── README.md
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| #   | Tema                           | Archivo                                                                   | Duración |
| --- | ------------------------------ | ------------------------------------------------------------------------- | -------- |
| 1   | Introducción a PyTorch         | [01-introduccion-pytorch.md](1-teoria/01-introduccion-pytorch.md)         | 20 min   |
| 2   | Tensores en PyTorch            | [02-tensores-pytorch.md](1-teoria/02-tensores-pytorch.md)                 | 25 min   |
| 3   | Autograd y Diferenciación      | [03-autograd-diferenciacion.md](1-teoria/03-autograd-diferenciacion.md)   | 25 min   |
| 4   | nn.Module y Training Loop      | [04-nn-module-training.md](1-teoria/04-nn-module-training.md)             | 20 min   |

### 💻 Prácticas (2.5 horas)

| #   | Ejercicio                | Carpeta                                                                          | Duración |
| --- | ------------------------ | -------------------------------------------------------------------------------- | -------- |
| 1   | Tensores PyTorch         | [ejercicio-01-tensores-pytorch/](2-practicas/ejercicio-01-tensores-pytorch/)     | 45 min   |
| 2   | Autograd y Gradientes    | [ejercicio-02-autograd-gradientes/](2-practicas/ejercicio-02-autograd-gradientes/) | 50 min   |
| 3   | Red Neuronal Manual      | [ejercicio-03-red-neuronal-manual/](2-practicas/ejercicio-03-red-neuronal-manual/) | 55 min   |

### 📦 Proyecto (2 horas)

| Proyecto                    | Descripción                                                  | Carpeta                                                        |
| --------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------- |
| Clasificador Fashion-MNIST  | Red neuronal completa en PyTorch para clasificar prendas     | [clasificador-fashion-mnist/](3-proyecto/clasificador-fashion-mnist/) |

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

   - [ ] Operaciones con tensores de PyTorch
   - [ ] Cálculo de gradientes con autograd
   - [ ] Red neuronal con nn.Module desde cero

2. **Proyecto semanal** (3-proyecto/)
   - [ ] Clasificador Fashion-MNIST con >88% accuracy
   - [ ] Training loop implementado manualmente
   - [ ] Visualización de pérdida y accuracy
   - [ ] Modelo guardado con torch.save()

---

## 🔑 Conceptos Clave

- **PyTorch**: Framework de Facebook/Meta para deep learning, preferido en investigación
- **Tensor**: Array multidimensional similar a NumPy pero con soporte GPU
- **Autograd**: Sistema de diferenciación automática de PyTorch
- **Computational Graph**: Grafo dinámico que registra operaciones para backprop
- **nn.Module**: Clase base para construir redes neuronales
- **requires_grad**: Flag que indica si un tensor necesita gradientes
- **backward()**: Método que calcula gradientes automáticamente
- **optimizer.step()**: Actualiza parámetros usando gradientes calculados

---

## 🔗 Navegación

| ⬅️ Anterior                       | 🏠 Módulo                     | Siguiente ➡️                      |
| --------------------------------- | ----------------------------- | --------------------------------- |
| [Semana 20](../week-20/README.md) | [Deep Learning](../README.md) | [Semana 22](../week-22/README.md) |

---

## 💡 Tips para esta Semana

> 🎯 **Consejo**: PyTorch es más "pythónico" que TensorFlow. Si te sientes cómodo con Python y NumPy, PyTorch te resultará muy natural. El training loop manual te da control total.

- **Define by Run**: PyTorch construye el grafo dinámicamente, ideal para debugging
- **NumPy-like**: Si sabes NumPy, ya sabes 70% de PyTorch
- **GPU fácil**: `.to(device)` mueve tensores entre CPU/GPU sin complicaciones
- **Debugging simple**: Puedes usar print() y pdb normalmente

---

## 📚 Recursos Rápidos

- 📖 [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- 📖 [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- 🎥 [Deep Learning with PyTorch - Full Course](https://www.youtube.com/watch?v=c36lUUr864M)
- 🔬 [PyTorch Examples Repository](https://github.com/pytorch/examples)

---

_Semana 21 de 36 | Módulo: Deep Learning | Bootcamp IA: Zero to Hero_
