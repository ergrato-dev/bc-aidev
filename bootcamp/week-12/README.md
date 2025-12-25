# Semana 12: SVM, KNN y Naive Bayes

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Implementar K-Nearest Neighbors y elegir el k óptimo
- ✅ Comprender Support Vector Machines y kernels
- ✅ Aplicar Naive Bayes para clasificación de texto
- ✅ Comparar y elegir el algoritmo adecuado para cada problema

## 📋 Contenido

### Teoría

| Archivo                                                               | Tema                    | Duración |
| --------------------------------------------------------------------- | ----------------------- | -------- |
| [01-knn.md](1-teoria/01-knn.md)                                       | K-Nearest Neighbors     | 20 min   |
| [02-svm.md](1-teoria/02-svm.md)                                       | Support Vector Machines | 25 min   |
| [03-naive-bayes.md](1-teoria/03-naive-bayes.md)                       | Naive Bayes             | 20 min   |
| [04-comparacion-algoritmos.md](1-teoria/04-comparacion-algoritmos.md) | Comparación y Selección | 15 min   |

### Prácticas

| Ejercicio                                             | Tema                       | Duración |
| ----------------------------------------------------- | -------------------------- | -------- |
| [ejercicio-01](2-practicas/ejercicio-01-knn/)         | KNN con Iris               | 30 min   |
| [ejercicio-02](2-practicas/ejercicio-02-svm/)         | SVM con diferentes kernels | 35 min   |
| [ejercicio-03](2-practicas/ejercicio-03-naive-bayes/) | Naive Bayes para texto     | 35 min   |
| [ejercicio-04](2-practicas/ejercicio-04-comparacion/) | Comparación de algoritmos  | 30 min   |

### Proyecto

| Proyecto                             | Descripción                     | Duración |
| ------------------------------------ | ------------------------------- | -------- |
| [Clasificación de Spam](3-proyecto/) | Comparar KNN, SVM y Naive Bayes | 2 horas  |

## 🗂️ Estructura de la Semana

```
week-12/
├── README.md
├── rubrica-evaluacion.md
├── 0-assets/
│   ├── 01-knn-distancias.svg
│   ├── 02-svm-hiperplano.svg
│   ├── 03-svm-kernels.svg
│   ├── 04-naive-bayes.svg
│   └── 05-comparacion-algoritmos.svg
├── 1-teoria/
│   ├── 01-knn.md
│   ├── 02-svm.md
│   ├── 03-naive-bayes.md
│   └── 04-comparacion-algoritmos.md
├── 2-practicas/
│   ├── ejercicio-01-knn/
│   ├── ejercicio-02-svm/
│   ├── ejercicio-03-naive-bayes/
│   └── ejercicio-04-comparacion/
├── 3-proyecto/
│   ├── README.md
│   ├── starter/
│   └── .solution/
├── 4-recursos/
│   ├── ebooks-free/
│   ├── videografia/
│   └── webgrafia/
└── 5-glosario/
```

## ⏱️ Distribución del Tiempo

| Actividad | Tiempo      |
| --------- | ----------- |
| Teoría    | 1.5 horas   |
| Prácticas | 2.5 horas   |
| Proyecto  | 2 horas     |
| **Total** | **6 horas** |

## 📚 Requisitos Previos

- ✅ Week-09: Fundamentos de ML
- ✅ Week-10: Regresión lineal y logística
- ✅ Week-11: Árboles de decisión y Random Forest
- ✅ Conocimientos de NumPy, Pandas, Matplotlib
- ✅ Familiaridad con scikit-learn

## 🔑 Conceptos Clave

### K-Nearest Neighbors (KNN)

- Algoritmo basado en instancias (lazy learning)
- Distancias: Euclidiana, Manhattan, Minkowski
- Elección del k óptimo
- Curse of dimensionality

### Support Vector Machines (SVM)

- Hiperplano de separación
- Vectores de soporte
- Margen máximo
- Kernels: linear, RBF, polynomial

### Naive Bayes

- Teorema de Bayes
- Asunción de independencia
- Tipos: Gaussian, Multinomial, Bernoulli
- Ideal para clasificación de texto

## 📌 Entregables

1. **Ejercicios completados** (4 ejercicios)
2. **Proyecto de clasificación de spam** con:
   - Implementación de los 3 algoritmos
   - Comparación de métricas
   - Análisis de resultados
   - Accuracy mínimo: 0.90

## 🔗 Navegación

| Anterior                                                    | Siguiente                                      |
| ----------------------------------------------------------- | ---------------------------------------------- |
| [⬅️ Week-11: Árboles y Random Forest](../week-11/README.md) | [Week-13: Clustering ➡️](../week-13/README.md) |

---

## 📖 Recursos Adicionales

- [KNN - sklearn](https://scikit-learn.org/stable/modules/neighbors.html)
- [SVM - sklearn](https://scikit-learn.org/stable/modules/svm.html)
- [Naive Bayes - sklearn](https://scikit-learn.org/stable/modules/naive_bayes.html)
