# 🌲 Semana 11: Árboles de Decisión y Random Forest

## 📋 Descripción

Esta semana exploramos los **modelos basados en árboles**, fundamentales en Machine Learning por su interpretabilidad y potencia. Comenzamos con árboles de decisión individuales y avanzamos hacia Random Forest, uno de los algoritmos más utilizados en la industria.

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Entender cómo funcionan los árboles de decisión (CART)
- ✅ Aplicar criterios de división: Gini e Information Gain (Entropy)
- ✅ Controlar overfitting con poda y límites de profundidad
- ✅ Implementar Random Forest para clasificación y regresión
- ✅ Interpretar feature importance en modelos de ensamble
- ✅ Visualizar árboles de decisión con sklearn y graphviz
- ✅ Ajustar hiperparámetros clave (n_estimators, max_depth, etc.)

## 📚 Requisitos Previos

- Semana 09: Fundamentos de ML
- Semana 10: Regresión lineal y logística
- Conocimiento de métricas de evaluación (accuracy, precision, recall)

## 🗂️ Estructura de la Semana

```
week-11/
├── README.md
├── rubrica-evaluacion.md
├── 0-assets/                    # Diagramas SVG
├── 1-teoria/
│   ├── 01-arboles-decision.md   # Fundamentos CART
│   ├── 02-criterios-division.md # Gini vs Entropy
│   ├── 03-random-forest.md      # Bagging y ensambles
│   └── 04-hiperparametros.md    # Tuning y validación
├── 2-practicas/
│   ├── ejercicio-01-arbol-clasificacion/
│   ├── ejercicio-02-arbol-regresion/
│   ├── ejercicio-03-random-forest/
│   └── ejercicio-04-feature-importance/
├── 3-proyecto/                  # Clasificación de especies (Iris/Wine)
├── 4-recursos/
│   ├── ebooks-free/
│   ├── videografia/
│   └── webgrafia/
└── 5-glosario/
```

## 📝 Contenidos

### Teoría (1.5 horas)

| Archivo                  | Tema                                     | Duración |
| ------------------------ | ---------------------------------------- | -------- |
| 01-arboles-decision.md   | Estructura, nodos, hojas, predicción     | 25 min   |
| 02-criterios-division.md | Gini Impurity, Entropy, Information Gain | 25 min   |
| 03-random-forest.md      | Bagging, OOB score, ensambles            | 25 min   |
| 04-hiperparametros.md    | max_depth, n_estimators, GridSearchCV    | 15 min   |

### Prácticas (2.5 horas)

| Ejercicio    | Descripción                           | Duración |
| ------------ | ------------------------------------- | -------- |
| ejercicio-01 | Árbol de clasificación (Iris dataset) | 35 min   |
| ejercicio-02 | Árbol de regresión (precios)          | 35 min   |
| ejercicio-03 | Random Forest clasificación           | 40 min   |
| ejercicio-04 | Feature importance y selección        | 40 min   |

### Proyecto (2 horas)

**Clasificador de Vinos**: Construir un modelo Random Forest para clasificar tipos de vino usando el Wine dataset de sklearn. Objetivo: accuracy ≥ 0.92 en test.

## ⏱️ Distribución del Tiempo (6 horas)

| Actividad | Tiempo    |
| --------- | --------- |
| Teoría    | 1.5 h     |
| Prácticas | 2.5 h     |
| Proyecto  | 2.0 h     |
| **Total** | **6.0 h** |

## 📌 Entregables

1. **Conocimiento 🧠**: Cuestionario sobre árboles y criterios de división
2. **Desempeño 💪**: 4 ejercicios prácticos completados
3. **Producto 📦**: Proyecto de clasificación con accuracy ≥ 0.92

## 🔗 Navegación

| ← Anterior                                   |           Inicio            |                                  Siguiente → |
| :------------------------------------------- | :-------------------------: | -------------------------------------------: |
| [Semana 10: Regresión](../week-10/README.md) | [Bootcamp](../../README.md) | [Semana 12: SVM y KNN](../week-12/README.md) |

---

## 💡 Conceptos Clave

```
Árbol de Decisión
       │
       ▼
┌──────────────┐
│   Nodo Raíz  │ ← Primera división (feature más importante)
└──────┬───────┘
       │
   ┌───┴───┐
   ▼       ▼
┌─────┐ ┌─────┐
│Nodo │ │Nodo │ ← Nodos internos (más divisiones)
└──┬──┘ └──┬──┘
   │       │
┌──┴──┐ ┌──┴──┐
│Hoja │ │Hoja │ ← Predicción final
└─────┘ └─────┘
```

### Fórmulas Principales

**Gini Impurity**:
$$Gini = 1 - \sum_{i=1}^{C} p_i^2$$

**Entropy**:
$$Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

**Information Gain**:
$$IG = Entropy(parent) - \sum \frac{n_{child}}{n_{parent}} \cdot Entropy(child)$$

---

_Semana 11 de 36 | Módulo: Machine Learning_
