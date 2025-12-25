# 🔧 Semana 14: Feature Engineering y Selección de Características

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender la importancia del Feature Engineering en ML
- ✅ Aplicar técnicas de transformación de variables numéricas
- ✅ Codificar variables categóricas correctamente
- ✅ Crear nuevas features a partir de datos existentes
- ✅ Manejar datos faltantes con estrategias apropiadas
- ✅ Seleccionar características relevantes con diferentes métodos
- ✅ Implementar pipelines de preprocesamiento en Scikit-learn

---

## 📚 Requisitos Previos

- Semana 13: Clustering completada
- Conocimiento de NumPy y Pandas
- Familiaridad con Scikit-learn
- Conceptos básicos de estadística

---

## 🗂️ Estructura de la Semana

```
week-14/
├── README.md                        # Este archivo
├── rubrica-evaluacion.md            # Criterios de evaluación
├── 0-assets/                        # Diagramas y recursos visuales
│   ├── 01-feature-engineering-overview.svg
│   ├── 02-transformaciones-numericas.svg
│   ├── 03-codificacion-categoricas.svg
│   ├── 04-feature-selection-methods.svg
│   └── 05-pipeline-sklearn.svg
├── 1-teoria/                        # Material teórico
│   ├── 01-introduccion-feature-engineering.md
│   ├── 02-transformaciones-numericas.md
│   ├── 03-codificacion-categoricas.md
│   ├── 04-creacion-features.md
│   └── 05-seleccion-caracteristicas.md
├── 2-practicas/                     # Ejercicios guiados
│   ├── ejercicio-01-transformaciones/
│   ├── ejercicio-02-categoricas/
│   ├── ejercicio-03-missing-data/
│   └── ejercicio-04-feature-selection/
├── 3-proyecto/                      # Proyecto semanal
│   └── pipeline-preprocesamiento/
├── 4-recursos/                      # Material adicional
│   ├── README.md
│   ├── ebooks-free/
│   ├── videografia/
│   └── webgrafia/
└── 5-glosario/                      # Términos clave
    └── README.md
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| #   | Tema                               | Archivo                                                                                   | Duración |
| --- | ---------------------------------- | ----------------------------------------------------------------------------------------- | -------- |
| 1   | Introducción a Feature Engineering | [01-introduccion-feature-engineering.md](1-teoria/01-introduccion-feature-engineering.md) | 15 min   |
| 2   | Transformaciones Numéricas         | [02-transformaciones-numericas.md](1-teoria/02-transformaciones-numericas.md)             | 20 min   |
| 3   | Codificación de Categóricas        | [03-codificacion-categoricas.md](1-teoria/03-codificacion-categoricas.md)                 | 20 min   |
| 4   | Creación de Features               | [04-creacion-features.md](1-teoria/04-creacion-features.md)                               | 20 min   |
| 5   | Selección de Características       | [05-seleccion-caracteristicas.md](1-teoria/05-seleccion-caracteristicas.md)               | 15 min   |

### 💻 Prácticas (2.5 horas)

| #   | Ejercicio                    | Carpeta                                                                        | Duración |
| --- | ---------------------------- | ------------------------------------------------------------------------------ | -------- |
| 1   | Transformaciones Numéricas   | [ejercicio-01-transformaciones/](2-practicas/ejercicio-01-transformaciones/)   | 40 min   |
| 2   | Codificación de Categóricas  | [ejercicio-02-categoricas/](2-practicas/ejercicio-02-categoricas/)             | 35 min   |
| 3   | Manejo de Datos Faltantes    | [ejercicio-03-missing-data/](2-practicas/ejercicio-03-missing-data/)           | 35 min   |
| 4   | Selección de Características | [ejercicio-04-feature-selection/](2-practicas/ejercicio-04-feature-selection/) | 40 min   |

### 📦 Proyecto (2 horas)

| Proyecto                     | Descripción                                                           | Carpeta                                                             |
| ---------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Pipeline de Preprocesamiento | Pipeline completo end-to-end con sklearn Pipeline y ColumnTransformer | [pipeline-preprocesamiento/](3-proyecto/pipeline-preprocesamiento/) |

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

Al finalizar la semana debes entregar:

1. **Ejercicios completados** (2-practicas/)

   - [ ] ejercicio-01: Transformaciones numéricas aplicadas
   - [ ] ejercicio-02: Codificación de categóricas implementada
   - [ ] ejercicio-03: Estrategias de missing data aplicadas
   - [ ] ejercicio-04: Feature selection con múltiples métodos

2. **Proyecto semanal** (3-proyecto/)

   - [ ] Pipeline de preprocesamiento completo
   - [ ] ColumnTransformer configurado correctamente
   - [ ] Modelo entrenado con features transformadas
   - [ ] Comparación de rendimiento antes/después

3. **Autoevaluación**
   - [ ] Completar checklist de verificación
   - [ ] Responder cuestionario de conocimientos

---

## 🎯 Competencias a Desarrollar

### Técnicas

- Transformaciones: StandardScaler, MinMaxScaler, Log, Box-Cox
- Encoding: OneHot, Label, Target, Ordinal
- Feature Selection: Filter, Wrapper, Embedded methods
- Pipelines: sklearn Pipeline y ColumnTransformer

### Transversales

- Pensamiento crítico para elegir transformaciones
- Análisis de datos para identificar patterns
- Documentación de decisiones de preprocesamiento

---

## 🔗 Navegación

| ⬅️ Anterior                       | 🏠 Inicio                    | Siguiente ➡️                      |
| --------------------------------- | ---------------------------- | --------------------------------- |
| [Semana 13](../week-13/README.md) | [Módulo ML](../../README.md) | [Semana 15](../week-15/README.md) |

---

## 💡 Tips para esta Semana

> 🎯 **Consejo**: "Garbage in, garbage out" - La calidad de tus features determina el límite superior de tu modelo. Un buen feature engineering puede mejorar más el rendimiento que cambiar de algoritmo.

- **Explora antes de transformar**: Conoce tus datos antes de aplicar transformaciones
- **Evita data leakage**: Las transformaciones deben fitear solo en train
- **Documenta decisiones**: Explica por qué elegiste cada transformación
- **Itera**: El feature engineering es un proceso iterativo

---

## 📚 Recursos Rápidos

- 📖 [Sklearn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- 📖 [Sklearn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- 📖 [Sklearn Pipeline](https://scikit-learn.org/stable/modules/compose.html)
- 🎥 [Feature Engineering Course - Kaggle](https://www.kaggle.com/learn/feature-engineering)

---

_Semana 14 de 36 | Módulo: Machine Learning | Bootcamp IA: Zero to Hero_
