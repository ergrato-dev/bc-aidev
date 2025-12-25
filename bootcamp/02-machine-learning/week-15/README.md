# 📊 Semana 15: Validación Cruzada y Métricas de Evaluación

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Aplicar diferentes estrategias de validación cruzada
- ✅ Seleccionar métricas apropiadas para clasificación y regresión
- ✅ Interpretar matrices de confusión y curvas ROC/PR
- ✅ Detectar y prevenir overfitting/underfitting
- ✅ Usar GridSearchCV y RandomizedSearchCV para optimización

---

## 📚 Requisitos Previos

- Semana 14: Feature Engineering completada
- Conocimiento de algoritmos de ML básicos
- Familiaridad con sklearn Pipeline

---

## 🗂️ Estructura de la Semana

```
week-15/
├── README.md                    # Este archivo
├── rubrica-evaluacion.md        # Criterios de evaluación
├── 0-assets/                    # Diagramas y recursos visuales
│   ├── 01-train-test-split.svg
│   ├── 02-cross-validation.svg
│   ├── 03-metricas-clasificacion.svg
│   ├── 04-curvas-roc-pr.svg
│   └── 05-bias-variance.svg
├── 1-teoria/                    # Material teórico
│   ├── 01-validacion-holdout.md
│   ├── 02-cross-validation.md
│   ├── 03-metricas-clasificacion.md
│   ├── 04-metricas-regresion.md
│   └── 05-optimizacion-hiperparametros.md
├── 2-practicas/                 # Ejercicios guiados
│   ├── ejercicio-01-cross-validation/
│   ├── ejercicio-02-metricas-clasificacion/
│   ├── ejercicio-03-metricas-regresion/
│   └── ejercicio-04-gridsearch/
├── 3-proyecto/                  # Proyecto semanal
│   └── evaluacion-completa-modelo/
├── 4-recursos/                  # Material adicional
│   └── README.md
└── 5-glosario/                  # Términos clave
    └── README.md
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| #   | Tema                            | Archivo                                                                       | Duración |
| --- | ------------------------------- | ----------------------------------------------------------------------------- | -------- |
| 1   | Validación Holdout              | [01-validacion-holdout.md](1-teoria/01-validacion-holdout.md)                 | 15 min   |
| 2   | Cross-Validation                | [02-cross-validation.md](1-teoria/02-cross-validation.md)                     | 25 min   |
| 3   | Métricas de Clasificación       | [03-metricas-clasificacion.md](1-teoria/03-metricas-clasificacion.md)         | 25 min   |
| 4   | Métricas de Regresión           | [04-metricas-regresion.md](1-teoria/04-metricas-regresion.md)                 | 15 min   |
| 5   | Optimización de Hiperparámetros | [05-optimizacion-hiperparametros.md](1-teoria/05-optimizacion-hiperparametros.md) | 20 min   |

### 💻 Prácticas (2.5 horas)

| #   | Ejercicio                    | Carpeta                                                                        | Duración |
| --- | ---------------------------- | ------------------------------------------------------------------------------ | -------- |
| 1   | Cross-Validation en Práctica | [ejercicio-01-cross-validation/](2-practicas/ejercicio-01-cross-validation/)   | 35 min   |
| 2   | Métricas de Clasificación    | [ejercicio-02-metricas-clasificacion/](2-practicas/ejercicio-02-metricas-clasificacion/) | 40 min   |
| 3   | Métricas de Regresión        | [ejercicio-03-metricas-regresion/](2-practicas/ejercicio-03-metricas-regresion/) | 30 min   |
| 4   | GridSearch y RandomSearch    | [ejercicio-04-gridsearch/](2-practicas/ejercicio-04-gridsearch/)               | 45 min   |

### 📦 Proyecto (2 horas)

| Proyecto                       | Descripción                                          | Carpeta                                                                    |
| ------------------------------ | ---------------------------------------------------- | -------------------------------------------------------------------------- |
| Evaluación Completa de Modelo  | Pipeline con CV, métricas múltiples y optimización   | [evaluacion-completa-modelo/](3-proyecto/evaluacion-completa-modelo/)      |

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
   - [ ] ejercicio-01: Cross-validation implementado
   - [ ] ejercicio-02: Métricas de clasificación calculadas
   - [ ] ejercicio-03: Métricas de regresión aplicadas
   - [ ] ejercicio-04: GridSearchCV optimizando modelo

2. **Proyecto semanal** (3-proyecto/)
   - [ ] Pipeline con cross-validation anidado
   - [ ] Reporte de métricas múltiples
   - [ ] Curvas ROC y PR generadas
   - [ ] Modelo optimizado con GridSearchCV

3. **Autoevaluación**
   - [ ] Completar checklist de verificación
   - [ ] Responder cuestionario de conocimientos

---

## 🎯 Competencias a Desarrollar

### Técnicas

- Estrategias de validación (holdout, k-fold, stratified, LOO)
- Métricas de evaluación (accuracy, precision, recall, F1, AUC)
- Interpretación de curvas ROC y Precision-Recall
- Optimización de hiperparámetros

### Transversales

- Pensamiento crítico para selección de métricas
- Análisis de trade-offs (bias-variance)
- Comunicación de resultados de evaluación

---

## 🔗 Navegación

| ⬅️ Anterior                     | 🏠 Inicio                                   | Siguiente ➡️                      |
| ------------------------------ | ------------------------------------------ | --------------------------------- |
| [Semana 14](../week-14/README.md) | [Módulo ML](../../README.md)               | [Semana 16](../week-16/README.md) |

---

## 💡 Tips para esta Semana

> 🎯 **Consejo**: La métrica correcta depende del problema. En fraude, recall es crucial; en spam, precision importa más. Siempre pregunta: ¿cuál es el costo de cada tipo de error?

- **No uses solo accuracy**: Es engañosa con clases desbalanceadas
- **Cross-validation siempre**: Un solo split puede dar resultados engañosos
- **Visualiza las curvas**: ROC y PR cuentan historias diferentes
- **Cuidado con data leakage**: La optimización debe estar dentro del CV

---

_Semana 15 de 36 | Módulo: Machine Learning | Bootcamp IA: Zero to Hero_
