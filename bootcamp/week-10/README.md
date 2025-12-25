# Semana 10: Regresión Lineal y Logística

## 📋 Información General

| Campo              | Detalle                         |
| ------------------ | ------------------------------- |
| **Módulo**         | Machine Learning (Semanas 9-18) |
| **Semana**         | 10 de 36                        |
| **Tema**           | Regresión Lineal y Logística    |
| **Duración**       | 6 horas                         |
| **Prerrequisitos** | Semana 09 (Fundamentos de ML)   |

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

1. **Comprender** la diferencia entre regresión y clasificación
2. **Implementar** regresión lineal simple y múltiple
3. **Aplicar** regresión logística para clasificación binaria
4. **Evaluar** modelos de regresión con métricas apropiadas
5. **Interpretar** coeficientes y su significado
6. **Regularizar** modelos con Ridge y Lasso

## 📚 Contenidos

### Teoría (1.5 horas)

1. [Regresión Lineal Simple](1-teoria/01-regresion-lineal-simple.md)
2. [Regresión Lineal Múltiple](1-teoria/02-regresion-lineal-multiple.md)
3. [Regresión Logística](1-teoria/03-regresion-logistica.md)
4. [Regularización: Ridge y Lasso](1-teoria/04-regularizacion.md)

### Prácticas (2.5 horas)

1. [Ejercicio 01: Regresión Lineal Simple](2-practicas/ejercicio-01-regresion-simple/)
2. [Ejercicio 02: Regresión Múltiple](2-practicas/ejercicio-02-regresion-multiple/)
3. [Ejercicio 03: Regresión Logística](2-practicas/ejercicio-03-regresion-logistica/)
4. [Ejercicio 04: Comparación de Modelos](2-practicas/ejercicio-04-comparacion/)

### Proyecto (2 horas)

- [Predicción de Precios de Casas](3-proyecto/)

## 🗂️ Estructura de la Semana

```
week-10/
├── README.md
├── rubrica-evaluacion.md
├── 0-assets/
│   ├── 01-regresion-lineal.svg
│   ├── 02-gradiente-descendente.svg
│   ├── 03-regresion-multiple.svg
│   ├── 04-sigmoide.svg
│   ├── 05-decision-boundary.svg
│   └── 06-regularizacion.svg
├── 1-teoria/
│   ├── 01-regresion-lineal-simple.md
│   ├── 02-regresion-lineal-multiple.md
│   ├── 03-regresion-logistica.md
│   └── 04-regularizacion.md
├── 2-practicas/
│   ├── ejercicio-01-regresion-simple/
│   ├── ejercicio-02-regresion-multiple/
│   ├── ejercicio-03-regresion-logistica/
│   └── ejercicio-04-comparacion/
├── 3-proyecto/
│   ├── README.md
│   └── starter/
├── 4-recursos/
│   ├── ebooks-free/
│   ├── videografia/
│   └── webgrafia/
└── 5-glosario/
```

## ⏱️ Distribución del Tiempo

| Actividad | Tiempo | Descripción                        |
| --------- | ------ | ---------------------------------- |
| Teoría    | 1.5h   | Lectura y comprensión de conceptos |
| Prácticas | 2.5h   | Ejercicios guiados de regresión    |
| Proyecto  | 2h     | Predicción de precios de casas     |
| **Total** | **6h** |                                    |

## 🔧 Herramientas y Tecnologías

- **Python 3.10+**
- **Scikit-learn**: LinearRegression, LogisticRegression, Ridge, Lasso
- **NumPy**: Operaciones numéricas
- **Pandas**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualización

## 📦 Instalación

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## 📌 Entregables

1. **Ejercicios completados** (4 ejercicios)
2. **Proyecto**: Modelo de predicción de precios con R² ≥ 0.7
3. **Análisis**: Interpretación de coeficientes

## 🔗 Navegación

| Anterior                                    | Índice             | Siguiente                                     |
| ------------------------------------------- | ------------------ | --------------------------------------------- |
| [Semana 09: Fundamentos de ML](../week-09/) | [Bootcamp](../../) | [Semana 11: Árboles de Decisión](../week-11/) |

---

## 📖 Referencias Principales

- [Sklearn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [ISLR - Chapter 3: Linear Regression](https://www.statlearning.com/)
- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=7ArmBVF2dCs)
