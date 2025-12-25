# 📊 Semana 08: Pandas para Manipulación de Datos

## 📋 Información General

| Campo          | Detalle                        |
| -------------- | ------------------------------ |
| **Módulo**     | Fundamentos (Semana 8 de 8)    |
| **Duración**   | 6 horas                        |
| **Nivel**      | Principiante-Intermedio        |
| **Requisitos** | Python básico, NumPy (Week-07) |

---

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana serás capaz de:

- ✅ Crear y manipular Series y DataFrames
- ✅ Cargar datos desde CSV, Excel y otras fuentes
- ✅ Seleccionar datos con loc, iloc y filtros booleanos
- ✅ Limpiar y transformar datos (missing values, duplicados)
- ✅ Agrupar datos y calcular estadísticas con groupby
- ✅ Combinar DataFrames con merge y concat
- ✅ Aplicar funciones personalizadas con apply

---

## 📚 Contenidos

### 1. Teoría (1.5 horas)

| Archivo                                                                 | Tema                                 | Duración |
| ----------------------------------------------------------------------- | ------------------------------------ | -------- |
| [01-introduccion-pandas.md](1-teoria/01-introduccion-pandas.md)         | Series, DataFrames, lectura de datos | 25 min   |
| [02-seleccion-filtrado.md](1-teoria/02-seleccion-filtrado.md)           | loc, iloc, filtros booleanos, query  | 25 min   |
| [03-limpieza-transformacion.md](1-teoria/03-limpieza-transformacion.md) | Missing values, duplicados, tipos    | 20 min   |
| [04-agrupacion-combinacion.md](1-teoria/04-agrupacion-combinacion.md)   | groupby, merge, concat, pivot        | 20 min   |

### 2. Prácticas (2.5 horas)

| Ejercicio                                                       | Tema                                 | Duración |
| --------------------------------------------------------------- | ------------------------------------ | -------- |
| [ejercicio-01-dataframes](2-practicas/ejercicio-01-dataframes/) | Creación y exploración de DataFrames | 35 min   |
| [ejercicio-02-seleccion](2-practicas/ejercicio-02-seleccion/)   | Selección y filtrado de datos        | 35 min   |
| [ejercicio-03-limpieza](2-practicas/ejercicio-03-limpieza/)     | Limpieza y transformación            | 40 min   |
| [ejercicio-04-agrupacion](2-practicas/ejercicio-04-agrupacion/) | Agrupación y agregación              | 40 min   |

### 3. Proyecto (2 horas)

| Proyecto                          | Descripción                                                               |
| --------------------------------- | ------------------------------------------------------------------------- |
| [Análisis de Ventas](3-proyecto/) | Analizar dataset de ventas: limpieza, exploración, agregaciones y reporte |

---

## 🗂️ Estructura de la Semana

```
week-08/
├── README.md
├── rubrica-evaluacion.md
├── 0-assets/
│   └── *.svg
├── 1-teoria/
│   ├── 01-introduccion-pandas.md
│   ├── 02-seleccion-filtrado.md
│   ├── 03-limpieza-transformacion.md
│   └── 04-agrupacion-combinacion.md
├── 2-practicas/
│   ├── ejercicio-01-dataframes/
│   ├── ejercicio-02-seleccion/
│   ├── ejercicio-03-limpieza/
│   └── ejercicio-04-agrupacion/
├── 3-proyecto/
│   ├── starter/
│   └── .solution/
├── 4-recursos/
│   ├── ebooks-free/
│   ├── videografia/
│   └── webgrafia/
└── 5-glosario/
```

---

## ⏱️ Distribución del Tiempo

| Actividad | Tiempo  | Porcentaje |
| --------- | ------- | ---------- |
| Teoría    | 1.5 h   | 25%        |
| Prácticas | 2.5 h   | 42%        |
| Proyecto  | 2 h     | 33%        |
| **Total** | **6 h** | **100%**   |

---

## 🔧 Requisitos Técnicos

### Software

- Python 3.10+
- pandas >= 2.0
- numpy (dependencia)
- openpyxl (para Excel)

### Instalación

```bash
pip install pandas openpyxl
```

### Verificación

```python
import pandas as pd
print(pd.__version__)  # >= 2.0.0
```

---

## 📌 Entregables

1. **Ejercicios completados** (4 ejercicios)
2. **Proyecto de Análisis de Ventas** funcional
3. **Reporte con hallazgos** del análisis

---

## 🔗 Navegación

| Anterior                           | Índice                            | Siguiente                                                   |
| ---------------------------------- | --------------------------------- | ----------------------------------------------------------- |
| [⬅️ Semana 07: NumPy](../week-07/) | [📚 Bootcamp](../../../README.md) | [➡️ Módulo 2: Machine Learning](../../02-machine-learning/) |

---

## 💡 Tips de la Semana

> **Pandas = Power** 🐼
>
> Pandas es la herramienta más usada para manipulación de datos en Python.
> Dominar Pandas es esencial para cualquier trabajo en Data Science o ML.
>
> - Siempre explora tus datos primero: `df.head()`, `df.info()`, `df.describe()`
> - Usa `loc` para selección por etiquetas, `iloc` para posición
> - Encadena operaciones para código más limpio
> - Evita loops cuando puedas usar operaciones vectorizadas

---

## 🏆 Logros Desbloqueables

- 🥉 **DataFrame Novice**: Crear tu primer DataFrame
- 🥈 **Data Cleaner**: Limpiar un dataset con missing values
- 🥇 **Aggregation Master**: Usar groupby con múltiples agregaciones
- 💎 **Pandas Expert**: Completar el proyecto de análisis de ventas
