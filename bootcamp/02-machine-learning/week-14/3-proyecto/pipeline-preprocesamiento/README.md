# 🔧 Proyecto: Pipeline de Preprocesamiento Completo

## 🎯 Objetivo

Construir un pipeline de preprocesamiento end-to-end que maneje datos mixtos (numéricos y categóricos), valores faltantes, y selección de características usando sklearn Pipeline y ColumnTransformer.

## 📋 Descripción

En este proyecto crearás un pipeline profesional que:

1. Maneja valores faltantes en numéricas y categóricas
2. Escala variables numéricas
3. Codifica variables categóricas
4. Selecciona las features más relevantes
5. Se integra con un modelo de clasificación
6. Es reproducible y listo para producción

## 📁 Estructura

```
pipeline-preprocesamiento/
├── README.md           # Este archivo
├── starter/
│   └── main.py         # Código para completar
└── solution/
    └── main.py         # Solución de referencia
```

## 📊 Dataset

Trabajaremos con el dataset **Adult Income** (Census Income), que predice si una persona gana más de $50K/año.

- **Fuente**: UCI Machine Learning Repository
- **Features**: 14 (numéricas y categóricas)
- **Target**: income (<=50K, >50K)
- **Samples**: ~48,000

## ⏱️ Tiempo Estimado

2 horas

## 🚀 Requisitos del Pipeline

### 1. Preprocesamiento Numérico

- Imputación con mediana
- Escalado con StandardScaler

### 2. Preprocesamiento Categórico

- Imputación con moda
- Codificación con OneHotEncoder

### 3. Selección de Features

- Aplicar SelectKBest o SelectFromModel
- Mantener las features más relevantes

### 4. Modelo

- LogisticRegression o RandomForestClassifier
- Evaluación con cross-validation

### 5. Entregables

- Pipeline funcional
- Métricas de evaluación (accuracy, precision, recall, F1)
- Análisis de features seleccionadas
- Código documentado

## 📝 Pasos Sugeridos

1. **Cargar y explorar datos**
2. **Identificar tipos de columnas**
3. **Crear pipelines individuales** (numérico, categórico)
4. **Combinar con ColumnTransformer**
5. **Añadir selector de features**
6. **Añadir clasificador**
7. **Evaluar con cross-validation**
8. **Analizar resultados**

## ✅ Criterios de Evaluación

| Criterio                           | Puntos  |
| ---------------------------------- | ------- |
| Pipeline funciona correctamente    | 25      |
| Maneja missing values              | 15      |
| Codificación categórica correcta   | 15      |
| Selección de features implementada | 15      |
| Evaluación con CV                  | 15      |
| Código documentado y limpio        | 15      |
| **Total**                          | **100** |

## 🎯 Métricas Objetivo

- Accuracy: > 0.82
- F1-Score: > 0.60 (para clase minoritaria >50K)

## 📚 Recursos

- [sklearn Pipeline](https://scikit-learn.org/stable/modules/compose.html)
- [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
