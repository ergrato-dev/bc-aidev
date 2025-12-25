# Proyecto: Clasificación de Vinos con Random Forest

## 🎯 Objetivo

Construir un clasificador de vinos usando Random Forest que alcance **accuracy ≥ 0.92** en el dataset Wine de sklearn.

## 📋 Descripción

El dataset Wine contiene resultados de análisis químicos de vinos cultivados en una misma región de Italia, pero derivados de tres variedades diferentes de uva. El objetivo es clasificar correctamente la variedad de vino basándose en 13 características químicas.

### Dataset Wine

| Característica | Valor                               |
| -------------- | ----------------------------------- |
| Muestras       | 178                                 |
| Features       | 13 (alcohol, malic_acid, ash, etc.) |
| Clases         | 3 (class_0, class_1, class_2)       |
| Tipo           | Clasificación multiclase            |

### Features Disponibles

1. Alcohol
2. Malic acid
3. Ash
4. Alcalinity of ash
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline

## 🏆 Criterios de Éxito

| Métrica          | Mínimo Requerido             |
| ---------------- | ---------------------------- |
| Test Accuracy    | ≥ 0.92                       |
| OOB Score        | Reportar                     |
| Cross-Validation | 5-fold, reportar media ± std |

## 📁 Estructura del Proyecto

```
3-proyecto/
├── README.md
├── starter/
│   └── main.py          # Código inicial con TODOs
└── .solution/           # Solución (no subir a git)
    └── main.py
```

## 📝 Tareas a Implementar

### 1. Carga y Exploración de Datos (20%)

- [ ] Cargar dataset Wine
- [ ] Explorar distribución de clases
- [ ] Mostrar estadísticas básicas de features

### 2. Preprocesamiento (15%)

- [ ] Dividir datos (train/test, 80/20)
- [ ] Usar stratify para mantener proporciones

### 3. Modelo Baseline (15%)

- [ ] Entrenar DecisionTreeClassifier como baseline
- [ ] Evaluar accuracy en train y test

### 4. Random Forest (25%)

- [ ] Entrenar RandomForestClassifier
- [ ] Configurar hiperparámetros adecuados
- [ ] Activar OOB Score
- [ ] Alcanzar accuracy ≥ 0.92

### 5. Evaluación Completa (15%)

- [ ] Calcular accuracy, precision, recall, F1
- [ ] Generar classification report
- [ ] Crear confusion matrix

### 6. Feature Importance (10%)

- [ ] Extraer importancia de features
- [ ] Visualizar top features
- [ ] Identificar las 3 más importantes

## ⏱️ Tiempo Estimado

2 horas

## 🔧 Librerías Requeridas

```python
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## 📊 Entregables

1. **Código funcional** (`main.py`) que:

   - Carga y preprocesa los datos
   - Entrena baseline y Random Forest
   - Alcanza accuracy ≥ 0.92
   - Genera visualizaciones

2. **Métricas reportadas**:

   - Accuracy (train y test)
   - OOB Score
   - Cross-validation scores
   - Classification report completo

3. **Visualizaciones**:
   - Confusion matrix
   - Feature importance (bar chart)

## 💡 Hints

1. **Random Forest funciona bien sin normalizar** - No necesitas StandardScaler
2. **n_estimators=100** es un buen punto de partida
3. **max_depth=None** puede funcionar bien con este dataset pequeño
4. **Usa random_state=42** para reproducibilidad

## 📚 Recursos

- [Wine Dataset - sklearn](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset)
- [RandomForestClassifier - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Classification Report - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

---

## ✅ Checklist Final

- [ ] Código ejecuta sin errores
- [ ] Accuracy test ≥ 0.92
- [ ] OOB Score reportado
- [ ] Cross-validation realizado
- [ ] Confusion matrix generada
- [ ] Feature importance visualizada
- [ ] Código bien comentado
