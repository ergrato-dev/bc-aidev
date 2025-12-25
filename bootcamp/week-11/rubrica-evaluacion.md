# 📊 Rúbrica de Evaluación - Semana 11

## Árboles de Decisión y Random Forest

### Distribución de Evidencias

| Tipo            | Porcentaje | Descripción                                |
| --------------- | ---------- | ------------------------------------------ |
| 🧠 Conocimiento | 30%        | Comprensión teórica de árboles y ensambles |
| 💪 Desempeño    | 40%        | Ejercicios prácticos completados           |
| 📦 Producto     | 30%        | Proyecto clasificador de vinos             |

---

## 🧠 Conocimiento (30%)

### Conceptos Evaluados

| Concepto               | Peso | Criterio de Evaluación              |
| ---------------------- | ---- | ----------------------------------- |
| Estructura de árboles  | 20%  | Explica nodos, hojas, profundidad   |
| Criterios de división  | 25%  | Diferencia Gini vs Entropy          |
| Overfitting en árboles | 20%  | Entiende poda y regularización      |
| Random Forest          | 25%  | Explica bagging y votación          |
| Feature importance     | 10%  | Interpreta importancia de variables |

### Niveles de Desempeño

| Nivel        | Puntos | Descripción                            |
| ------------ | ------ | -------------------------------------- |
| Excelente    | 90-100 | Explica conceptos con ejemplos propios |
| Bueno        | 75-89  | Comprende y aplica correctamente       |
| Suficiente   | 60-74  | Conoce conceptos básicos               |
| Insuficiente | <60    | No demuestra comprensión               |

---

## 💪 Desempeño (40%)

### Ejercicios Prácticos

| Ejercicio              | Peso | Criterios                                  |
| ---------------------- | ---- | ------------------------------------------ |
| Árbol de clasificación | 25%  | Entrena, evalúa, visualiza árbol           |
| Árbol de regresión     | 25%  | Aplica DecisionTreeRegressor correctamente |
| Random Forest          | 30%  | Implementa RF con parámetros ajustados     |
| Feature importance     | 20%  | Extrae e interpreta importancias           |

### Criterios por Ejercicio

#### Ejercicio 01: Árbol de Clasificación

| Criterio                            | Puntos |
| ----------------------------------- | ------ |
| Carga y prepara datos correctamente | 20     |
| Entrena DecisionTreeClassifier      | 25     |
| Evalúa con métricas apropiadas      | 25     |
| Visualiza el árbol                  | 20     |
| Código limpio y comentado           | 10     |

#### Ejercicio 02: Árbol de Regresión

| Criterio                                     | Puntos |
| -------------------------------------------- | ------ |
| Usa DecisionTreeRegressor                    | 25     |
| Controla profundidad para evitar overfitting | 30     |
| Evalúa con R² y MAE                          | 25     |
| Compara diferentes profundidades             | 20     |

#### Ejercicio 03: Random Forest

| Criterio                          | Puntos |
| --------------------------------- | ------ |
| Implementa RandomForestClassifier | 25     |
| Ajusta n_estimators y max_depth   | 25     |
| Usa OOB score o cross-validation  | 25     |
| Compara con árbol individual      | 25     |

#### Ejercicio 04: Feature Importance

| Criterio                          | Puntos |
| --------------------------------- | ------ |
| Extrae feature*importances*       | 30     |
| Visualiza importancias en gráfico | 30     |
| Interpreta resultados             | 25     |
| Experimenta eliminando features   | 15     |

---

## 📦 Producto (30%)

### Proyecto: Clasificador de Vinos

**Dataset**: Wine dataset (sklearn) - 3 clases, 13 features

| Criterio          | Peso | Descripción                      |
| ----------------- | ---- | -------------------------------- |
| **Funcionalidad** | 35%  | Código ejecuta sin errores       |
| **Rendimiento**   | 25%  | Accuracy ≥ 0.92 en test          |
| **Metodología**   | 20%  | Train/test split, CV para tuning |
| **Análisis**      | 15%  | Feature importance, conclusiones |
| **Código**        | 5%   | Limpio, documentado, modular     |

### Niveles de Rendimiento

| Accuracy Test | Calificación     |
| ------------- | ---------------- |
| ≥ 0.95        | Excelente (100%) |
| 0.92 - 0.94   | Bueno (85%)      |
| 0.88 - 0.91   | Suficiente (70%) |
| < 0.88        | Requiere mejora  |

### Checklist del Proyecto

```
□ EDA básico del dataset
□ Train/test split (80/20)
□ Random Forest entrenado
□ GridSearchCV o RandomizedSearchCV para tuning
□ Métricas: accuracy, precision, recall, F1
□ Matriz de confusión
□ Feature importance visualizada
□ Comparación con árbol individual
□ Conclusiones documentadas
```

---

## 📋 Criterios Generales

### Código Python

| Aspecto          | Esperado                                 |
| ---------------- | ---------------------------------------- |
| Estilo           | PEP 8, nombres descriptivos              |
| Documentación    | Docstrings en funciones                  |
| Imports          | Organizados (stdlib, third-party, local) |
| Reproducibilidad | random_state fijado                      |

### Visualizaciones

| Aspecto  | Esperado                          |
| -------- | --------------------------------- |
| Claridad | Títulos, labels, leyendas         |
| Formato  | Figuras guardadas en PNG          |
| Estilo   | Tema consistente (dark preferido) |

---

## 🎯 Calificación Final

```
Nota Final = (Conocimiento × 0.30) + (Desempeño × 0.40) + (Producto × 0.30)
```

### Escala de Aprobación

| Rango | Resultado     |
| ----- | ------------- |
| ≥ 90  | Sobresaliente |
| 80-89 | Notable       |
| 70-79 | Aprobado      |
| 60-69 | Suficiente    |
| < 60  | No aprobado   |

**Nota mínima para aprobar**: 70% en cada tipo de evidencia.

---

_Rúbrica Semana 11 | Árboles de Decisión y Random Forest_
