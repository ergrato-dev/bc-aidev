# 🏠 Proyecto: Predicción de Precios de Casas

## 🎯 Objetivo

Construir un modelo de regresión completo para predecir precios de casas utilizando el dataset California Housing, aplicando todo lo aprendido en la semana.

## 📋 Competencias a Evaluar

- Análisis exploratorio de datos (EDA)
- Preprocesamiento y feature engineering
- Implementación de regresión lineal múltiple
- Aplicación de regularización (Ridge/Lasso)
- Evaluación y comparación de modelos
- Interpretación de resultados

## 🛠️ Requisitos

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 📊 Dataset

Usaremos **California Housing** de sklearn (20,640 muestras, 8 features):

| Feature    | Descripción                    |
| ---------- | ------------------------------ |
| MedInc     | Ingreso medio del área         |
| HouseAge   | Edad media de las casas        |
| AveRooms   | Promedio de habitaciones       |
| AveBedrms  | Promedio de dormitorios        |
| Population | Población del área             |
| AveOccup   | Ocupación promedio             |
| Latitude   | Latitud                        |
| Longitude  | Longitud                       |
| **Target** | **Precio medio (en $100,000)** |

## 📝 Estructura del Proyecto

```
3-proyecto/
├── README.md
├── starter/
│   └── main.py          # Código base con TODOs
└── .solution/           # (Solo local, en .gitignore)
    └── main.py
```

## 🚀 Instrucciones

### Parte 1: Carga y Exploración (EDA)

1. Cargar el dataset California Housing
2. Explorar estadísticas descriptivas
3. Visualizar distribuciones y correlaciones
4. Identificar posibles outliers

### Parte 2: Preprocesamiento

1. Dividir en train/test (80/20)
2. Escalar features con StandardScaler
3. Analizar correlación entre features (multicolinealidad)

### Parte 3: Modelado

1. Entrenar LinearRegression como baseline
2. Entrenar Ridge con varios valores de α
3. Entrenar Lasso con varios valores de α
4. Usar cross-validation para seleccionar mejor α

### Parte 4: Evaluación

1. Calcular métricas en test: R², MAE, RMSE
2. Comparar los 3 modelos
3. Analizar coeficientes e importancia de features
4. Visualizar predicciones vs valores reales

### Parte 5: Conclusiones

1. ¿Qué modelo funciona mejor y por qué?
2. ¿Qué features son más importantes?
3. ¿Hay evidencia de multicolinealidad?
4. ¿Cómo mejorarías el modelo?

---

## ✅ Criterios de Éxito

| Métrica          | Mínimo Esperado     |
| ---------------- | ------------------- |
| R² en Test       | ≥ 0.60              |
| Código funcional | Sin errores         |
| Visualizaciones  | Mínimo 3 gráficos   |
| Comparación      | 3 modelos evaluados |

## 📦 Entregables

1. `main.py` completado y funcional
2. Gráficos generados (PNG)
3. Respuestas a las preguntas de conclusión (en comentarios o print)

---

## 💡 Hints

```python
# Cargar dataset
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

# Cross-validation para α
from sklearn.linear_model import RidgeCV, LassoCV
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_cv = RidgeCV(alphas=alphas, cv=5)

# Métricas
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

---

## 📚 Recursos

- [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- [RidgeCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)
- [Feature Importance in Linear Models](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html)

---

## ⏱️ Tiempo Estimado

- Parte 1 (EDA): 30 min
- Parte 2 (Preprocesamiento): 20 min
- Parte 3 (Modelado): 40 min
- Parte 4 (Evaluación): 20 min
- Parte 5 (Conclusiones): 10 min
- **Total**: ~2 horas
