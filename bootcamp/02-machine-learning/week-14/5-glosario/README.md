# 📖 Glosario - Semana 14: Feature Engineering

## B

### Binning (Discretización)
Proceso de convertir variables numéricas continuas en categorías discretas. Útil cuando la relación con el target no es lineal.

```python
from sklearn.preprocessing import KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
```

### Box-Cox Transform
Transformación de potencia que normaliza distribuciones sesgadas. Solo funciona con valores estrictamente positivos.

$$y(\lambda) = \begin{cases} \frac{x^\lambda - 1}{\lambda} & \text{si } \lambda \neq 0 \\ \ln(x) & \text{si } \lambda = 0 \end{cases}$$

## C

### Cardinalidad
Número de valores únicos en una variable categórica. Alta cardinalidad (muchas categorías únicas) puede causar problemas con one-hot encoding.

### ColumnTransformer
Clase de sklearn que permite aplicar diferentes transformaciones a diferentes columnas de un DataFrame.

```python
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])
```

### Cross-Validation (Validación Cruzada)
Técnica para evaluar modelos dividiendo los datos en múltiples folds, entrenando en algunos y validando en otros.

## D

### Data Leakage (Fuga de Datos)
Error que ocurre cuando información del conjunto de test "se filtra" al entrenamiento, resultando en métricas optimistas pero mal rendimiento en producción.

**Causa común**: Hacer fit de transformadores en todo el dataset antes del split.

## E

### Embedded Methods
Métodos de selección de features donde la selección ocurre durante el entrenamiento del modelo (ej: Lasso, Random Forest feature importances).

## F

### Feature Engineering
Proceso de crear, transformar y seleccionar características (features) para mejorar el rendimiento de modelos de ML.

### Feature Importance
Medida de cuánto contribuye cada feature a las predicciones del modelo. Los modelos basados en árboles proporcionan esta métrica directamente.

### Feature Selection
Proceso de seleccionar un subconjunto de features relevantes para el modelo, eliminando redundantes o irrelevantes.

### Filter Methods
Métodos de selección de features que evalúan cada feature independientemente del modelo, usando métricas estadísticas.

### Fit / Transform
- **fit()**: Calcula parámetros del transformador usando los datos (ej: media y std para StandardScaler)
- **transform()**: Aplica la transformación usando parámetros ya calculados
- **fit_transform()**: Hace ambos en un solo paso (solo para train)

## I

### Imputation (Imputación)
Proceso de rellenar valores faltantes con valores estimados.

| Estrategia | Uso |
|------------|-----|
| mean | Numéricas, distribución simétrica |
| median | Numéricas, con outliers |
| most_frequent | Categóricas |
| KNN | Cuando hay correlación entre features |

### IQR (Interquartile Range)
Rango intercuartílico: diferencia entre el percentil 75 (Q3) y el percentil 25 (Q1). Usado por RobustScaler.

$$IQR = Q_3 - Q_1$$

## K

### KNNImputer
Imputador que rellena valores faltantes basándose en los K vecinos más cercanos de cada muestra.

## L

### LabelEncoder
Codificador que convierte categorías a enteros. **Solo debe usarse para la variable target**, nunca para features.

### Log Transform
Transformación logarítmica para reducir el sesgo en distribuciones con cola larga.

```python
import numpy as np
X_log = np.log1p(X)  # log(1+x) para manejar ceros
X_original = np.expm1(X_log)  # Inversa
```

## M

### MCAR/MAR/MNAR
Tipos de valores faltantes:
- **MCAR** (Missing Completely At Random): La probabilidad de missing no depende de ninguna variable
- **MAR** (Missing At Random): La probabilidad depende de variables observadas
- **MNAR** (Missing Not At Random): La probabilidad depende del valor faltante mismo

### MinMaxScaler
Escalador que transforma features al rango [0, 1].

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

### Missing Indicator
Feature binaria que indica si el valor original estaba faltante. Útil cuando el missing es informativo.

### Multicolinealidad
Cuando dos o más features están altamente correlacionadas. Puede causar problemas en modelos lineales.

## N

### Nominal Variable
Variable categórica sin orden natural (ej: colores, ciudades). Usar OneHotEncoder.

## O

### OneHotEncoder
Codificador que crea una columna binaria por cada categoría única.

```
Color: [rojo, verde, azul]
→ color_rojo: [1, 0, 0]
→ color_verde: [0, 1, 0]
→ color_azul: [0, 0, 1]
```

### Ordinal Variable
Variable categórica con orden natural (ej: bajo < medio < alto). Usar OrdinalEncoder.

### OrdinalEncoder
Codificador que asigna enteros respetando un orden definido.

## P

### Pipeline
Secuencia de transformaciones y un estimador final, encapsulados en un solo objeto.

```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
```

### PowerTransformer
Transformador que aplica Box-Cox o Yeo-Johnson para normalizar distribuciones.

## R

### RFE (Recursive Feature Elimination)
Método wrapper que elimina features recursivamente basándose en la importancia del modelo.

### RobustScaler
Escalador que usa la mediana y el IQR, siendo robusto a outliers.

$$x' = \frac{x - Q_2}{Q_3 - Q_1}$$

## S

### Scaling (Escalado)
Proceso de transformar features para que tengan escalas comparables.

### SelectFromModel
Selector de features basado en las importancias de un modelo entrenado.

### SelectKBest
Selector que mantiene las K features con mejores scores según una métrica estadística.

### Skewness (Sesgo)
Medida de asimetría de una distribución. Valores > 0 indican cola hacia la derecha.

### StandardScaler
Escalador que centra los datos con media 0 y desviación estándar 1 (Z-score).

$$z = \frac{x - \mu}{\sigma}$$

## T

### TargetEncoder
Codificador que reemplaza cada categoría por la media del target para esa categoría.

### Transformer
Objeto que implementa fit() y transform() para transformar datos.

## V

### VarianceThreshold
Selector que elimina features con varianza menor a un umbral. Útil para eliminar features constantes.

## W

### Wrapper Methods
Métodos de selección de features que usan un modelo como evaluador (ej: RFE, Forward Selection).

## Y

### Yeo-Johnson Transform
Similar a Box-Cox pero funciona con valores negativos y cero.

---

## Fórmulas Clave

| Transformación | Fórmula |
|----------------|---------|
| Z-score | $z = \frac{x - \mu}{\sigma}$ |
| Min-Max | $x' = \frac{x - min}{max - min}$ |
| Robust | $x' = \frac{x - Q_2}{Q_3 - Q_1}$ |
| Log | $x' = \log(x + 1)$ |
