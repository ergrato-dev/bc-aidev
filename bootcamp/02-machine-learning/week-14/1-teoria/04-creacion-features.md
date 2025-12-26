# 🛠️ Creación de Features y Manejo de Missing Data

## 🎯 Objetivos

- Crear nuevas features derivadas
- Dominar estrategias de imputación
- Aplicar técnicas de feature engineering temporal

---

## 📋 Contenido

### 1. Creación de Features Derivadas

#### Interacciones Polinómicas

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.array([[2, 3], [4, 5]])

# Grado 2: incluye interacciones x1*x2 y términos x1², x2²
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print(poly.get_feature_names_out(['x1', 'x2']))
# ['x1', 'x2', 'x1^2', 'x1 x2', 'x2^2']
```

#### Features Matemáticas

```python
import pandas as pd

df = pd.DataFrame({
    'largo': [10, 20, 15],
    'ancho': [5, 8, 6],
    'alto': [3, 4, 5]
})

# Crear features derivadas
df['area'] = df['largo'] * df['ancho']
df['volumen'] = df['largo'] * df['ancho'] * df['alto']
df['ratio_largo_ancho'] = df['largo'] / df['ancho']
df['perimetro'] = 2 * (df['largo'] + df['ancho'])
```

#### Features Agregadas

```python
# Agregaciones por grupo
df['precio_medio_categoria'] = df.groupby('categoria')['precio'].transform('mean')
df['precio_max_ciudad'] = df.groupby('ciudad')['precio'].transform('max')
df['count_por_usuario'] = df.groupby('usuario_id')['id'].transform('count')
```

### 2. Feature Engineering Temporal

```python
# Extraer componentes de fecha
df['fecha'] = pd.to_datetime(df['fecha'])

df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month
df['dia'] = df['fecha'].dt.day
df['dia_semana'] = df['fecha'].dt.dayofweek  # 0=Lunes
df['es_fin_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
df['trimestre'] = df['fecha'].dt.quarter
df['dia_año'] = df['fecha'].dt.dayofyear

# Features cíclicas (para capturar periodicidad)
import numpy as np
df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
```

### 3. Manejo de Valores Faltantes

#### Diagnóstico

```python
# Ver missing values
print(df.isnull().sum())
print(df.isnull().sum() / len(df) * 100)  # Porcentaje

# Visualizar patrón
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
plt.title('Missing Values Pattern')
plt.show()
```

#### Estrategias de Imputación

| Estrategia           | Cuándo Usar                       |
| -------------------- | --------------------------------- |
| Eliminar filas       | < 5% missing, aleatorio           |
| Media/Mediana        | Numéricas, distribución simétrica |
| Moda                 | Categóricas                       |
| KNN                  | Relaciones entre features         |
| Indicador de missing | El missing es informativo         |

#### SimpleImputer

```python
from sklearn.impute import SimpleImputer

# Para numéricas
imputer_num = SimpleImputer(strategy='median')  # o 'mean'

# Para categóricas
imputer_cat = SimpleImputer(strategy='most_frequent')

# Valor constante
imputer_const = SimpleImputer(strategy='constant', fill_value=0)
```

#### KNNImputer

```python
from sklearn.impute import KNNImputer

# Imputa basándose en los K vecinos más cercanos
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

#### IterativeImputer (MICE)

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Imputa usando un modelo iterativo
imputer = IterativeImputer(max_iter=10, random_state=42)
X_imputed = imputer.fit_transform(X)
```

### 4. Missing Indicator

A veces el hecho de que falte un valor es informativo:

```python
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion

# Combinar imputación + indicador
transformer = FeatureUnion([
    ('imputer', SimpleImputer(strategy='median')),
    ('indicator', MissingIndicator())
])

X_transformed = transformer.fit_transform(X)
```

### 5. Binning (Discretización)

Convertir numéricas continuas en categorías:

```python
from sklearn.preprocessing import KBinsDiscretizer

# Discretizar en bins
discretizer = KBinsDiscretizer(
    n_bins=5,
    encode='ordinal',  # o 'onehot'
    strategy='quantile'  # o 'uniform', 'kmeans'
)

df['edad_bins'] = discretizer.fit_transform(df[['edad']])

# Manual con pandas
df['edad_grupo'] = pd.cut(
    df['edad'],
    bins=[0, 18, 35, 50, 65, 100],
    labels=['joven', 'adulto_joven', 'adulto', 'senior', 'mayor']
)
```

### 6. Feature Engineering por Dominio

#### E-commerce

```python
df['dias_desde_ultima_compra'] = (hoy - df['ultima_compra']).dt.days
df['frecuencia_compra'] = df['total_compras'] / df['meses_cliente']
df['ticket_medio'] = df['total_gastado'] / df['total_compras']
```

#### Finanzas

```python
df['ratio_deuda_ingreso'] = df['deuda_total'] / df['ingreso_anual']
df['utilizacion_credito'] = df['saldo_credito'] / df['limite_credito']
```

#### Geoespacial

```python
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """Distancia entre dos puntos geográficos"""
    R = 6371  # Radio de la Tierra en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))
```

---

## 💻 Ejemplo Completo

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Definir columnas
numeric_features = ['edad', 'ingresos', 'antiguedad']
categorical_features = ['ciudad', 'educacion']

# Pipeline numérico
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline categórico
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
```

---

## ✅ Checklist de Verificación

- [ ] Puedo crear features derivadas
- [ ] Sé extraer features de fechas
- [ ] Domino las estrategias de imputación
- [ ] Entiendo cuándo usar KNNImputer vs SimpleImputer
