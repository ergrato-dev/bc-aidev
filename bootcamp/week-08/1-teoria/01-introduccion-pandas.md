# 📊 Introducción a Pandas

## 🎯 Objetivos

- Entender qué es Pandas y su rol en el ecosistema Python
- Dominar las estructuras básicas: Series y DataFrame
- Cargar datos desde diferentes fuentes
- Explorar datasets con métodos básicos

---

## 1. ¿Qué es Pandas?

**Pandas** es la librería estándar de Python para manipulación y análisis de datos tabulares. Su nombre viene de "Panel Data" (datos de panel en econometría).

### ¿Por qué Pandas?

| Característica             | Descripción                          |
| -------------------------- | ------------------------------------ |
| **Estructuras potentes**   | Series (1D) y DataFrame (2D)         |
| **Lectura de datos**       | CSV, Excel, SQL, JSON, HTML, Parquet |
| **Manipulación eficiente** | Filtrado, agrupación, transformación |
| **Manejo de missing data** | NaN handling integrado               |
| **Integración**            | NumPy, Matplotlib, Scikit-learn      |

### Ecosistema

![Ecosistema Pandas](../0-assets/01-pandas-ecosystem.svg)

```python
import pandas as pd
import numpy as np

print(pd.__version__)  # >= 2.0.0
```

---

## 2. Series: Arrays 1D con Etiquetas

Una **Series** es un array unidimensional con etiquetas (índice).

### Creación de Series

```python
import pandas as pd

# Desde lista
s = pd.Series([10, 20, 30, 40])
print(s)
# 0    10
# 1    20
# 2    30
# 3    40
# dtype: int64

# Con índice personalizado
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)
# a    10
# b    20
# c    30

# Desde diccionario
d = {'manzanas': 10, 'naranjas': 20, 'plátanos': 15}
s = pd.Series(d)
print(s)
# manzanas    10
# naranjas    20
# plátanos    15
```

### Atributos de Series

```python
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'], name='ventas')

print(s.values)   # array([10, 20, 30])
print(s.index)    # Index(['a', 'b', 'c'], dtype='object')
print(s.dtype)    # int64
print(s.name)     # 'ventas'
print(s.shape)    # (3,)
print(len(s))     # 3
```

### Acceso a Elementos

```python
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])

# Por etiqueta
print(s['a'])       # 10
print(s[['a', 'c']])  # Series con a y c

# Por posición
print(s[0])         # 10
print(s[0:2])       # Primeros 2 elementos

# Por condición
print(s[s > 20])    # Elementos > 20
```

### Operaciones con Series

```python
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# Operaciones vectorizadas
print(s1 + s2)      # [11, 22, 33]
print(s1 * 2)       # [2, 4, 6]
print(s2 / s1)      # [10, 10, 10]

# Funciones NumPy
print(np.sqrt(s2))  # Raíz cuadrada

# Métodos estadísticos
print(s2.sum())     # 60
print(s2.mean())    # 20.0
print(s2.std())     # 10.0
```

---

## 3. DataFrame: Tablas 2D

Un **DataFrame** es una estructura bidimensional con filas y columnas etiquetadas. Es la estructura más usada en Pandas.

### Creación de DataFrames

```python
import pandas as pd

# Desde diccionario de listas
data = {
    'nombre': ['Ana', 'Bob', 'Carlos'],
    'edad': [25, 30, 35],
    'ciudad': ['Madrid', 'Barcelona', 'Valencia']
}
df = pd.DataFrame(data)
print(df)
#   nombre  edad     ciudad
# 0    Ana    25     Madrid
# 1    Bob    30  Barcelona
# 2 Carlos    35   Valencia

# Desde lista de diccionarios
data = [
    {'nombre': 'Ana', 'edad': 25},
    {'nombre': 'Bob', 'edad': 30},
    {'nombre': 'Carlos', 'edad': 35}
]
df = pd.DataFrame(data)

# Desde array NumPy
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
```

### Atributos del DataFrame

```python
df = pd.DataFrame({
    'nombre': ['Ana', 'Bob', 'Carlos'],
    'edad': [25, 30, 35],
    'salario': [50000, 60000, 55000]
})

print(df.shape)      # (3, 3) - filas, columnas
print(df.columns)    # Index(['nombre', 'edad', 'salario'])
print(df.index)      # RangeIndex(start=0, stop=3, step=1)
print(df.dtypes)     # Tipos de cada columna
print(df.values)     # Array NumPy subyacente
print(len(df))       # 3 filas
```

### Acceso a Columnas

```python
# Una columna (retorna Series)
print(df['nombre'])
print(df.nombre)     # Notación de atributo (si no hay espacios)

# Múltiples columnas (retorna DataFrame)
print(df[['nombre', 'edad']])
```

---

## 4. Lectura de Datos

Pandas puede leer datos de múltiples fuentes.

### CSV (Comma-Separated Values)

```python
# Leer CSV
df = pd.read_csv('datos.csv')

# Con opciones
df = pd.read_csv(
    'datos.csv',
    sep=';',              # Separador (default: ',')
    header=0,             # Fila de encabezados (default: 0)
    index_col='id',       # Columna como índice
    usecols=['col1', 'col2'],  # Solo estas columnas
    nrows=100,            # Primeras N filas
    skiprows=1,           # Saltar filas iniciales
    na_values=['NA', ''],  # Valores a tratar como NaN
    parse_dates=['fecha'],  # Parsear como fecha
    encoding='utf-8'      # Encoding del archivo
)

# Guardar a CSV
df.to_csv('output.csv', index=False)
```

### Excel

```python
# Leer Excel (requiere openpyxl)
df = pd.read_excel('datos.xlsx')

# Hoja específica
df = pd.read_excel('datos.xlsx', sheet_name='Ventas')

# Todas las hojas (retorna diccionario)
all_sheets = pd.read_excel('datos.xlsx', sheet_name=None)

# Guardar a Excel
df.to_excel('output.xlsx', index=False, sheet_name='Datos')
```

### Otras Fuentes

```python
# JSON
df = pd.read_json('datos.json')

# SQL (requiere SQLAlchemy)
from sqlalchemy import create_engine
engine = create_engine('sqlite:///database.db')
df = pd.read_sql('SELECT * FROM tabla', engine)

# Desde URL
url = 'https://ejemplo.com/datos.csv'
df = pd.read_csv(url)

# Clipboard (útil para copiar de Excel)
df = pd.read_clipboard()
```

---

## 5. Exploración Básica

Métodos esenciales para entender tus datos.

### Vista Rápida

```python
df = pd.read_csv('ventas.csv')

# Primeras/últimas filas
print(df.head())      # Primeras 5 filas
print(df.head(10))    # Primeras 10 filas
print(df.tail())      # Últimas 5 filas

# Muestra aleatoria
print(df.sample(5))   # 5 filas aleatorias
```

### Información del DataFrame

```python
# Información general
print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1000 entries, 0 to 999
# Data columns (total 5 columns):
#  #   Column   Non-Null Count  Dtype
# ---  ------   --------------  -----
#  0   id       1000 non-null   int64
#  1   nombre   998 non-null    object
#  2   precio   950 non-null    float64
# ...

# Dimensiones
print(df.shape)       # (1000, 5)

# Tipos de datos
print(df.dtypes)

# Uso de memoria
print(df.memory_usage(deep=True))
```

### Estadísticas Descriptivas

```python
# Resumen estadístico (columnas numéricas)
print(df.describe())
#        precio     cantidad
# count  950.00      1000.00
# mean    45.50        10.20
# std     12.30         3.50
# min     10.00         1.00
# 25%     35.00         8.00
# 50%     45.00        10.00
# 75%     55.00        12.00
# max     90.00        25.00

# Incluir todas las columnas
print(df.describe(include='all'))

# Estadísticas individuales
print(df['precio'].mean())    # Media
print(df['precio'].median())  # Mediana
print(df['precio'].std())     # Desviación estándar
print(df['precio'].min())     # Mínimo
print(df['precio'].max())     # Máximo
```

### Valores Únicos y Conteos

```python
# Valores únicos
print(df['categoria'].unique())       # Array de valores únicos
print(df['categoria'].nunique())      # Número de valores únicos

# Conteo de valores
print(df['categoria'].value_counts())
# Electrónica    350
# Ropa           280
# Hogar          200
# Alimentos      170
```

---

## 6. Ejemplo Completo

```python
import pandas as pd

# Crear DataFrame
ventas = pd.DataFrame({
    'producto': ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Laptop'],
    'precio': [1200, 25, 75, 300, 1100],
    'cantidad': [5, 50, 30, 10, 8],
    'categoria': ['Electrónica', 'Accesorios', 'Accesorios', 'Electrónica', 'Electrónica']
})

# Exploración
print("=== Información del DataFrame ===")
print(f"Shape: {ventas.shape}")
print(f"Columnas: {list(ventas.columns)}")
print()

print("=== Primeras filas ===")
print(ventas.head())
print()

print("=== Estadísticas ===")
print(ventas.describe())
print()

print("=== Ventas por categoría ===")
print(ventas['categoria'].value_counts())
print()

# Calcular total de ventas
ventas['total'] = ventas['precio'] * ventas['cantidad']
print("=== Con columna total ===")
print(ventas)
print()

print(f"Ingresos totales: ${ventas['total'].sum():,}")
```

---

## ✅ Resumen

| Concepto           | Descripción                   |
| ------------------ | ----------------------------- |
| **Series**         | Array 1D con índice           |
| **DataFrame**      | Tabla 2D con filas y columnas |
| **read_csv()**     | Cargar datos desde CSV        |
| **head()/tail()**  | Ver primeras/últimas filas    |
| **info()**         | Información del DataFrame     |
| **describe()**     | Estadísticas descriptivas     |
| **value_counts()** | Contar valores únicos         |

---

## 🔗 Siguiente

[Selección y Filtrado de Datos →](02-seleccion-filtrado.md)
