# 📖 Glosario - Semana 08: Pandas

## A

### agg() / aggregate()

Método para aplicar múltiples funciones de agregación a grupos o columnas.

```python
df.groupby('col').agg(['sum', 'mean', 'count'])
```

### apply()

Método para aplicar una función a lo largo de un eje del DataFrame o a una Series.

```python
df['col'].apply(lambda x: x * 2)
df.apply(func, axis=1)  # Por fila
```

### astype()

Método para convertir el tipo de datos de una columna.

```python
df['col'].astype(int)
df['col'].astype('category')
```

---

## B

### Boolean Indexing

Técnica de filtrado usando condiciones booleanas.

```python
df[df['edad'] > 30]
df[(df['a'] > 1) & (df['b'] < 5)]
```

### Broadcasting

Operación donde Pandas alinea automáticamente datos de diferentes formas.

---

## C

### concat()

Función para concatenar DataFrames vertical u horizontalmente.

```python
pd.concat([df1, df2], axis=0)  # Vertical
pd.concat([df1, df2], axis=1)  # Horizontal
```

### crosstab()

Función para calcular tablas de contingencia (frecuencias cruzadas).

```python
pd.crosstab(df['col1'], df['col2'])
```

---

## D

### DataFrame

Estructura de datos 2D con filas y columnas etiquetadas. Tabla de datos principal en Pandas.

```python
df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
```

### dropna()

Método para eliminar filas o columnas con valores faltantes.

```python
df.dropna()              # Elimina filas con NaN
df.dropna(axis=1)        # Elimina columnas con NaN
df.dropna(subset=['col']) # Solo si NaN en columnas específicas
```

### drop_duplicates()

Método para eliminar filas duplicadas.

```python
df.drop_duplicates()
df.drop_duplicates(subset=['col'])
```

### dtypes

Atributo que muestra los tipos de datos de cada columna.

```python
df.dtypes
```

### duplicated()

Método que retorna una Series booleana indicando filas duplicadas.

```python
df.duplicated()
df[df.duplicated()]  # Ver duplicados
```

---

## F

### fillna()

Método para rellenar valores faltantes.

```python
df['col'].fillna(0)
df['col'].fillna(df['col'].mean())
df.fillna({'col1': 0, 'col2': 'N/A'})
```

### filter()

Método de groupby para filtrar grupos basados en condición.

```python
df.groupby('col').filter(lambda x: len(x) > 2)
```

---

## G

### groupby()

Método para agrupar datos por una o más columnas y aplicar operaciones.

```python
df.groupby('col')['valor'].sum()
df.groupby(['col1', 'col2']).agg({'valor': 'mean'})
```

---

## H

### head()

Método para ver las primeras n filas (default 5).

```python
df.head()
df.head(10)
```

---

## I

### iloc

Indexador basado en posición entera (integer location).

```python
df.iloc[0]        # Primera fila
df.iloc[0:3, 1:3] # Filas 0-2, columnas 1-2
df.iloc[[0, 2, 4]] # Filas específicas
```

### Index

Etiquetas de las filas de un DataFrame o Series.

```python
df.index
df.set_index('col')
df.reset_index()
```

### info()

Método que muestra información resumida del DataFrame.

```python
df.info()
```

### Inner Join

Tipo de merge que solo incluye filas con coincidencias en ambos DataFrames.

```python
pd.merge(df1, df2, on='col', how='inner')
```

### isna() / isnull()

Método para detectar valores faltantes.

```python
df.isna()
df.isna().sum()
```

### isin()

Método para filtrar valores que están en una lista.

```python
df[df['col'].isin(['a', 'b', 'c'])]
```

---

## L

### Left Join

Tipo de merge que incluye todas las filas del DataFrame izquierdo.

```python
pd.merge(df1, df2, on='col', how='left')
```

### loc

Indexador basado en etiquetas (labels).

```python
df.loc[0]              # Fila con índice 0
df.loc[0:2, 'col']     # Filas 0-2, columna 'col'
df.loc[df['a'] > 1]    # Filtro booleano
```

---

## M

### map()

Método de Series para mapear valores usando diccionario o función.

```python
df['col'].map({'a': 1, 'b': 2})
df['col'].map(lambda x: x.upper())
```

### melt()

Función para convertir columnas a filas (unpivot).

```python
pd.melt(df, id_vars=['id'], value_vars=['col1', 'col2'])
```

### merge()

Función para combinar DataFrames (similar a JOIN en SQL).

```python
pd.merge(df1, df2, on='col')
pd.merge(df1, df2, left_on='a', right_on='b', how='left')
```

### Missing Value

Valor faltante representado como NaN (Not a Number) o None.

---

## N

### NaN

Not a Number - representa valores faltantes en Pandas.

```python
import numpy as np
np.nan
```

### nunique()

Método que cuenta valores únicos (excluyendo NaN).

```python
df['col'].nunique()
```

---

## O

### Outer Join

Tipo de merge que incluye todas las filas de ambos DataFrames.

```python
pd.merge(df1, df2, on='col', how='outer')
```

---

## P

### pivot()

Método para reorganizar datos (sin agregación).

```python
df.pivot(index='fecha', columns='producto', values='ventas')
```

### pivot_table()

Función para crear tablas dinámicas con agregación.

```python
pd.pivot_table(df, values='ventas', index='region',
               columns='producto', aggfunc='sum')
```

---

## Q

### query()

Método para filtrar con sintaxis similar a SQL.

```python
df.query('edad > 30')
df.query('edad > @min_edad and ciudad == "Madrid"')
```

---

## R

### read_csv()

Función para leer archivos CSV.

```python
df = pd.read_csv('archivo.csv')
df = pd.read_csv('archivo.csv', sep=';', encoding='utf-8')
```

### rename()

Método para renombrar columnas o índice.

```python
df.rename(columns={'old': 'new'})
```

### replace()

Método para reemplazar valores.

```python
df['col'].replace('old', 'new')
df['col'].replace({'a': 1, 'b': 2})
```

### reset_index()

Método para resetear el índice a numérico secuencial.

```python
df.reset_index(drop=True)
```

### Right Join

Tipo de merge que incluye todas las filas del DataFrame derecho.

```python
pd.merge(df1, df2, on='col', how='right')
```

---

## S

### Series

Estructura de datos 1D con índice. Es una columna de un DataFrame.

```python
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
```

### set_index()

Método para establecer una columna como índice.

```python
df.set_index('col')
```

### shape

Atributo que retorna las dimensiones (filas, columnas).

```python
df.shape  # (100, 5)
```

### sort_values()

Método para ordenar por valores de columnas.

```python
df.sort_values('col')
df.sort_values(['col1', 'col2'], ascending=[True, False])
```

### str

Accessor para métodos de string en Series.

```python
df['col'].str.upper()
df['col'].str.contains('texto')
df['col'].str.strip()
```

---

## T

### tail()

Método para ver las últimas n filas.

```python
df.tail()
df.tail(10)
```

### to_csv()

Método para guardar DataFrame como CSV.

```python
df.to_csv('archivo.csv', index=False)
```

### to_datetime()

Función para convertir a tipo datetime.

```python
pd.to_datetime(df['fecha'])
pd.to_datetime(df['fecha'], format='%d/%m/%Y')
```

### to_numeric()

Función para convertir a tipo numérico.

```python
pd.to_numeric(df['col'], errors='coerce')
```

### transform()

Método de groupby que retorna resultado del mismo tamaño.

```python
df['media_grupo'] = df.groupby('col')['valor'].transform('mean')
```

---

## U

### unique()

Método que retorna valores únicos.

```python
df['col'].unique()
```

### unstack()

Método para convertir índice multinivel a columnas.

```python
df.groupby(['a', 'b'])['c'].sum().unstack()
```

---

## V

### value_counts()

Método que cuenta frecuencia de valores únicos.

```python
df['col'].value_counts()
df['col'].value_counts(normalize=True)  # Porcentajes
```

### values

Atributo que retorna los datos como array NumPy.

```python
df.values
df['col'].values
```

---

## W

### where()

Método para reemplazar valores donde la condición es False.

```python
df['col'].where(df['col'] > 0, 0)  # Negativos a 0
```
