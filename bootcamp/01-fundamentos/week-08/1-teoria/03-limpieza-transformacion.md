# 🧹 Limpieza y Transformación de Datos

## 🎯 Objetivos

- Detectar y manejar valores faltantes (NaN)
- Identificar y eliminar duplicados
- Convertir tipos de datos
- Renombrar y reordenar columnas
- Aplicar transformaciones con apply y map

---

## 1. Valores Faltantes (Missing Values)

Los valores faltantes en Pandas se representan como `NaN` (Not a Number) o `None`.

### Detectar Missing Values

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'nombre': ['Ana', 'Bob', None, 'Diana'],
    'edad': [25, np.nan, 35, 28],
    'ciudad': ['Madrid', 'Barcelona', 'Valencia', None]
})

print(df)
#   nombre  edad     ciudad
# 0    Ana  25.0     Madrid
# 1    Bob   NaN  Barcelona
# 2   None  35.0   Valencia
# 3  Diana  28.0       None

# Detectar NaN (retorna DataFrame de booleanos)
print(df.isna())      # o df.isnull()

# Contar NaN por columna
print(df.isna().sum())
# nombre    1
# edad      1
# ciudad    1

# Porcentaje de NaN
print(df.isna().mean() * 100)

# ¿Hay algún NaN en cada fila?
print(df.isna().any(axis=1))

# Total de NaN en todo el DataFrame
print(df.isna().sum().sum())
```

### Eliminar Missing Values: dropna()

```python
# Eliminar filas con cualquier NaN
df_clean = df.dropna()

# Eliminar filas donde TODAS son NaN
df_clean = df.dropna(how='all')

# Eliminar si hay NaN en columnas específicas
df_clean = df.dropna(subset=['nombre', 'edad'])

# Mantener filas con al menos N valores no-NaN
df_clean = df.dropna(thresh=2)  # Al menos 2 valores válidos

# Eliminar columnas con NaN (axis=1)
df_clean = df.dropna(axis=1)
```

### Rellenar Missing Values: fillna()

```python
# Rellenar con valor constante
df['edad'] = df['edad'].fillna(0)
df['nombre'] = df['nombre'].fillna('Desconocido')

# Rellenar con estadísticas
df['edad'] = df['edad'].fillna(df['edad'].mean())     # Media
df['edad'] = df['edad'].fillna(df['edad'].median())   # Mediana
df['edad'] = df['edad'].fillna(df['edad'].mode()[0])  # Moda

# Forward fill: propagar valor anterior
df['valor'] = df['valor'].fillna(method='ffill')  # o .ffill()

# Backward fill: propagar valor siguiente
df['valor'] = df['valor'].fillna(method='bfill')  # o .bfill()

# Interpolar valores numéricos
df['valor'] = df['valor'].interpolate()

# Rellenar con diccionario (diferente valor por columna)
df = df.fillna({'nombre': 'N/A', 'edad': 0, 'ciudad': 'Sin ciudad'})
```

### Reemplazar Valores

```python
# replace(): más general que fillna()
df['ciudad'] = df['ciudad'].replace('Madrid', 'MAD')

# Múltiples reemplazos
df['ciudad'] = df['ciudad'].replace({
    'Madrid': 'MAD',
    'Barcelona': 'BCN',
    'Valencia': 'VLC'
})

# Reemplazar NaN
df['edad'] = df['edad'].replace(np.nan, 0)

# Con regex
df['nombre'] = df['nombre'].replace(r'^A.*', 'Nombre con A', regex=True)
```

---

## 2. Manejo de Duplicados

### Detectar Duplicados

```python
df = pd.DataFrame({
    'nombre': ['Ana', 'Bob', 'Ana', 'Carlos', 'Bob'],
    'edad': [25, 30, 25, 35, 30],
    'ciudad': ['Madrid', 'Barcelona', 'Madrid', 'Valencia', 'Barcelona']
})

# Filas duplicadas completas
print(df.duplicated())
# 0    False
# 1    False
# 2     True  <- duplicado de fila 0
# 3    False
# 4     True  <- duplicado de fila 1

# Ver las filas duplicadas
print(df[df.duplicated()])

# Duplicados en columnas específicas
print(df.duplicated(subset=['nombre']))

# Marcar el primero como duplicado (keep='last')
print(df.duplicated(keep='last'))

# Marcar todos los duplicados (keep=False)
print(df.duplicated(keep=False))
```

### Eliminar Duplicados

```python
# Eliminar duplicados (mantiene el primero por defecto)
df_unique = df.drop_duplicates()

# Mantener el último
df_unique = df.drop_duplicates(keep='last')

# Eliminar todos los duplicados
df_unique = df.drop_duplicates(keep=False)

# Basado en columnas específicas
df_unique = df.drop_duplicates(subset=['nombre'])

# In-place
df.drop_duplicates(inplace=True)
```

---

## 3. Conversión de Tipos de Datos

### Ver Tipos Actuales

```python
print(df.dtypes)
# nombre    object
# edad     float64
# fecha     object
# activo    object
```

### Convertir Tipos con astype()

```python
# A entero
df['edad'] = df['edad'].astype(int)

# A float
df['precio'] = df['precio'].astype(float)

# A string
df['codigo'] = df['codigo'].astype(str)

# A categoría (eficiente para pocas categorías únicas)
df['departamento'] = df['departamento'].astype('category')

# Múltiples columnas
df = df.astype({
    'edad': 'int32',
    'precio': 'float32',
    'activo': 'bool'
})
```

### Convertir a Numérico: to_numeric()

```python
# Convertir a numérico (maneja errores)
df['edad'] = pd.to_numeric(df['edad'], errors='coerce')  # NaN si falla

# errors='raise': lanza error (default)
# errors='coerce': convierte errores a NaN
# errors='ignore': deja el valor original
```

### Convertir a Fechas: to_datetime()

```python
# Convertir a datetime
df['fecha'] = pd.to_datetime(df['fecha'])

# Con formato específico
df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')

# Manejar errores
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

# Extraer componentes de fecha
df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month
df['dia'] = df['fecha'].dt.day
df['dia_semana'] = df['fecha'].dt.day_name()
```

---

## 4. Renombrar y Reordenar

### Renombrar Columnas

```python
# Con diccionario
df = df.rename(columns={
    'nombre': 'name',
    'edad': 'age',
    'ciudad': 'city'
})

# Todas las columnas
df.columns = ['col1', 'col2', 'col3']

# Transformar nombres
df.columns = df.columns.str.lower()           # Minúsculas
df.columns = df.columns.str.replace(' ', '_') # Espacios a guiones
df.columns = df.columns.str.strip()           # Quitar espacios

# Con función
df = df.rename(columns=str.upper)
```

### Renombrar Índice

```python
df = df.rename(index={0: 'primero', 1: 'segundo'})

# Resetear índice a numérico
df = df.reset_index(drop=True)

# Usar columna como índice
df = df.set_index('nombre')
```

### Reordenar Columnas

```python
# Especificar orden
df = df[['ciudad', 'nombre', 'edad']]

# Mover columna al inicio
cols = ['ciudad'] + [c for c in df.columns if c != 'ciudad']
df = df[cols]

# Ordenar columnas alfabéticamente
df = df.reindex(sorted(df.columns), axis=1)
```

### Ordenar Filas

```python
# Por una columna
df = df.sort_values('edad')

# Descendente
df = df.sort_values('edad', ascending=False)

# Por múltiples columnas
df = df.sort_values(['departamento', 'edad'], ascending=[True, False])

# Por índice
df = df.sort_index()
```

---

## 5. Transformaciones con apply()

`apply()` permite aplicar funciones a filas, columnas o elementos.

### Aplicar a Columna (Series)

```python
df = pd.DataFrame({
    'nombre': ['ana', 'bob', 'carlos'],
    'salario': [50000, 60000, 55000]
})

# Con función lambda
df['nombre'] = df['nombre'].apply(lambda x: x.upper())

# Con función definida
def calcular_impuesto(salario):
    if salario > 55000:
        return salario * 0.30
    else:
        return salario * 0.20

df['impuesto'] = df['salario'].apply(calcular_impuesto)

# Con función built-in
df['nombre'] = df['nombre'].apply(str.title)
```

### Aplicar a DataFrame

```python
# A cada columna (axis=0, default)
df_numeric = df[['salario', 'impuesto']]
print(df_numeric.apply(sum))          # Suma de cada columna
print(df_numeric.apply(np.mean))      # Media de cada columna

# A cada fila (axis=1)
df['total'] = df.apply(lambda row: row['salario'] - row['impuesto'], axis=1)

# Función que recibe toda la fila
def clasificar(row):
    if row['salario'] > 55000:
        return 'Alto'
    else:
        return 'Normal'

df['nivel'] = df.apply(clasificar, axis=1)
```

### applymap() para Elementos (Deprecado en Pandas 2.0+)

```python
# En Pandas < 2.0
df = df.applymap(lambda x: x.upper() if isinstance(x, str) else x)

# En Pandas >= 2.0, usar map() en DataFrame
df = df.map(lambda x: x.upper() if isinstance(x, str) else x)
```

---

## 6. Transformaciones con map()

`map()` es para Series, mapea valores según diccionario o función.

```python
df = pd.DataFrame({
    'grado': ['A', 'B', 'C', 'A', 'B'],
    'departamento': ['IT', 'Ventas', 'IT', 'RRHH', 'Ventas']
})

# Mapear con diccionario
grado_a_puntos = {'A': 4.0, 'B': 3.0, 'C': 2.0}
df['puntos'] = df['grado'].map(grado_a_puntos)

# Mapear con función
df['grado_desc'] = df['grado'].map(lambda x: f'Grado {x}')

# Valores no encontrados -> NaN
# Para evitarlo, usar .map().fillna()
```

---

## 7. Operaciones con Strings

Pandas tiene métodos especiales para strings accesibles via `.str`.

```python
df = pd.DataFrame({
    'nombre': ['  Ana García  ', 'BOB SMITH', 'carlos lópez'],
    'email': ['ana@gmail.com', 'bob@yahoo.com', 'carlos@hotmail.com']
})

# Limpiar espacios
df['nombre'] = df['nombre'].str.strip()

# Cambiar caso
df['nombre'] = df['nombre'].str.title()    # Ana García
df['nombre'] = df['nombre'].str.upper()    # ANA GARCÍA
df['nombre'] = df['nombre'].str.lower()    # ana garcía

# Extraer partes
df['dominio'] = df['email'].str.split('@').str[1]

# Reemplazar
df['email'] = df['email'].str.replace('gmail', 'google')

# Contiene
df['es_gmail'] = df['email'].str.contains('gmail')

# Longitud
df['len_nombre'] = df['nombre'].str.len()

# Extraer con regex
df['usuario'] = df['email'].str.extract(r'(.+)@')
```

---

## 8. Ejemplo Completo: Limpieza de Dataset

```python
import pandas as pd
import numpy as np

# Dataset sucio
data = {
    'Nombre ': ['  Ana  ', 'BOB', None, 'diana', 'Ana'],
    'EDAD': ['25', '30', '35', 'treinta', '25'],
    'Ciudad': ['Madrid', 'barcelona', 'VALENCIA', None, 'Madrid'],
    'Salario': [50000, 60000, np.nan, 52000, 50000],
    'Fecha_Ingreso': ['2020-01-15', '2019/06/20', '2021-03-10', '2020-11-05', '2020-01-15']
}
df = pd.DataFrame(data)

print("=== DataFrame Original ===")
print(df)
print()

# 1. Renombrar columnas (limpiar espacios, lowercase)
df.columns = df.columns.str.strip().str.lower()
print(f"Columnas: {list(df.columns)}")

# 2. Limpiar nombres
df['nombre'] = df['nombre'].str.strip().str.title()

# 3. Convertir edad a numérico (errores a NaN)
df['edad'] = pd.to_numeric(df['edad'], errors='coerce')

# 4. Estandarizar ciudades
df['ciudad'] = df['ciudad'].str.title()

# 5. Rellenar missing values
df['nombre'] = df['nombre'].fillna('Desconocido')
df['edad'] = df['edad'].fillna(df['edad'].median())
df['ciudad'] = df['ciudad'].fillna('No especificada')
df['salario'] = df['salario'].fillna(df['salario'].mean())

# 6. Convertir fecha
df['fecha_ingreso'] = pd.to_datetime(df['fecha_ingreso'], errors='coerce')

# 7. Eliminar duplicados
df = df.drop_duplicates()

# 8. Resetear índice
df = df.reset_index(drop=True)

print("=== DataFrame Limpio ===")
print(df)
print()
print(df.dtypes)
```

---

## ✅ Resumen

| Tarea               | Método                                      |
| ------------------- | ------------------------------------------- |
| Detectar NaN        | `isna()`, `isnull()`                        |
| Eliminar NaN        | `dropna()`                                  |
| Rellenar NaN        | `fillna()`                                  |
| Detectar duplicados | `duplicated()`                              |
| Eliminar duplicados | `drop_duplicates()`                         |
| Convertir tipos     | `astype()`, `to_numeric()`, `to_datetime()` |
| Renombrar           | `rename()`                                  |
| Ordenar             | `sort_values()`, `sort_index()`             |
| Transformar         | `apply()`, `map()`                          |
| Strings             | `.str.method()`                             |

---

## 🔗 Navegación

| Anterior                                           | Siguiente                                                  |
| -------------------------------------------------- | ---------------------------------------------------------- |
| [← Selección y Filtrado](02-seleccion-filtrado.md) | [Agrupación y Combinación →](04-agrupacion-combinacion.md) |
