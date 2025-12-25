# 🎯 Selección y Filtrado de Datos

## 🎯 Objetivos

- Dominar los métodos de selección: loc, iloc, at, iat
- Aplicar filtros booleanos para seleccionar filas
- Usar el método query() para filtrado legible
- Combinar condiciones múltiples

---

## 1. Selección de Columnas

### Una Columna (retorna Series)

```python
import pandas as pd

df = pd.DataFrame({
    'nombre': ['Ana', 'Bob', 'Carlos', 'Diana'],
    'edad': [25, 30, 35, 28],
    'ciudad': ['Madrid', 'Barcelona', 'Valencia', 'Sevilla'],
    'salario': [50000, 60000, 55000, 52000]
})

# Notación de corchetes
print(df['nombre'])

# Notación de atributo (si no hay espacios ni caracteres especiales)
print(df.nombre)
```

### Múltiples Columnas (retorna DataFrame)

```python
# Lista de columnas
print(df[['nombre', 'edad']])

# Selección por patrón
print(df.filter(like='a'))     # Columnas que contienen 'a'
print(df.filter(regex='^s'))   # Columnas que empiezan con 's'
```

---

## 2. Selección con loc (por etiquetas)

`loc` selecciona por **etiquetas** de índice y columnas.

### Sintaxis

```python
df.loc[filas, columnas]
```

### Ejemplos

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40],
    'C': [100, 200, 300, 400]
}, index=['w', 'x', 'y', 'z'])

#    A   B    C
# w  1  10  100
# x  2  20  200
# y  3  30  300
# z  4  40  400

# Una fila (retorna Series)
print(df.loc['w'])
# A      1
# B     10
# C    100

# Múltiples filas
print(df.loc[['w', 'y']])
#    A   B    C
# w  1  10  100
# y  3  30  300

# Rango de filas (incluye ambos extremos)
print(df.loc['w':'y'])
#    A   B    C
# w  1  10  100
# x  2  20  200
# y  3  30  300

# Filas y columnas específicas
print(df.loc['w', 'A'])        # 1 (escalar)
print(df.loc['w', ['A', 'B']]) # Series
print(df.loc[['w', 'x'], ['A', 'B']])  # DataFrame

# Todas las filas, algunas columnas
print(df.loc[:, ['A', 'C']])

# Todas las columnas, algunas filas
print(df.loc[['w', 'z'], :])
```

---

## 3. Selección con iloc (por posición)

`iloc` selecciona por **posición numérica** (índice entero).

### Sintaxis

```python
df.iloc[filas, columnas]
```

### Ejemplos

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40],
    'C': [100, 200, 300, 400]
})

# Una fila (retorna Series)
print(df.iloc[0])       # Primera fila
print(df.iloc[-1])      # Última fila

# Múltiples filas
print(df.iloc[[0, 2]])  # Filas 0 y 2

# Rango de filas (NO incluye extremo final)
print(df.iloc[0:2])     # Filas 0 y 1

# Filas y columnas por posición
print(df.iloc[0, 0])           # 1 (escalar)
print(df.iloc[0, [0, 1]])      # Series
print(df.iloc[[0, 1], [0, 1]]) # DataFrame

# Slicing
print(df.iloc[:2, :2])   # Primeras 2 filas, primeras 2 columnas
print(df.iloc[::2, :])   # Filas alternas (0, 2)
```

### loc vs iloc

| Característica | loc                | iloc              |
| -------------- | ------------------ | ----------------- |
| Selección por  | Etiquetas          | Posición          |
| Rango incluye  | Ambos extremos     | Solo inicio       |
| Uso típico     | Índices con nombre | Índices numéricos |

```python
df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])

df.loc['a':'b']   # Filas 'a' y 'b' (incluye 'b')
df.iloc[0:2]      # Filas 0 y 1 (no incluye 2)
```

---

## 4. Acceso Rápido: at y iat

Para acceder a un **único valor**, `at` y `iat` son más rápidos.

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [10, 20, 30]
}, index=['x', 'y', 'z'])

# at: por etiqueta
print(df.at['x', 'A'])    # 1

# iat: por posición
print(df.iat[0, 0])       # 1

# Modificar valor
df.at['x', 'A'] = 100
df.iat[0, 1] = 999
```

---

## 5. Filtrado con Condiciones Booleanas

El filtrado booleano es la forma más común de seleccionar filas.

### Condición Simple

```python
df = pd.DataFrame({
    'nombre': ['Ana', 'Bob', 'Carlos', 'Diana'],
    'edad': [25, 30, 35, 28],
    'salario': [50000, 60000, 55000, 52000]
})

# Crear máscara booleana
mask = df['edad'] > 28
print(mask)
# 0    False
# 1     True
# 2     True
# 3    False

# Aplicar máscara
print(df[mask])
#   nombre  edad  salario
# 1    Bob    30    60000
# 2 Carlos    35    55000

# En una línea
print(df[df['edad'] > 28])
```

### Condiciones Múltiples

```python
# AND: usar & (con paréntesis!)
print(df[(df['edad'] > 25) & (df['salario'] > 52000)])

# OR: usar |
print(df[(df['edad'] < 26) | (df['edad'] > 32)])

# NOT: usar ~
print(df[~(df['edad'] > 30)])

# Equivalentes
print(df[(df['edad'] >= 25) & (df['edad'] <= 30)])  # Entre 25 y 30
print(df[df['edad'].between(25, 30)])               # Más legible
```

### Operadores de Comparación

```python
# Igualdad
print(df[df['nombre'] == 'Ana'])

# Diferente
print(df[df['nombre'] != 'Ana'])

# En una lista
print(df[df['nombre'].isin(['Ana', 'Bob'])])

# No en una lista
print(df[~df['nombre'].isin(['Ana', 'Bob'])])

# Contiene string (requiere .str)
print(df[df['nombre'].str.contains('a', case=False)])

# Empieza/termina con
print(df[df['nombre'].str.startswith('A')])
print(df[df['nombre'].str.endswith('s')])
```

---

## 6. Método query()

`query()` permite filtrar con sintaxis tipo SQL, más legible.

```python
df = pd.DataFrame({
    'nombre': ['Ana', 'Bob', 'Carlos', 'Diana'],
    'edad': [25, 30, 35, 28],
    'departamento': ['Ventas', 'IT', 'IT', 'Ventas'],
    'salario': [50000, 60000, 55000, 52000]
})

# Sintaxis básica
print(df.query('edad > 28'))

# Múltiples condiciones
print(df.query('edad > 25 and salario > 52000'))
print(df.query('departamento == "IT" or edad < 26'))

# Con variables externas (usar @)
edad_minima = 28
print(df.query('edad > @edad_minima'))

# Columnas con espacios (usar backticks)
df2 = df.rename(columns={'salario': 'salario anual'})
print(df2.query('`salario anual` > 52000'))

# Comparar con lista
print(df.query('nombre in ["Ana", "Bob"]'))
print(df.query('departamento == "IT"'))
```

### Comparación: Filtro Booleano vs query()

```python
# Filtro booleano (más flexible)
df[(df['edad'] > 25) & (df['salario'] > 52000) & (df['departamento'] == 'IT')]

# query() (más legible)
df.query('edad > 25 and salario > 52000 and departamento == "IT"')
```

---

## 7. Selección con where() y mask()

### where(): Mantiene valores que cumplen condición

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

# where: mantiene valores donde es True, NaN donde es False
print(df.where(df['A'] > 2))
#      A
# 0  NaN
# 1  NaN
# 2  3.0
# 3  4.0
# 4  5.0

# Con valor de reemplazo
print(df.where(df['A'] > 2, other=0))
#    A
# 0  0
# 1  0
# 2  3
# 3  4
# 4  5
```

### mask(): Opuesto a where

```python
# mask: NaN donde es True
print(df.mask(df['A'] > 2))
#      A
# 0  1.0
# 1  2.0
# 2  NaN
# 3  NaN
# 4  NaN
```

---

## 8. Modificar Valores Seleccionados

### Con loc

```python
df = pd.DataFrame({
    'nombre': ['Ana', 'Bob', 'Carlos'],
    'edad': [25, 30, 35],
    'activo': [True, True, False]
})

# Modificar una celda
df.loc[0, 'edad'] = 26

# Modificar toda una columna
df.loc[:, 'activo'] = True

# Modificar filas que cumplen condición
df.loc[df['edad'] > 28, 'categoria'] = 'Senior'
df.loc[df['edad'] <= 28, 'categoria'] = 'Junior'

print(df)
```

### Crear Columnas Condicionales

```python
# Con np.where
import numpy as np
df['nivel'] = np.where(df['edad'] > 30, 'Alto', 'Bajo')

# Con apply y lambda
df['nivel'] = df['edad'].apply(lambda x: 'Alto' if x > 30 else 'Bajo')

# Con múltiples condiciones (np.select)
conditions = [
    df['edad'] < 26,
    df['edad'].between(26, 32),
    df['edad'] > 32
]
choices = ['Junior', 'Mid', 'Senior']
df['nivel'] = np.select(conditions, choices, default='Unknown')
```

---

## 9. Ejemplo Completo

```python
import pandas as pd
import numpy as np

# Crear dataset de empleados
empleados = pd.DataFrame({
    'id': range(1, 11),
    'nombre': ['Ana', 'Bob', 'Carlos', 'Diana', 'Eva',
               'Frank', 'Grace', 'Henry', 'Iris', 'Jack'],
    'departamento': ['IT', 'Ventas', 'IT', 'RRHH', 'Ventas',
                     'IT', 'RRHH', 'Ventas', 'IT', 'RRHH'],
    'salario': [55000, 48000, 62000, 45000, 51000,
                58000, 47000, 53000, 60000, 49000],
    'años_experiencia': [3, 5, 7, 2, 4, 6, 3, 8, 5, 4]
})

print("=== Dataset Original ===")
print(empleados.head())
print()

# Selección con loc
print("=== Empleados IT (loc) ===")
print(empleados.loc[empleados['departamento'] == 'IT', ['nombre', 'salario']])
print()

# Filtros booleanos
print("=== Salario > 50000 y Experiencia > 3 ===")
filtro = (empleados['salario'] > 50000) & (empleados['años_experiencia'] > 3)
print(empleados[filtro])
print()

# Con query
print("=== Usando query() ===")
print(empleados.query('departamento == "IT" and salario > 55000'))
print()

# Agregar categoría
empleados['categoria'] = np.select(
    [empleados['años_experiencia'] <= 3,
     empleados['años_experiencia'].between(4, 6),
     empleados['años_experiencia'] > 6],
    ['Junior', 'Mid', 'Senior']
)

print("=== Con Categorías ===")
print(empleados[['nombre', 'años_experiencia', 'categoria']])
```

---

## ✅ Resumen

| Método        | Uso             | Ejemplo                             |
| ------------- | --------------- | ----------------------------------- |
| `[]`          | Columnas        | `df['col']`, `df[['col1', 'col2']]` |
| `loc`         | Por etiquetas   | `df.loc['a':'c', 'X':'Z']`          |
| `iloc`        | Por posición    | `df.iloc[0:3, 0:2]`                 |
| `at/iat`      | Valor único     | `df.at['a', 'X']`, `df.iat[0, 0]`   |
| `[]` con bool | Filtrar filas   | `df[df['col'] > 5]`                 |
| `query()`     | Filtro SQL-like | `df.query('col > 5')`               |

---

## 🔗 Navegación

| Anterior                                             | Siguiente                                                    |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| [← Introducción a Pandas](01-introduccion-pandas.md) | [Limpieza y Transformación →](03-limpieza-transformacion.md) |
