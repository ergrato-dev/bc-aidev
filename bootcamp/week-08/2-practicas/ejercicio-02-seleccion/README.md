# 🎯 Ejercicio 02: Selección y Filtrado de Datos

## 🎯 Objetivos

- Seleccionar datos con loc (por etiquetas)
- Seleccionar datos con iloc (por posición)
- Aplicar filtros booleanos
- Usar query() para filtrado SQL-like

---

## 📋 Instrucciones

Sigue los pasos en orden. Cada paso introduce un concepto nuevo.

**Abre `starter/main.py`** y descomenta el código de cada sección según avances.

---

## Paso 1: Preparar el Dataset

Trabajaremos con un dataset de empleados:

```python
df = pd.DataFrame({
    'nombre': ['Ana', 'Bob', 'Carlos', 'Diana', 'Eva'],
    'departamento': ['IT', 'Ventas', 'IT', 'RRHH', 'Ventas'],
    'salario': [55000, 48000, 62000, 45000, 51000],
    'años_exp': [5, 3, 8, 2, 4]
})
```

**Descomenta** la sección del Paso 1.

---

## Paso 2: Selección con loc (etiquetas)

`loc` selecciona por etiquetas de índice y nombres de columna:

```python
# Una fila por índice
df.loc[0]

# Rango de filas
df.loc[0:2]  # Incluye el 2

# Filas y columnas específicas
df.loc[0, 'nombre']
df.loc[0:2, ['nombre', 'salario']]
```

**Descomenta** la sección del Paso 2.

---

## Paso 3: Selección con iloc (posición)

`iloc` selecciona por posición numérica (como arrays de NumPy):

```python
# Primera fila
df.iloc[0]

# Rango de filas (excluye el final)
df.iloc[0:2]  # No incluye el 2

# Filas y columnas por posición
df.iloc[0, 1]          # Fila 0, columna 1
df.iloc[0:2, 0:2]      # Primeras 2 filas, primeras 2 columnas
df.iloc[[0, 2, 4], :]  # Filas específicas
```

**Descomenta** la sección del Paso 3.

---

## Paso 4: loc vs iloc con Índice Personalizado

La diferencia se hace más clara con índice no numérico:

```python
df_indexed = df.set_index('nombre')

# loc usa etiquetas
df_indexed.loc['Ana']

# iloc usa posiciones
df_indexed.iloc[0]
```

**Descomenta** la sección del Paso 4.

---

## Paso 5: Filtros Booleanos

Crear máscaras booleanas para filtrar:

```python
# Condición simple
mask = df['salario'] > 50000
df_filtrado = df[mask]

# Directamente
df_filtrado = df[df['salario'] > 50000]
```

**Descomenta** la sección del Paso 5.

---

## Paso 6: Condiciones Múltiples

Combinar condiciones con operadores:

```python
# AND: &
df[(df['salario'] > 50000) & (df['departamento'] == 'IT')]

# OR: |
df[(df['departamento'] == 'IT') | (df['departamento'] == 'Ventas')]

# NOT: ~
df[~(df['departamento'] == 'RRHH')]
```

**Importante**: Usar paréntesis alrededor de cada condición.

**Descomenta** la sección del Paso 6.

---

## Paso 7: Método isin()

Filtrar por valores en una lista:

```python
# Equivalente a OR múltiple
departamentos = ['IT', 'Ventas']
df[df['departamento'].isin(departamentos)]

# Negación
df[~df['departamento'].isin(['RRHH'])]
```

**Descomenta** la sección del Paso 7.

---

## Paso 8: Método query()

Sintaxis más legible, similar a SQL:

```python
# Sin & ni paréntesis
df.query('salario > 50000')
df.query('salario > 50000 and departamento == "IT"')

# Con variables
min_salario = 50000
df.query('salario > @min_salario')
```

**Descomenta** la sección del Paso 8.

---

## Paso 9: between() y str.contains()

Métodos especializados de filtrado:

```python
# Rango numérico
df[df['salario'].between(45000, 55000)]

# Texto que contiene
df[df['nombre'].str.contains('a', case=False)]
```

**Descomenta** la sección del Paso 9.

---

## ✅ Verificación

Al completar todos los pasos, deberías poder:

- [ ] Usar loc para seleccionar por etiquetas
- [ ] Usar iloc para seleccionar por posición
- [ ] Crear filtros con condiciones booleanas
- [ ] Combinar múltiples condiciones
- [ ] Usar query() para filtrado legible

---

## 🔗 Navegación

| Anterior                                               | Siguiente                                                      |
| ------------------------------------------------------ | -------------------------------------------------------------- |
| [← Ejercicio 01](../ejercicio-01-dataframes/README.md) | [Ejercicio 03: Limpieza →](../ejercicio-03-limpieza/README.md) |
