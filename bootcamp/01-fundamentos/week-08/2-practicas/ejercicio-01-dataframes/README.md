# 📊 Ejercicio 01: Creación y Exploración de DataFrames

## 🎯 Objetivos

- Crear Series y DataFrames desde diferentes fuentes
- Leer archivos CSV
- Explorar datos con métodos básicos
- Acceder a atributos del DataFrame

---

## 📋 Instrucciones

Sigue los pasos en orden. Cada paso introduce un concepto nuevo.

**Abre `starter/main.py`** y descomenta el código de cada sección según avances.

---

## Paso 1: Importar Pandas

Pandas se importa convencionalmente como `pd`:

```python
import pandas as pd
```

**Descomenta** la sección del Paso 1 en `starter/main.py`.

---

## Paso 2: Crear una Series

Una Series es un array unidimensional con etiquetas (índice):

```python
# Desde lista
notas = pd.Series([85, 92, 78, 95])

# Con índice personalizado
notas = pd.Series([85, 92, 78, 95], index=['Ana', 'Bob', 'Carlos', 'Diana'])
```

**Descomenta** la sección del Paso 2 y observa:

- Los valores de la Series
- El índice
- El tipo de datos

---

## Paso 3: Operaciones con Series

Las Series soportan operaciones vectorizadas:

```python
# Operaciones aritméticas
notas_ajustadas = notas + 5

# Estadísticas
media = notas.mean()
maximo = notas.max()
```

**Descomenta** la sección del Paso 3 y experimenta con diferentes operaciones.

---

## Paso 4: Crear un DataFrame desde Diccionario

Un DataFrame es una tabla 2D con columnas etiquetadas:

```python
datos = {
    'nombre': ['Ana', 'Bob', 'Carlos'],
    'edad': [25, 30, 35],
    'ciudad': ['Madrid', 'Barcelona', 'Valencia']
}
df = pd.DataFrame(datos)
```

**Descomenta** la sección del Paso 4.

---

## Paso 5: Atributos del DataFrame

Explora la estructura del DataFrame:

```python
df.shape      # (filas, columnas)
df.columns    # Nombres de columnas
df.index      # Índice de filas
df.dtypes     # Tipos de datos por columna
```

**Descomenta** la sección del Paso 5.

---

## Paso 6: Leer CSV

Pandas puede leer múltiples formatos de archivo:

```python
df = pd.read_csv('archivo.csv')
```

Usaremos datos de ejemplo creados en el ejercicio.

**Descomenta** la sección del Paso 6.

---

## Paso 7: Exploración Básica

Métodos esenciales para explorar datos:

```python
df.head(5)       # Primeras 5 filas
df.tail(3)       # Últimas 3 filas
df.info()        # Resumen de estructura
df.describe()    # Estadísticas descriptivas
```

**Descomenta** la sección del Paso 7.

---

## Paso 8: Acceso a Columnas

Diferentes formas de acceder a columnas:

```python
# Como atributo (si el nombre no tiene espacios)
df.nombre

# Con corchetes (siempre funciona)
df['nombre']

# Múltiples columnas
df[['nombre', 'edad']]
```

**Descomenta** la sección del Paso 8.

---

## Paso 9: Value Counts

Contar valores únicos en una columna:

```python
df['ciudad'].value_counts()
df['ciudad'].nunique()  # Número de valores únicos
```

**Descomenta** la sección del Paso 9.

---

## ✅ Verificación

Al completar todos los pasos, deberías poder:

- [ ] Crear Series con índice personalizado
- [ ] Crear DataFrames desde diccionarios
- [ ] Leer archivos CSV
- [ ] Explorar datos con head(), info(), describe()
- [ ] Acceder a columnas de diferentes formas

---

## 🔗 Navegación

| Anterior                                             | Siguiente                                                        |
| ---------------------------------------------------- | ---------------------------------------------------------------- |
| [← Teoría](../../1-teoria/01-introduccion-pandas.md) | [Ejercicio 02: Selección →](../ejercicio-02-seleccion/README.md) |
