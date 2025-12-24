# 🔗 Ejercicio 04: Agrupación y Combinación

## 🎯 Objetivos

- Agrupar datos con groupby()
- Aplicar agregaciones múltiples
- Combinar DataFrames con merge()
- Concatenar datos con concat()

---

## 📋 Instrucciones

Sigue los pasos en orden. Cada paso introduce un concepto nuevo.

**Abre `starter/main.py`** y descomenta el código de cada sección según avances.

---

## Paso 1: Dataset de Ventas

Trabajaremos con datos de ventas:

```python
ventas = pd.DataFrame({
    'producto': ['A', 'B', 'A', 'C', 'B', 'A'],
    'region': ['Norte', 'Sur', 'Norte', 'Sur', 'Norte', 'Sur'],
    'cantidad': [10, 15, 8, 20, 12, 9],
    'precio': [100, 80, 100, 50, 80, 100]
})
```

**Descomenta** la sección del Paso 1.

---

## Paso 2: Agrupación Básica con groupby

Agrupar y calcular estadísticas:

```python
df.groupby('columna')['valor'].sum()
df.groupby('columna')['valor'].mean()
df.groupby('columna').size()  # Conteo
```

**Descomenta** la sección del Paso 2.

---

## Paso 3: Múltiples Agregaciones con agg()

Aplicar varias funciones a la vez:

```python
df.groupby('col').agg(['sum', 'mean', 'count'])
df.groupby('col').agg({
    'col1': 'sum',
    'col2': ['mean', 'max']
})
```

**Descomenta** la sección del Paso 3.

---

## Paso 4: Agrupación por Múltiples Columnas

Agrupar por dos o más columnas:

```python
df.groupby(['col1', 'col2'])['valor'].sum()
df.groupby(['col1', 'col2']).agg({...})
```

**Descomenta** la sección del Paso 4.

---

## Paso 5: Agregaciones con Nombres (Named Aggregation)

Sintaxis más clara para agregaciones:

```python
df.groupby('col').agg(
    total=('valor', 'sum'),
    promedio=('valor', 'mean'),
    conteo=('valor', 'count')
)
```

**Descomenta** la sección del Paso 5.

---

## Paso 6: Transform para Mantener Tamaño

Agregar resultado al DataFrame original:

```python
df['media_grupo'] = df.groupby('col')['valor'].transform('mean')
```

**Descomenta** la sección del Paso 6.

---

## Paso 7: Merge (Join) de DataFrames

Combinar DataFrames por columnas comunes:

```python
pd.merge(df1, df2, on='col')           # Inner join
pd.merge(df1, df2, on='col', how='left')  # Left join
```

**Descomenta** la sección del Paso 7.

---

## Paso 8: Concat para Apilar DataFrames

Concatenar verticalmente u horizontalmente:

```python
pd.concat([df1, df2])                    # Vertical
pd.concat([df1, df2], axis=1)            # Horizontal
pd.concat([df1, df2], ignore_index=True) # Resetear índice
```

**Descomenta** la sección del Paso 8.

---

## Paso 9: Pivot Table

Crear tabla resumen estilo Excel:

```python
pd.pivot_table(df, values='valor', index='fila',
               columns='columna', aggfunc='sum')
```

**Descomenta** la sección del Paso 9.

---

## ✅ Verificación

Al completar todos los pasos, deberías poder:

- [ ] Agrupar datos con groupby()
- [ ] Aplicar múltiples agregaciones
- [ ] Usar transform para mantener tamaño
- [ ] Combinar DataFrames con merge
- [ ] Concatenar DataFrames con concat

---

## 🔗 Navegación

| Anterior                                             | Índice                          |
| ---------------------------------------------------- | ------------------------------- |
| [← Ejercicio 03](../ejercicio-03-limpieza/README.md) | [📚 Semana 08](../../README.md) |
