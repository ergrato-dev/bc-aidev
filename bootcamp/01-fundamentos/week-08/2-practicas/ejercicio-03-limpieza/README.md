# 🧹 Ejercicio 03: Limpieza de Datos

## 🎯 Objetivos

- Detectar y manejar valores faltantes (NaN)
- Identificar y eliminar duplicados
- Convertir tipos de datos
- Aplicar transformaciones con apply()

---

## 📋 Instrucciones

Sigue los pasos en orden. Cada paso introduce un concepto nuevo.

**Abre `starter/main.py`** y descomenta el código de cada sección según avances.

---

## Paso 1: Dataset con Datos Sucios

Trabajaremos con un dataset que tiene problemas comunes:

```python
df = pd.DataFrame({
    'nombre': ['  Ana  ', 'BOB', None, 'diana', 'Ana'],
    'edad': ['25', '30', '35', 'treinta', '25'],
    'salario': [50000, 60000, np.nan, 52000, 50000]
})
```

**Descomenta** la sección del Paso 1.

---

## Paso 2: Detectar Missing Values

Usar `isna()` para encontrar valores faltantes:

```python
df.isna()           # DataFrame booleano
df.isna().sum()     # Conteo por columna
df.isna().any()     # ¿Hay NaN en cada columna?
```

**Descomenta** la sección del Paso 2.

---

## Paso 3: Eliminar Missing Values (dropna)

Eliminar filas o columnas con NaN:

```python
df.dropna()                    # Elimina filas con cualquier NaN
df.dropna(subset=['columna'])  # Solo si NaN en columnas específicas
df.dropna(how='all')           # Solo si TODA la fila es NaN
```

**Descomenta** la sección del Paso 3.

---

## Paso 4: Rellenar Missing Values (fillna)

Reemplazar NaN con valores:

```python
df['col'].fillna(0)              # Valor constante
df['col'].fillna(df['col'].mean())  # Media
df.fillna({'col1': 0, 'col2': 'N/A'})  # Por columna
```

**Descomenta** la sección del Paso 4.

---

## Paso 5: Detectar Duplicados

Encontrar filas repetidas:

```python
df.duplicated()                 # Marca duplicados
df.duplicated(subset=['col'])   # Por columnas específicas
df[df.duplicated()]             # Ver los duplicados
```

**Descomenta** la sección del Paso 5.

---

## Paso 6: Eliminar Duplicados

Remover filas duplicadas:

```python
df.drop_duplicates()                    # Mantiene el primero
df.drop_duplicates(keep='last')         # Mantiene el último
df.drop_duplicates(subset=['nombre'])   # Por columnas
```

**Descomenta** la sección del Paso 6.

---

## Paso 7: Conversión de Tipos

Cambiar tipos de datos:

```python
df['col'].astype(int)                      # A entero
pd.to_numeric(df['col'], errors='coerce')  # A numérico (NaN si falla)
pd.to_datetime(df['fecha'])                # A fecha
```

**Descomenta** la sección del Paso 7.

---

## Paso 8: Limpieza de Strings

Métodos de texto con `.str`:

```python
df['nombre'].str.strip()    # Quitar espacios
df['nombre'].str.lower()    # Minúsculas
df['nombre'].str.title()    # Capitalizar
df['nombre'].str.replace('a', 'x')  # Reemplazar
```

**Descomenta** la sección del Paso 8.

---

## Paso 9: Transformaciones con apply()

Aplicar funciones personalizadas:

```python
df['col'].apply(lambda x: x * 2)
df['col'].apply(mi_funcion)
df.apply(funcion, axis=1)  # Por fila
```

**Descomenta** la sección del Paso 9.

---

## ✅ Verificación

Al completar todos los pasos, deberías poder:

- [ ] Detectar valores faltantes con isna()
- [ ] Eliminar o rellenar NaN según convenga
- [ ] Identificar y eliminar duplicados
- [ ] Convertir tipos de datos correctamente
- [ ] Limpiar strings y aplicar transformaciones

---

## 🔗 Navegación

| Anterior                                              | Siguiente                                                          |
| ----------------------------------------------------- | ------------------------------------------------------------------ |
| [← Ejercicio 02](../ejercicio-02-seleccion/README.md) | [Ejercicio 04: Agrupación →](../ejercicio-04-agrupacion/README.md) |
