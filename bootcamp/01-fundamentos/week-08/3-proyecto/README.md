# 📊 Proyecto: Análisis de Ventas

## 🎯 Descripción

En este proyecto analizarás un dataset de ventas de una empresa ficticia. Aplicarás todas las técnicas de Pandas aprendidas durante la semana para limpiar, explorar, transformar y analizar los datos.

---

## 📋 Objetivos

- Cargar y explorar un dataset de ventas
- Limpiar datos (missing values, duplicados, tipos)
- Analizar ventas por diferentes dimensiones
- Crear reportes con agrupaciones y pivots
- Generar insights de negocio

---

## 📁 Estructura del Proyecto

```
3-proyecto/
├── README.md           # Este archivo
├── data/
│   └── ventas.csv      # Dataset de ventas (se genera automáticamente)
├── starter/
│   └── main.py         # Código inicial con TODOs
└── solution/
    └── main.py         # Solución completa (referencia)
```

---

## 📊 Dataset

El dataset `ventas.csv` contiene registros de ventas con las siguientes columnas:

| Columna         | Descripción                                  |
| --------------- | -------------------------------------------- |
| fecha           | Fecha de la venta                            |
| producto        | Código del producto (A, B, C, D)             |
| categoria       | Categoría del producto                       |
| region          | Región de la venta (Norte, Sur, Este, Oeste) |
| vendedor        | Nombre del vendedor                          |
| cantidad        | Unidades vendidas                            |
| precio_unitario | Precio por unidad                            |
| descuento       | Porcentaje de descuento aplicado             |

---

## 🎯 Tareas a Completar

### 1. Carga y Exploración (20%)

- [ ] Cargar el dataset
- [ ] Mostrar información básica (shape, dtypes, head)
- [ ] Identificar valores faltantes

### 2. Limpieza de Datos (20%)

- [ ] Manejar valores faltantes
- [ ] Eliminar duplicados
- [ ] Convertir tipos de datos
- [ ] Crear columna de total de venta

### 3. Análisis por Dimensiones (30%)

- [ ] Ventas totales por producto
- [ ] Ventas por región
- [ ] Rendimiento por vendedor
- [ ] Análisis temporal (por mes)

### 4. Reportes y Pivots (20%)

- [ ] Pivot table: Ventas por región y producto
- [ ] Top 5 productos más vendidos
- [ ] Vendedor del mes

### 5. Insights (10%)

- [ ] Identificar patrones
- [ ] Conclusiones del análisis

---

## 💡 Hints

- Usa `pd.to_datetime()` para convertir fechas
- Recuerda que `total = cantidad * precio_unitario * (1 - descuento)`
- Usa `groupby()` con `agg()` para análisis por dimensiones
- `pivot_table()` es útil para reportes cruzados

---

## ✅ Criterios de Evaluación

| Criterio                          | Puntos  |
| --------------------------------- | ------- |
| Código funcional sin errores      | 30      |
| Limpieza de datos correcta        | 20      |
| Análisis completo por dimensiones | 25      |
| Reportes y pivots correctos       | 15      |
| Código limpio y comentado         | 10      |
| **Total**                         | **100** |

---

## 🚀 Cómo Ejecutar

```bash
cd bootcamp/week-08/3-proyecto/starter
python main.py
```

---

## 🔗 Navegación

| Prácticas                          | Recursos                        |
| ---------------------------------- | ------------------------------- |
| [← Ejercicios](../../2-practicas/) | [Recursos →](../../4-recursos/) |
