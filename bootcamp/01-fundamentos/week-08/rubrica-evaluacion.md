# 📋 Rúbrica de Evaluación - Semana 08

## 📊 Pandas para Manipulación de Datos

---

## 🎯 Criterios de Evaluación

### Distribución de Pesos

| Tipo de Evidencia | Peso | Descripción                      |
| ----------------- | ---- | -------------------------------- |
| 🧠 Conocimiento   | 30%  | Comprensión teórica de Pandas    |
| 💪 Desempeño      | 40%  | Ejercicios prácticos completados |
| 📦 Producto       | 30%  | Proyecto de Análisis de Ventas   |

---

## 🧠 Conocimiento (30%)

### Conceptos Evaluados

| Concepto                 | Peso | Indicadores                                       |
| ------------------------ | ---- | ------------------------------------------------- |
| Series y DataFrames      | 8%   | Diferencia entre estructuras, creación, atributos |
| Selección de datos       | 8%   | loc vs iloc, filtros booleanos, slicing           |
| Limpieza de datos        | 7%   | Missing values, duplicados, conversión de tipos   |
| Agrupación y combinación | 7%   | groupby, merge, concat, agregaciones              |

### Niveles de Desempeño

| Nivel        | Rango   | Descripción                            |
| ------------ | ------- | -------------------------------------- |
| Excelente    | 90-100% | Explica conceptos con ejemplos propios |
| Bueno        | 70-89%  | Comprende y aplica correctamente       |
| Suficiente   | 50-69%  | Conoce conceptos básicos               |
| Insuficiente | <50%    | No demuestra comprensión               |

---

## 💪 Desempeño (40%)

### Ejercicios Prácticos

| Ejercicio       | Peso | Criterios de Éxito                           |
| --------------- | ---- | -------------------------------------------- |
| 01 - DataFrames | 10%  | Crea DataFrames, usa atributos, lee archivos |
| 02 - Selección  | 10%  | Usa loc/iloc correctamente, aplica filtros   |
| 03 - Limpieza   | 10%  | Maneja NaN, duplicados, convierte tipos      |
| 04 - Agrupación | 10%  | Usa groupby, aplica agregaciones múltiples   |

### Rúbrica por Ejercicio

| Criterio    | Excelente (100%)   | Bueno (75%)     | Suficiente (50%)      | Insuficiente (25%) |
| ----------- | ------------------ | --------------- | --------------------- | ------------------ |
| Completitud | Todos los pasos    | 80% pasos       | 60% pasos             | <60% pasos         |
| Correctitud | Sin errores        | Errores menores | Algunos errores       | Muchos errores     |
| Código      | Limpio y eficiente | Funcional       | Funciona parcialmente | No funciona        |

---

## 📦 Producto (30%)

### Proyecto: Análisis de Ventas

#### Descripción

Analizar un dataset de ventas para extraer insights de negocio, incluyendo:

- Limpieza de datos
- Análisis exploratorio
- Agregaciones por categoría/fecha
- Reporte con hallazgos

#### Criterios de Evaluación

| Criterio              | Peso | Descripción                                      |
| --------------------- | ---- | ------------------------------------------------ |
| Carga de datos        | 5%   | Lee correctamente el dataset                     |
| Limpieza              | 10%  | Maneja missing values, duplicados, tipos         |
| Análisis exploratorio | 5%   | Usa describe(), info(), visualiza distribuciones |
| Agregaciones          | 5%   | Calcula métricas por categoría, fecha, región    |
| Reporte               | 5%   | Documenta hallazgos y conclusiones               |

#### Niveles de Desempeño

| Nivel        | Puntos | Descripción                                             |
| ------------ | ------ | ------------------------------------------------------- |
| Excelente    | 90-100 | Análisis completo, insights valiosos, código optimizado |
| Bueno        | 70-89  | Análisis correcto, hallazgos claros                     |
| Suficiente   | 50-69  | Análisis básico funcional                               |
| Insuficiente | <50    | Análisis incompleto o incorrecto                        |

---

## 📝 Checklist de Entrega

### Ejercicios

- [ ] ejercicio-01-dataframes completado
- [ ] ejercicio-02-seleccion completado
- [ ] ejercicio-03-limpieza completado
- [ ] ejercicio-04-agrupacion completado

### Proyecto

- [ ] Dataset cargado correctamente
- [ ] Datos limpios (sin NaN críticos, sin duplicados)
- [ ] Al menos 5 agregaciones calculadas
- [ ] Reporte con hallazgos documentado
- [ ] Código ejecutable sin errores

---

## 🎯 Competencias Desarrolladas

| Competencia                     | Nivel Esperado    |
| ------------------------------- | ----------------- |
| Manipulación de datos tabulares | Intermedio        |
| Limpieza de datos               | Básico-Intermedio |
| Análisis exploratorio           | Básico            |
| Pensamiento analítico           | Básico            |

---

## 📊 Escala de Calificación Final

| Calificación | Rango   | Descripción   |
| ------------ | ------- | ------------- |
| A            | 90-100% | Sobresaliente |
| B            | 80-89%  | Notable       |
| C            | 70-79%  | Aprobado      |
| D            | 60-69%  | Suficiente    |
| F            | <60%    | No aprobado   |

---

## 💡 Retroalimentación

### Áreas de Mejora Comunes

1. **Confusión loc/iloc**: Recordar que `loc` usa etiquetas, `iloc` usa posiciones
2. **Missing values**: Decidir entre dropna() y fillna() según el contexto
3. **Groupby**: Entender que retorna objeto agrupado, necesita agregación
4. **Merge**: Especificar `on`, `how` para uniones correctas

### Recursos de Refuerzo

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
