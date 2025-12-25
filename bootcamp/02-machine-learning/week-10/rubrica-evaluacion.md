# Rúbrica de Evaluación - Semana 10

## Regresión Lineal y Logística

### 📊 Distribución de Puntos

| Tipo de Evidencia | Porcentaje | Puntos  |
| ----------------- | ---------- | ------- |
| Conocimiento 🧠   | 30%        | 30      |
| Desempeño 💪      | 40%        | 40      |
| Producto 📦       | 30%        | 30      |
| **Total**         | **100%**   | **100** |

---

## 🧠 Conocimiento (30 puntos)

### Conceptos Teóricos

| Criterio                | Excelente (10)                                                                | Bueno (7)                                                     | Suficiente (5)                       | Insuficiente (0-4)               |
| ----------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------ | -------------------------------- |
| **Regresión Lineal**    | Explica correctamente la ecuación, interpretación de coeficientes y supuestos | Explica la ecuación y coeficientes con pequeñas imprecisiones | Comprensión básica del concepto      | No comprende el modelo           |
| **Regresión Logística** | Comprende función sigmoide, probabilidades y clasificación                    | Entiende el uso pero confunde algunos aspectos                | Conocimiento superficial             | No distingue de regresión lineal |
| **Regularización**      | Diferencia claramente Ridge, Lasso y sus efectos                              | Conoce las técnicas pero confunde aplicaciones                | Sabe que existen pero no cuándo usar | Desconoce regularización         |

---

## 💪 Desempeño (40 puntos)

### Ejercicios Prácticos

| Ejercicio                   | Puntos | Criterios de Evaluación                         |
| --------------------------- | ------ | ----------------------------------------------- |
| **01: Regresión Simple**    | 10     | Modelo entrenado, visualización, interpretación |
| **02: Regresión Múltiple**  | 10     | Múltiples features, análisis de coeficientes    |
| **03: Regresión Logística** | 10     | Clasificación binaria, probabilidades           |
| **04: Comparación**         | 10     | Comparar modelos, seleccionar el mejor          |

### Criterios por Ejercicio

| Nivel        | Puntos | Descripción                                                |
| ------------ | ------ | ---------------------------------------------------------- |
| Excelente    | 10     | Código correcto, bien comentado, métricas interpretadas    |
| Bueno        | 7-9    | Código funcional con pequeños errores, métricas calculadas |
| Suficiente   | 5-6    | Código parcialmente funcional, métricas básicas            |
| Insuficiente | 0-4    | Código no funciona o incompleto                            |

---

## 📦 Producto (30 puntos)

### Proyecto: Predicción de Precios de Casas

| Criterio             | Excelente (10)                                                     | Bueno (7)                                       | Suficiente (5)                  | Insuficiente (0-4)   |
| -------------------- | ------------------------------------------------------------------ | ----------------------------------------------- | ------------------------------- | -------------------- |
| **Preprocesamiento** | Datos limpios, features bien seleccionadas, normalización aplicada | Limpieza correcta, algunas features relevantes  | Limpieza básica, pocas features | Sin preprocesamiento |
| **Modelo**           | R² ≥ 0.8, múltiples modelos probados, mejor seleccionado           | R² ≥ 0.7, al menos 2 modelos                    | R² ≥ 0.6, un modelo             | R² < 0.6             |
| **Análisis**         | Interpretación completa de coeficientes, residuos analizados       | Coeficientes interpretados, métricas reportadas | Métricas básicas                | Sin análisis         |

### Métricas del Proyecto

| Métrica  | Mínimo Aceptable | Objetivo     | Excelente  |
| -------- | ---------------- | ------------ | ---------- |
| R² Score | 0.60             | 0.70         | ≥ 0.80     |
| RMSE     | Reportado        | Interpretado | Optimizado |
| MAE      | Reportado        | Interpretado | Optimizado |

---

## 📋 Checklist de Entrega

### Ejercicios

- [ ] Ejercicio 01 completado y funcional
- [ ] Ejercicio 02 completado y funcional
- [ ] Ejercicio 03 completado y funcional
- [ ] Ejercicio 04 completado y funcional

### Proyecto

- [ ] Código completo y ejecutable
- [ ] R² ≥ 0.70 en conjunto de test
- [ ] Coeficientes interpretados
- [ ] Visualizaciones incluidas (scatter plot, residuos)

### Documentación

- [ ] Código comentado
- [ ] Conclusiones escritas
- [ ] Glosario de términos consultado

---

## 🎯 Criterios de Aprobación

| Requisito        | Mínimo   |
| ---------------- | -------- |
| Puntuación total | ≥ 70/100 |
| Conocimiento     | ≥ 21/30  |
| Desempeño        | ≥ 28/40  |
| Producto         | ≥ 21/30  |
| Proyecto R²      | ≥ 0.70   |

---

## 📝 Retroalimentación

### Fortalezas Comunes

- Implementación correcta de sklearn
- Visualizaciones claras
- Interpretación de métricas

### Áreas de Mejora Frecuentes

- Confusión entre R² y correlación
- Olvidar escalar features en regresión múltiple
- No verificar supuestos del modelo lineal
- Confundir regresión con clasificación

---

## 🔗 Recursos de Apoyo

Si tienes dificultades, consulta:

- [Teoría de Regresión Lineal](1-teoria/01-regresion-lineal-simple.md)
- [Sklearn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [StatQuest Videos](https://www.youtube.com/user/joshstarmer)
