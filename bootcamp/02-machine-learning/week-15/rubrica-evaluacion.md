# 📋 Rúbrica de Evaluación - Semana 15

## 📊 Distribución de Puntos

| Tipo de Evidencia | Porcentaje | Puntos |
|-------------------|------------|--------|
| 🧠 Conocimiento   | 30%        | 30     |
| 💪 Desempeño      | 40%        | 40     |
| 📦 Producto       | 30%        | 30     |
| **Total**         | **100%**   | **100** |

---

## 🧠 Conocimiento (30 puntos)

### Conceptos de Validación (15 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Holdout vs CV | 5 | Explica cuándo usar cada estrategia |
| Tipos de CV | 5 | Conoce K-Fold, Stratified, Leave-One-Out |
| Data Leakage | 5 | Identifica y previene fugas de datos |

### Métricas de Evaluación (15 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Métricas clasificación | 5 | Calcula precision, recall, F1 correctamente |
| Métricas regresión | 5 | Interpreta MSE, MAE, R² |
| Selección de métrica | 5 | Justifica métrica según el problema |

---

## 💪 Desempeño (40 puntos)

### Ejercicio 01: Cross-Validation (10 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| K-Fold básico | 3 | Implementa cross_val_score correctamente |
| Stratified K-Fold | 3 | Aplica estratificación para clasificación |
| Análisis de resultados | 4 | Interpreta media y desviación del CV |

### Ejercicio 02: Métricas Clasificación (10 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Matriz de confusión | 3 | Genera e interpreta correctamente |
| Precision/Recall/F1 | 4 | Calcula y explica trade-offs |
| Curva ROC/AUC | 3 | Genera curva y calcula área |

### Ejercicio 03: Métricas Regresión (8 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| MSE y RMSE | 3 | Calcula e interpreta |
| MAE | 2 | Compara con MSE |
| R² | 3 | Interpreta coeficiente de determinación |

### Ejercicio 04: GridSearchCV (12 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Grid de parámetros | 3 | Define búsqueda razonable |
| Scoring múltiple | 4 | Usa refit con múltiples métricas |
| Análisis resultados | 3 | Visualiza y selecciona mejor modelo |
| RandomizedSearchCV | 2 | Compara con Grid |

---

## 📦 Producto (30 puntos)

### Proyecto: Evaluación Completa de Modelo

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Pipeline funcional | 5 | Preprocesamiento + modelo en pipeline |
| Cross-validation anidado | 6 | CV externo para evaluación, interno para optimización |
| Métricas múltiples | 5 | Reporta accuracy, precision, recall, F1, AUC |
| Visualizaciones | 5 | Curvas ROC, PR, matriz de confusión |
| Análisis bias-variance | 4 | Detecta over/underfitting |
| Documentación | 5 | Código comentado, conclusiones claras |

---

## 📈 Niveles de Desempeño

### Escala de Evaluación

| Nivel | Rango | Descripción |
|-------|-------|-------------|
| 🌟 Excelente | 90-100 | Dominio completo, análisis profundo |
| ✅ Satisfactorio | 70-89 | Cumple objetivos, comprensión sólida |
| ⚠️ En desarrollo | 50-69 | Comprensión parcial, necesita refuerzo |
| ❌ Insuficiente | 0-49 | No alcanza objetivos mínimos |

---

## ✅ Checklist de Autoevaluación

### Cross-Validation
- [ ] Sé la diferencia entre holdout y cross-validation
- [ ] Puedo implementar K-Fold y Stratified K-Fold
- [ ] Entiendo cuándo usar Leave-One-Out
- [ ] Sé prevenir data leakage en CV

### Métricas de Clasificación
- [ ] Puedo calcular precision, recall, F1 manualmente
- [ ] Interpreto correctamente una matriz de confusión
- [ ] Entiendo la curva ROC y el AUC
- [ ] Sé cuándo usar Precision-Recall vs ROC

### Métricas de Regresión
- [ ] Calculo e interpreto MSE, RMSE, MAE
- [ ] Entiendo R² y sus limitaciones
- [ ] Sé comparar modelos de regresión

### Optimización
- [ ] Uso GridSearchCV correctamente
- [ ] Conozco RandomizedSearchCV
- [ ] Implemento CV anidado para evaluación justa

---

## 🎯 Criterios de Aprobación

- **Mínimo para aprobar**: 70 puntos totales
- **Mínimo por categoría**: 50% en cada tipo de evidencia
- **Ejercicios obligatorios**: Todos deben estar completados
- **Proyecto**: Debe ejecutar sin errores

---

## 📝 Notas Adicionales

- La selección correcta de métricas según el problema es tan importante como calcularlas
- El análisis de resultados vale tanto como la implementación técnica
- Se valora la capacidad de explicar trade-offs y decisiones
