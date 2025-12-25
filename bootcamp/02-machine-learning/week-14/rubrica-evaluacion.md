# 📋 Rúbrica de Evaluación - Semana 14

## Feature Engineering y Selección de Características

### 📊 Distribución de Puntos

| Tipo de Evidencia | Porcentaje | Puntos  |
| ----------------- | ---------- | ------- |
| 🧠 Conocimiento   | 30%        | 30      |
| 💪 Desempeño      | 40%        | 40      |
| 📦 Producto       | 30%        | 30      |
| **Total**         | **100%**   | **100** |

---

## 🧠 Conocimiento (30 puntos)

### Conceptos de Feature Engineering (15 puntos)

| Criterio                        | Excelente (15)                                       | Bueno (12)                              | Suficiente (9)                  | Insuficiente (0-6)               |
| ------------------------------- | ---------------------------------------------------- | --------------------------------------- | ------------------------------- | -------------------------------- |
| Comprensión de transformaciones | Explica cuándo usar cada transformación con ejemplos | Conoce las transformaciones principales | Conoce algunas transformaciones | Confusión sobre transformaciones |
| Encoding de categóricas         | Diferencia claramente OneHot, Label, Target encoding | Conoce los métodos principales          | Conoce OneHot básico            | No distingue entre métodos       |
| Data Leakage                    | Identifica y previene data leakage correctamente     | Conoce el concepto y lo aplica          | Conocimiento básico             | No comprende el concepto         |

### Métodos de Feature Selection (15 puntos)

| Criterio         | Excelente (15)                                         | Bueno (12)               | Suficiente (9)            | Insuficiente (0-6)       |
| ---------------- | ------------------------------------------------------ | ------------------------ | ------------------------- | ------------------------ |
| Filter methods   | Explica correlación, varianza, chi-square, mutual info | Conoce múltiples métodos | Conoce correlación básica | Desconoce filter methods |
| Wrapper methods  | Implementa RFE, forward/backward selection             | Conoce RFE               | Conocimiento teórico      | No conoce wrappers       |
| Embedded methods | Usa feature importance de modelos (RF, LASSO)          | Conoce el concepto       | Conocimiento básico       | No conoce embedded       |

---

## 💪 Desempeño (40 puntos)

### Ejercicio 1: Transformaciones Numéricas (10 puntos)

| Criterio                   | Excelente (10)                                 | Bueno (8)            | Suficiente (6)             | Insuficiente (0-4)         |
| -------------------------- | ---------------------------------------------- | -------------------- | -------------------------- | -------------------------- |
| StandardScaler             | Aplica correctamente y explica mean=0, std=1   | Aplica correctamente | Aplica con errores menores | No puede aplicar           |
| MinMaxScaler               | Aplica y entiende rango [0,1]                  | Aplica correctamente | Aplica con errores         | No puede aplicar           |
| Transformación Log/Box-Cox | Aplica a distribuciones sesgadas correctamente | Aplica con guía      | Conoce el concepto         | No aplica transformaciones |

### Ejercicio 2: Codificación de Categóricas (10 puntos)

| Criterio                    | Excelente (10)                               | Bueno (8)                | Suficiente (6)                 | Insuficiente (0-4)          |
| --------------------------- | -------------------------------------------- | ------------------------ | ------------------------------ | --------------------------- |
| OneHotEncoder               | Implementa correctamente evitando dummy trap | Implementa correctamente | Implementa con errores menores | No puede implementar        |
| LabelEncoder                | Usa solo para ordinales o target             | Usa correctamente        | Confunde con nominales         | Uso incorrecto              |
| Manejo de categorías nuevas | Configura handle_unknown correctamente       | Conoce el parámetro      | Conocimiento básico            | No maneja categorías nuevas |

### Ejercicio 3: Datos Faltantes (10 puntos)

| Criterio            | Excelente (10)                             | Bueno (8)                          | Suficiente (6)           | Insuficiente (0-4)          |
| ------------------- | ------------------------------------------ | ---------------------------------- | ------------------------ | --------------------------- |
| Análisis de missing | Identifica patrones (MCAR, MAR, MNAR)      | Analiza porcentajes y distribución | Cuenta valores faltantes | No analiza missing          |
| Imputación simple   | Aplica media, mediana, moda según contexto | Aplica imputación básica           | Solo usa un método       | No imputa correctamente     |
| Imputación avanzada | Implementa KNN/iterative imputer           | Conoce métodos avanzados           | Conocimiento teórico     | Desconoce métodos avanzados |

### Ejercicio 4: Feature Selection (10 puntos)

| Criterio           | Excelente (10)                       | Bueno (8)                | Suficiente (6)      | Insuficiente (0-4) |
| ------------------ | ------------------------------------ | ------------------------ | ------------------- | ------------------ |
| Variance Threshold | Elimina features con baja varianza   | Implementa correctamente | Implementa con guía | No implementa      |
| SelectKBest        | Usa con chi2, f_classif, mutual_info | Usa correctamente        | Usa con un criterio | No usa SelectKBest |
| RFE                | Implementa con cross-validation      | Implementa básico        | Conoce el concepto  | No implementa RFE  |

---

## 📦 Producto (30 puntos)

### Pipeline de Preprocesamiento

| Criterio                       | Excelente (30)                                     | Bueno (24)                     | Suficiente (18)      | Insuficiente (0-12)               |
| ------------------------------ | -------------------------------------------------- | ------------------------------ | -------------------- | --------------------------------- |
| **Estructura del Pipeline**    | Pipeline con ColumnTransformer bien organizado     | Pipeline funcional completo    | Pipeline básico      | Pipeline incompleto o no funciona |
| **Transformaciones numéricas** | Aplica scaling + transformaciones apropiadas       | Aplica scaling correcto        | Solo scaling básico  | Sin transformaciones numéricas    |
| **Encoding categóricas**       | Maneja correctamente nominales y ordinales         | Encoding correcto              | Encoding básico      | Encoding incorrecto               |
| **Feature Selection**          | Integrado en pipeline con método apropiado         | Feature selection aplicado     | Selección manual     | Sin feature selection             |
| **Prevención data leakage**    | Fit solo en train, transform en test correctamente | Pipeline previene leakage      | Conoce el concepto   | Data leakage presente             |
| **Modelo integrado**           | Pipeline end-to-end con modelo y evaluación        | Modelo entrenado correctamente | Modelo básico        | Sin modelo o errores graves       |
| **Comparación rendimiento**    | Compara métricas antes/después con análisis        | Muestra métricas comparativas  | Métricas básicas     | Sin comparación                   |
| **Documentación**              | Código documentado, decisiones justificadas        | Documentación clara            | Documentación básica | Sin documentación                 |

---

## 📝 Criterios de Aprobación

- **Mínimo para aprobar**: 70 puntos
- **Cada tipo de evidencia**: Mínimo 50% (15/30 conocimiento, 20/40 desempeño, 15/30 producto)

---

## 🎯 Indicadores de Logro

### Nivel Experto (90-100 puntos)

- Pipeline robusto con todas las transformaciones
- Prevención completa de data leakage
- Feature selection integrado y justificado
- Mejora significativa en métricas del modelo

### Nivel Avanzado (80-89 puntos)

- Pipeline completo y funcional
- Transformaciones apropiadas aplicadas
- Feature selection implementado
- Documentación clara

### Nivel Intermedio (70-79 puntos)

- Pipeline básico funcional
- Transformaciones principales aplicadas
- Conocimiento de feature selection
- Código funcional

### Nivel Básico (< 70 puntos)

- Pipeline incompleto o con errores
- Transformaciones incorrectas o faltantes
- Data leakage presente
- Requiere refuerzo

---

## 📌 Entregables Requeridos

1. **Ejercicios** (4 notebooks/scripts completados)
2. **Proyecto** (Pipeline completo con documentación)
3. **Autoevaluación** (Checklist completado)

---

## 🔗 Navegación

| ⬅️ Semana            | 🏠 Módulo                        |
| -------------------- | -------------------------------- |
| [Week 14](README.md) | [Machine Learning](../README.md) |
