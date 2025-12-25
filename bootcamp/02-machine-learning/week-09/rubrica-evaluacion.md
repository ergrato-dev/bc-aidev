# 📋 Rúbrica de Evaluación - Semana 09

## Fundamentos de Machine Learning

---

## 📊 Distribución de Puntos

| Componente            | Porcentaje | Puntos  |
| --------------------- | ---------- | ------- |
| Conocimiento (Teoría) | 30%        | 30      |
| Desempeño (Prácticas) | 40%        | 40      |
| Producto (Proyecto)   | 30%        | 30      |
| **Total**             | **100%**   | **100** |

---

## 🧠 Conocimiento (30 puntos)

Evaluación de comprensión teórica de fundamentos de ML.

### Criterios

| Criterio                     | Excelente (10)                                                         | Bueno (7)                                | Básico (4)                 | Insuficiente (0)           |
| ---------------------------- | ---------------------------------------------------------------------- | ---------------------------------------- | -------------------------- | -------------------------- |
| **Tipos de aprendizaje**     | Explica claramente supervisado, no supervisado y refuerzo con ejemplos | Explica los 3 tipos con algunos errores  | Confunde algunos conceptos | No distingue los tipos     |
| **Conceptos ML**             | Define correctamente features, labels, training, testing               | Define la mayoría correctamente          | Define algunos con errores | No comprende los conceptos |
| **Overfitting/Underfitting** | Explica causas, consecuencias y soluciones                             | Explica el concepto pero faltan detalles | Comprensión superficial    | No comprende el concepto   |

---

## 💪 Desempeño (40 puntos)

Evaluación de ejercicios prácticos.

### Ejercicio 1: Exploración de Datasets (10 puntos)

| Criterio                                   | Puntos |
| ------------------------------------------ | ------ |
| Carga correcta de dataset                  | 2      |
| Análisis de features (tipos, distribución) | 3      |
| Identificación de variable target          | 2      |
| Detección de valores faltantes             | 3      |

### Ejercicio 2: Train/Test Split (10 puntos)

| Criterio                                  | Puntos |
| ----------------------------------------- | ------ |
| Split correcto con sklearn                | 3      |
| Proporción adecuada (70-30 o 80-20)       | 2      |
| Uso de random_state para reproducibilidad | 2      |
| Estratificación cuando corresponde        | 3      |

### Ejercicio 3: Primer Modelo (10 puntos)

| Criterio                          | Puntos |
| --------------------------------- | ------ |
| Instanciación correcta del modelo | 3      |
| Entrenamiento con fit()           | 3      |
| Predicción con predict()          | 2      |
| Código limpio y comentado         | 2      |

### Ejercicio 4: Evaluación Básica (10 puntos)

| Criterio                     | Puntos |
| ---------------------------- | ------ |
| Cálculo de accuracy          | 3      |
| Uso de score() o metrics     | 3      |
| Interpretación de resultados | 2      |
| Comparación train vs test    | 2      |

---

## 📦 Producto (30 puntos)

Proyecto: Predicción de Supervivencia Titanic

### Criterios de Evaluación

| Criterio                   | Excelente (6)                    | Bueno (4)                   | Básico (2)              | Insuficiente (0) |
| -------------------------- | -------------------------------- | --------------------------- | ----------------------- | ---------------- |
| **Carga y exploración**    | EDA completo con visualizaciones | EDA básico                  | Solo carga datos        | No funciona      |
| **Preprocesamiento**       | Maneja NaN, encoding, scaling    | Maneja mayoría de issues    | Preprocesamiento básico | No preprocesa    |
| **Modelado**               | Entrena modelo correctamente     | Entrena con errores menores | Modelo incompleto       | No entrena       |
| **Evaluación**             | Métricas múltiples, análisis     | Solo accuracy               | Evaluación incompleta   | No evalúa        |
| **Código y documentación** | Limpio, comentado, modular       | Legible pero mejorable      | Difícil de seguir       | Sin estructura   |

---

## 📝 Criterios de Aprobación

- **Mínimo para aprobar**: 70 puntos totales
- **Mínimo por componente**: 50% de cada sección
  - Conocimiento: ≥ 15 puntos
  - Desempeño: ≥ 20 puntos
  - Producto: ≥ 15 puntos

---

## 🎯 Rúbrica de Calidad de Código

| Aspecto         | Excelente                    | Aceptable              | Necesita Mejora           |
| --------------- | ---------------------------- | ---------------------- | ------------------------- |
| **Legibilidad** | Código claro, bien indentado | Mayormente legible     | Difícil de leer           |
| **Comentarios** | Explican el "por qué"        | Presentes pero básicos | Ausentes o confusos       |
| **Nombres**     | Descriptivos y consistentes  | Aceptables             | Confusos (x, temp, data1) |
| **Modularidad** | Funciones bien definidas     | Algo de estructura     | Todo en un bloque         |

---

## 📅 Fechas Importantes

- **Entrega ejercicios**: Durante la semana
- **Entrega proyecto**: Fin de semana
- **Retroalimentación**: Inicio semana siguiente

---

## 💡 Consejos para Éxito

1. **Practica con datasets reales** - Kaggle tiene muchos datasets de práctica
2. **Entiende antes de codear** - Dibuja el flujo del problema
3. **Usa random_state** - Para resultados reproducibles
4. **No ignores warnings** - Scikit-learn advierte sobre problemas comunes
5. **Documenta tu razonamiento** - Explica por qué eliges ciertas decisiones
