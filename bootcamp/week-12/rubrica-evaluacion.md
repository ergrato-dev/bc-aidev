# Rúbrica de Evaluación - Semana 12

## SVM, KNN y Naive Bayes

### 📊 Distribución de Puntos

| Categoría       | Porcentaje | Puntos  |
| --------------- | ---------- | ------- |
| Conocimiento 🧠 | 30%        | 30      |
| Desempeño 💪    | 40%        | 40      |
| Producto 📦     | 30%        | 30      |
| **Total**       | **100%**   | **100** |

---

## 🧠 Conocimiento (30 puntos)

### Teoría KNN (8 puntos)

| Criterio    | Excelente (8)                                          | Bueno (6)                          | Suficiente (4)            | Insuficiente (0-3)        |
| ----------- | ------------------------------------------------------ | ---------------------------------- | ------------------------- | ------------------------- |
| Comprensión | Explica distancias, k óptimo y curse of dimensionality | Explica distancias y elección de k | Comprensión básica de KNN | No comprende el algoritmo |

### Teoría SVM (10 puntos)

| Criterio    | Excelente (10)                                            | Bueno (7)                            | Suficiente (5)            | Insuficiente (0-4)        |
| ----------- | --------------------------------------------------------- | ------------------------------------ | ------------------------- | ------------------------- |
| Comprensión | Explica hiperplano, margen, vectores de soporte y kernels | Explica hiperplano y kernels básicos | Comprensión básica de SVM | No comprende el algoritmo |

### Teoría Naive Bayes (7 puntos)

| Criterio    | Excelente (7)                                               | Bueno (5)                   | Suficiente (3)     | Insuficiente (0-2)        |
| ----------- | ----------------------------------------------------------- | --------------------------- | ------------------ | ------------------------- |
| Comprensión | Explica teorema de Bayes, asunción de independencia y tipos | Explica Bayes y tipos de NB | Comprensión básica | No comprende el algoritmo |

### Comparación de Algoritmos (5 puntos)

| Criterio              | Excelente (5)                                           | Bueno (4)                   | Suficiente (3)    | Insuficiente (0-2)      |
| --------------------- | ------------------------------------------------------- | --------------------------- | ----------------- | ----------------------- |
| Criterio de selección | Identifica cuándo usar cada algoritmo con justificación | Conoce ventajas/desventajas | Diferencia básica | No distingue algoritmos |

---

## 💪 Desempeño (40 puntos)

### Ejercicio 01: KNN (10 puntos)

| Criterio       | Excelente (10)                                             | Bueno (7)                | Suficiente (5)        | Insuficiente (0-4) |
| -------------- | ---------------------------------------------------------- | ------------------------ | --------------------- | ------------------ |
| Implementación | KNN funcional, k óptimo encontrado, normalización aplicada | KNN funcional con k fijo | Implementación básica | No funciona        |

### Ejercicio 02: SVM (10 puntos)

| Criterio       | Excelente (10)                           | Bueno (7)            | Suficiente (5)       | Insuficiente (0-4) |
| -------------- | ---------------------------------------- | -------------------- | -------------------- | ------------------ |
| Implementación | Múltiples kernels, C y gamma optimizados | Kernel RBF funcional | Kernel linear básico | No funciona        |

### Ejercicio 03: Naive Bayes (10 puntos)

| Criterio       | Excelente (10)                                         | Bueno (7)               | Suficiente (5)    | Insuficiente (0-4) |
| -------------- | ------------------------------------------------------ | ----------------------- | ----------------- | ------------------ |
| Implementación | NB para texto con TF-IDF, múltiples variantes probadas | MultinomialNB funcional | GaussianNB básico | No funciona        |

### Ejercicio 04: Comparación (10 puntos)

| Criterio | Excelente (10)                                               | Bueno (7)                     | Suficiente (5) | Insuficiente (0-4)     |
| -------- | ------------------------------------------------------------ | ----------------------------- | -------------- | ---------------------- |
| Análisis | Comparación completa con métricas, tiempos y visualizaciones | Comparación con accuracy y F1 | Solo accuracy  | Sin comparación válida |

---

## 📦 Producto (30 puntos)

### Proyecto: Clasificación de Spam

#### Funcionalidad (12 puntos)

| Criterio       | Excelente (12)                                     | Bueno (9)                     | Suficiente (6)           | Insuficiente (0-5) |
| -------------- | -------------------------------------------------- | ----------------------------- | ------------------------ | ------------------ |
| Implementación | 3 algoritmos funcionando, accuracy ≥ 0.90 en todos | 3 algoritmos, accuracy ≥ 0.85 | 2 algoritmos funcionando | < 2 algoritmos     |

#### Preprocesamiento de Texto (6 puntos)

| Criterio | Excelente (6)                                       | Bueno (4)                | Suficiente (3)       | Insuficiente (0-2)   |
| -------- | --------------------------------------------------- | ------------------------ | -------------------- | -------------------- |
| Pipeline | TF-IDF, limpieza, stopwords, stemming/lemmatization | TF-IDF y limpieza básica | Solo CountVectorizer | Sin preprocesamiento |

#### Comparación y Análisis (7 puntos)

| Criterio | Excelente (7)                                             | Bueno (5)                  | Suficiente (3) | Insuficiente (0-2) |
| -------- | --------------------------------------------------------- | -------------------------- | -------------- | ------------------ |
| Análisis | Métricas completas, confusion matrix, análisis de errores | Métricas y visualizaciones | Solo accuracy  | Sin análisis       |

#### Documentación y Código (5 puntos)

| Criterio | Excelente (5)                                      | Bueno (4)         | Suficiente (3)   | Insuficiente (0-2) |
| -------- | -------------------------------------------------- | ----------------- | ---------------- | ------------------ |
| Calidad  | Código limpio, comentado, funciones bien definidas | Código organizado | Código funcional | Código desordenado |

---

## 📋 Criterios de Aprobación

| Requisito         | Mínimo                |
| ----------------- | --------------------- |
| Puntuación total  | ≥ 70/100              |
| Conocimiento      | ≥ 21/30               |
| Desempeño         | ≥ 28/40               |
| Producto          | ≥ 21/30               |
| Accuracy proyecto | ≥ 0.90 (mejor modelo) |

---

## 🎯 Métricas del Proyecto

### Objetivos de Rendimiento

| Algoritmo   | Accuracy Mínimo | Accuracy Objetivo |
| ----------- | --------------- | ----------------- |
| KNN         | 0.85            | 0.90              |
| SVM         | 0.88            | 0.93              |
| Naive Bayes | 0.90            | 0.95              |

### Métricas a Reportar

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Tiempo de entrenamiento
- Tiempo de predicción

---

## 📝 Entrega

| Elemento             | Formato              | Obligatorio |
| -------------------- | -------------------- | ----------- |
| Ejercicios (4)       | Python (.py)         | ✅          |
| Proyecto             | Python (.py)         | ✅          |
| Visualizaciones      | PNG/SVG              | ✅          |
| Análisis comparativo | En código o markdown | ✅          |

---

## 💡 Consejos

1. **KNN**: Siempre normalizar features antes de usar KNN
2. **SVM**: Probar diferentes kernels y usar GridSearchCV
3. **Naive Bayes**: Ideal para texto, usar TF-IDF para mejores resultados
4. **Comparación**: No solo mirar accuracy, considerar tiempo y interpretabilidad
