# 📋 Rúbrica de Evaluación - Semana 20

## 🔷 TensorFlow y Keras

---

## 📊 Distribución de Puntos

| Componente               | Porcentaje | Puntos  |
| ------------------------ | ---------- | ------- |
| 🧠 Conocimiento          | 30%        | 30      |
| 💪 Desempeño (Prácticas) | 35%        | 35      |
| 📦 Producto (Proyecto)   | 35%        | 35      |
| **Total**                | **100%**   | **100** |

---

## 🧠 Conocimiento (30 puntos)

### Conceptos Teóricos

| Criterio                                         | Puntos |
| ------------------------------------------------ | ------ |
| Explica la arquitectura de TensorFlow            | 5      |
| Diferencia entre Sequential y Functional API    | 5      |
| Conoce los tipos de capas principales            | 8      |
| Entiende el proceso de compilación y fit         | 7      |
| Sabe cuándo usar cada callback                   | 5      |

### Niveles de Desempeño - Conocimiento

| Nivel        | Rango | Descripción                                    |
| ------------ | ----- | ---------------------------------------------- |
| Insuficiente | 0-17  | No comprende la estructura básica de Keras     |
| Suficiente   | 18-21 | Puede crear modelos simples con Sequential     |
| Bueno        | 22-26 | Entiende compilación, callbacks y métricas     |
| Excelente    | 27-30 | Domina TensorFlow y puede optimizar modelos    |

---

## 💪 Desempeño - Prácticas (35 puntos)

### Ejercicio 1: Tensores Básicos (10 puntos)

| Criterio                                | Puntos |
| --------------------------------------- | ------ |
| Crea tensores de diferentes tipos       | 2      |
| Realiza operaciones matemáticas         | 3      |
| Manipula shapes y dimensiones           | 3      |
| Entiende broadcasting                   | 2      |

### Ejercicio 2: Modelo Sequential (12 puntos)

| Criterio                                | Puntos |
| --------------------------------------- | ------ |
| Construye modelo con múltiples capas    | 4      |
| Usa correctamente activaciones          | 3      |
| Configura correctamente input_shape     | 3      |
| Visualiza arquitectura con summary()    | 2      |

### Ejercicio 3: Callbacks y Checkpoints (13 puntos)

| Criterio                                | Puntos |
| --------------------------------------- | ------ |
| Implementa EarlyStopping correctamente  | 4      |
| Configura ModelCheckpoint               | 4      |
| Usa TensorBoard para visualización      | 3      |
| Guarda y carga modelos exitosamente     | 2      |

---

## 📦 Producto - Proyecto (35 puntos)

### Clasificador MNIST

| Criterio                             | Puntos |
| ------------------------------------ | ------ |
| **Arquitectura** (12 pts)            |        |
| - Input layer configurado            | 2      |
| - Hidden layers apropiadas           | 4      |
| - Output layer con softmax           | 3      |
| - Arquitectura bien justificada      | 3      |
| **Entrenamiento** (13 pts)           |        |
| - Compilación correcta               | 3      |
| - Callbacks implementados            | 4      |
| - Accuracy > 97%                     | 4      |
| - Sin overfitting significativo      | 2      |
| **Evaluación y Documentación** (10 pts) |     |
| - Métricas de evaluación             | 3      |
| - Visualización de predicciones      | 3      |
| - Código documentado                 | 2      |
| - Modelo exportado correctamente     | 2      |

---

## 🎯 Criterios de Aprobación

- ✅ Mínimo **70%** en cada componente
- ✅ Modelo MNIST con accuracy ≥ 97% en test set
- ✅ Uso correcto de al menos 2 callbacks
- ✅ Modelo guardado en formato .keras o SavedModel

---

## 📈 Métricas de Evaluación del Modelo

### Accuracy Esperada (MNIST)

| Nivel      | Accuracy | Puntos |
| ---------- | -------- | ------ |
| Mínimo     | 95-96%   | 2      |
| Esperado   | 97-98%   | 3      |
| Excelente  | >98%     | 4      |

### Indicadores de Calidad

- Loss de entrenamiento decrece consistentemente
- Validation loss no diverge significativamente
- No hay overfitting (train acc ≈ val acc)
- Tiempo de entrenamiento razonable (<5 min)

---

## 🔍 Checklist de Entrega

### Ejercicios Prácticos

- [ ] `ejercicio-01-tensores-basicos/` completado
- [ ] `ejercicio-02-modelo-sequential/` completado
- [ ] `ejercicio-03-callbacks-checkpoints/` completado

### Proyecto

- [ ] Código fuente funcional
- [ ] Modelo entrenado guardado
- [ ] Visualizaciones de entrenamiento
- [ ] Matriz de confusión
- [ ] README con instrucciones

---

## 💡 Notas Adicionales

- Se permite usar GPU si está disponible
- El entrenamiento debe ser reproducible (fijar seed)
- Los notebooks deben ejecutarse sin errores
- Se valora el código limpio y bien comentado

---

_Rúbrica Semana 20 | TensorFlow y Keras | Bootcamp IA: Zero to Hero_
