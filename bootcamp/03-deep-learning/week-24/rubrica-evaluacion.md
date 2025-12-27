# 📋 Rúbrica de Evaluación - Semana 24

## 🎯 Competencias a Evaluar

| Competencia | Descripción |
|-------------|-------------|
| **Técnica** | Implementación de arquitecturas recurrentes |
| **Analítica** | Comprensión del flujo de información en secuencias |
| **Práctica** | Aplicación a problemas de series temporales |

---

## 📊 Distribución de Puntos

| Componente | Porcentaje | Puntos |
|------------|------------|--------|
| Conocimiento (Teoría) | 30% | 30 |
| Desempeño (Prácticas) | 35% | 35 |
| Producto (Proyecto) | 35% | 35 |
| **Total** | **100%** | **100** |

---

## 🧠 Conocimiento (30 puntos)

### Conceptos Teóricos

| Criterio | Excelente (10) | Bueno (7) | Regular (5) | Insuficiente (0-3) |
|----------|----------------|-----------|-------------|-------------------|
| **Arquitectura RNN** | Explica completamente el flujo recurrente, estados ocultos y BPTT | Comprende la estructura básica y backpropagation | Entiende parcialmente la recurrencia | No comprende la arquitectura |
| **LSTM Gates** | Describe las 4 puertas, sus funciones y el flujo del cell state | Conoce las puertas principales y su propósito | Identifica algunas puertas | No diferencia las puertas |
| **GRU vs LSTM** | Compara detalladamente ventajas, desventajas y casos de uso | Identifica diferencias principales | Reconoce que son diferentes | No distingue entre ambas |

---

## 💪 Desempeño (35 puntos)

### Ejercicio 01: RNN Básica (10 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Implementación forward | 4 | RNNCell funcional con estado oculto |
| Loop temporal | 3 | Procesamiento correcto de secuencias |
| Backpropagation | 3 | Gradientes calculados correctamente |

### Ejercicio 02: LSTM y GRU (13 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| LSTM implementation | 5 | Las 4 puertas implementadas correctamente |
| GRU implementation | 4 | Reset y update gates funcionales |
| Comparación | 4 | Análisis de parámetros y rendimiento |

### Ejercicio 03: Series Temporales (12 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Preparación datos | 4 | Secuencias y ventanas deslizantes |
| Entrenamiento | 4 | Training loop con loss decreciente |
| Predicción | 4 | Predicciones multi-step razonables |

---

## 📦 Producto (35 puntos)

### Proyecto: Predictor de Temperatura

| Criterio | Excelente (35) | Bueno (28) | Regular (21) | Insuficiente (0-14) |
|----------|----------------|------------|--------------|---------------------|
| **Funcionalidad** | MAE < 2°C, predicciones precisas | MAE < 3°C, buenas predicciones | MAE < 5°C, predicciones aceptables | MAE > 5°C o no funciona |
| **Arquitectura** | LSTM/GRU optimizado, bidireccional o stacked | Modelo LSTM/GRU funcional | RNN básica implementada | Modelo incorrecto |
| **Evaluación** | Métricas completas, visualización clara | Métricas básicas, gráficos | Algunas métricas | Sin evaluación |
| **Código** | Limpio, documentado, modular | Organizado, algunos comentarios | Funcional pero desordenado | Difícil de seguir |

### Rúbrica Detallada del Proyecto

| Componente | Puntos | Requisito |
|------------|--------|-----------|
| Carga de datos | 5 | Dataset cargado y preprocesado |
| Ventanas temporales | 5 | Secuencias de entrada correctas |
| Modelo LSTM/GRU | 8 | Arquitectura apropiada |
| Entrenamiento | 7 | Loss convergente, sin overfitting |
| Predicción | 5 | MAE < 2°C en test |
| Visualización | 5 | Gráficos de predicción vs real |

---

## 📈 Escala de Calificación

| Puntuación | Calificación | Descripción |
|------------|--------------|-------------|
| 90-100 | A | Excelente dominio de RNNs |
| 80-89 | B | Buen manejo de secuencias |
| 70-79 | C | Comprensión adecuada |
| 60-69 | D | Necesita refuerzo |
| < 60 | F | No aprobado |

---

## ✅ Checklist de Entrega

### Ejercicios
- [ ] ejercicio-01: RNN básica implementada y funcionando
- [ ] ejercicio-02: LSTM y GRU comparados
- [ ] ejercicio-03: Serie temporal predicha

### Proyecto
- [ ] Modelo LSTM/GRU entrenado
- [ ] MAE < 2°C en conjunto de test
- [ ] Visualización de predicciones
- [ ] Código documentado

### Documentación
- [ ] Comentarios explicativos en código
- [ ] Análisis de resultados
- [ ] Comparación de arquitecturas

---

## 🎯 Criterios de Aprobación

- **Mínimo 70 puntos** totales
- **Mínimo 60%** en cada componente
- **Proyecto funcional** con MAE < 3°C
- **Todos los ejercicios** completados

---

## 📚 Recursos de Apoyo

Si tienes dificultades:

1. Revisa la teoría de [01-introduccion-rnns.md](1-teoria/01-introduccion-rnns.md)
2. Consulta los diagramas en [0-assets/](0-assets/)
3. Estudia las soluciones de ejercicios anteriores
4. Practica con secuencias sintéticas simples
