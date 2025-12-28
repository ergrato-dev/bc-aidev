# 📋 Rúbrica de Evaluación - Semana 27

## Optimización en Deep Learning

---

## 📊 Distribución de Puntos

| Componente | Puntos | Porcentaje |
|------------|--------|------------|
| Conocimiento (Teoría) | 30 | 30% |
| Desempeño (Ejercicios) | 35 | 35% |
| Producto (Proyecto) | 35 | 35% |
| **Total** | **100** | **100%** |

---

## 🧠 Conocimiento (30 puntos)

### Conceptos Evaluados

| Tema | Puntos | Criterio |
|------|--------|----------|
| Optimizadores | 8 | Diferencia SGD/Adam/AdamW |
| Learning Rate | 8 | Schedules y su impacto |
| Inicialización | 7 | Xavier vs He, cuándo usar |
| Callbacks | 7 | Propósito y tipos |

### Niveles de Desempeño

| Nivel | Puntos | Descripción |
|-------|--------|-------------|
| Excelente | 27-30 | Explica trade-offs entre optimizadores |
| Bueno | 21-26 | Conoce los principales y cuándo usarlos |
| Suficiente | 15-20 | Entiende conceptos básicos |
| Insuficiente | <15 | Confunde optimizadores o schedules |

---

## 💪 Desempeño - Ejercicios (35 puntos)

### Ejercicio 01: Comparar Optimizadores (12 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Implementación | 4 | SGD, Momentum, Adam, AdamW funcionan |
| Comparación | 4 | Gráficas de loss y accuracy |
| Análisis | 4 | Conclusiones sobre velocidad/estabilidad |

### Ejercicio 02: LR Schedules (12 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Schedules | 4 | StepLR, CosineAnnealing, OneCycle |
| Visualización | 4 | Curvas de LR por época |
| Comparación | 4 | Impacto en convergencia |

### Ejercicio 03: Callbacks (11 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| EarlyStopping | 4 | Implementado correctamente |
| ModelCheckpoint | 4 | Guarda mejor modelo |
| Custom Callback | 3 | Logger o métrica custom |

---

## 📦 Producto - Proyecto (35 puntos)

### Entrenador Optimizado

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Arquitectura | 8 | Modelo con BatchNorm, Dropout |
| Optimizer | 7 | AdamW con weight decay |
| LR Schedule | 7 | OneCycleLR o Cosine configurado |
| Callbacks | 7 | Early stopping + checkpoint |
| Resultados | 6 | Test accuracy > 80% |

### Niveles de Calidad

| Nivel | Puntos | Descripción |
|-------|--------|-------------|
| Excelente | 32-35 | Pipeline robusto, métricas logged |
| Bueno | 25-31 | Funcional con mejoras menores |
| Suficiente | 18-24 | Básico pero funciona |
| Insuficiente | <18 | No entrena o falla |

---

## ✅ Checklist de Entrega

### Ejercicios
- [ ] Ejercicio 01: Gráficas comparativas de optimizadores
- [ ] Ejercicio 02: Visualización de LR schedules
- [ ] Ejercicio 03: Callbacks implementados y funcionando

### Proyecto
- [ ] Código ejecutable sin errores
- [ ] Pipeline completo con todas las técnicas
- [ ] Gráficas de entrenamiento guardadas
- [ ] Test accuracy reportado

---

## 📅 Fecha de Entrega

- **Ejercicios**: Final de semana 27
- **Proyecto**: Final de semana 27

---

## 💡 Criterios de Aprobación

- Mínimo **70%** en cada componente
- Todos los ejercicios deben ejecutar sin errores
- Proyecto debe completar entrenamiento exitosamente
