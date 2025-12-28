# ⚡ Semana 27: Optimización en Deep Learning

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender cómo funcionan los optimizadores modernos
- ✅ Implementar learning rate schedules efectivos
- ✅ Usar callbacks para monitorear y controlar entrenamiento
- ✅ Aplicar técnicas de inicialización de pesos
- ✅ Implementar gradient clipping para estabilidad

---

## 📚 Requisitos Previos

- Semana 26: Regularización completada
- Backpropagation y gradientes
- PyTorch básico

---

## 🗂️ Estructura de la Semana

```
week-27/
├── README.md
├── rubrica-evaluacion.md
├── 0-assets/
│   ├── 01-optimizers-comparison.svg
│   ├── 02-learning-rate-schedules.svg
│   ├── 03-gradient-flow.svg
│   └── 04-callbacks-workflow.svg
├── 1-teoria/
│   ├── 01-optimizadores.md
│   ├── 02-learning-rate-schedules.md
│   ├── 03-inicializacion-pesos.md
│   └── 04-callbacks-checkpoints.md
├── 2-practicas/
│   ├── ejercicio-01-optimizers/
│   ├── ejercicio-02-lr-schedules/
│   └── ejercicio-03-callbacks/
├── 3-proyecto/
│   └── entrenador-optimizado/
├── 4-recursos/
│   └── README.md
└── 5-glosario/
    └── README.md
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| # | Tema | Archivo | Duración |
|---|------|---------|----------|
| 1 | Optimizadores Modernos | [01-optimizadores.md](1-teoria/01-optimizadores.md) | 25 min |
| 2 | Learning Rate Schedules | [02-learning-rate-schedules.md](1-teoria/02-learning-rate-schedules.md) | 25 min |
| 3 | Inicialización de Pesos | [03-inicializacion-pesos.md](1-teoria/03-inicializacion-pesos.md) | 20 min |
| 4 | Callbacks y Checkpoints | [04-callbacks-checkpoints.md](1-teoria/04-callbacks-checkpoints.md) | 20 min |

### 💻 Prácticas (2.5 horas)

| # | Ejercicio | Carpeta | Duración |
|---|-----------|---------|----------|
| 1 | Comparar Optimizadores | [ejercicio-01-optimizers/](2-practicas/ejercicio-01-optimizers/) | 50 min |
| 2 | LR Schedules | [ejercicio-02-lr-schedules/](2-practicas/ejercicio-02-lr-schedules/) | 50 min |
| 3 | Callbacks Custom | [ejercicio-03-callbacks/](2-practicas/ejercicio-03-callbacks/) | 50 min |

### 📦 Proyecto (2 horas)

| Proyecto | Descripción | Carpeta |
|----------|-------------|---------|
| Entrenador Optimizado | Pipeline completo con mejores prácticas | [entrenador-optimizado/](3-proyecto/entrenador-optimizado/) |

---

## ⏱️ Distribución del Tiempo

```
Total: 6 horas

┌─────────────────────────────────────────────────────────┐
│  📖 Teoría      │████████░░░░░░░░░░░░░░░░│  1.5h (25%)  │
│  💻 Prácticas   │████████████████░░░░░░░░│  2.5h (42%)  │
│  📦 Proyecto    │████████████░░░░░░░░░░░░│  2.0h (33%)  │
└─────────────────────────────────────────────────────────┘
```

---

## 🔑 Conceptos Clave

| Concepto | Descripción |
|----------|-------------|
| **SGD + Momentum** | Optimizador clásico con aceleración |
| **Adam** | Adaptive moments, el más popular |
| **AdamW** | Adam con weight decay correcto |
| **StepLR** | Reduce LR cada N épocas |
| **CosineAnnealing** | LR decrece siguiendo coseno |
| **OneCycleLR** | Ciclo único, warmup + decay |
| **Gradient Clipping** | Limita magnitud de gradientes |
| **Xavier/He Init** | Inicialización inteligente de pesos |

---

## 📌 Entregables

1. **Ejercicios completados** (2-practicas/)
   - [ ] Comparación de optimizadores con métricas
   - [ ] LR schedules visualizados y comparados
   - [ ] Callbacks personalizados funcionando

2. **Proyecto** (3-proyecto/)
   - [ ] Pipeline de entrenamiento completo
   - [ ] Early stopping + checkpointing
   - [ ] LR schedule optimizado
   - [ ] Logging de métricas

---

## 🔗 Navegación

| ⬅️ Anterior | 🏠 Módulo | Siguiente ➡️ |
|-------------|-----------|--------------|
| [Semana 26](../week-26/README.md) | [Deep Learning](../README.md) | [Semana 28](../week-28/README.md) |

---

_Semana 27 de 36 | Módulo: Deep Learning | Bootcamp IA: Zero to Hero_
