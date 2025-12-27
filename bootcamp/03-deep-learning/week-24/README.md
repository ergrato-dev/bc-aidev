# 🔄 Semana 24: Redes Neuronales Recurrentes (RNNs)

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender la arquitectura de redes recurrentes
- ✅ Entender el problema del vanishing gradient en secuencias
- ✅ Implementar celdas LSTM y GRU desde cero
- ✅ Construir modelos para procesamiento de secuencias
- ✅ Aplicar RNNs a predicción de series temporales
- ✅ Usar capas bidireccionales y stacked RNNs

---

## 📚 Requisitos Previos

- Semana 19-21: Fundamentos de redes neuronales
- Semana 22-23: CNNs (conceptos de arquitecturas)
- Python con PyTorch
- Álgebra lineal (multiplicación de matrices)

---

## 🗂️ Estructura de la Semana

```
week-24/
├── README.md
├── rubrica-evaluacion.md
├── 0-assets/
│   ├── 01-rnn-desplegada.svg
│   ├── 02-lstm-celda.svg
│   ├── 03-gru-celda.svg
│   └── 04-bidirectional-rnn.svg
├── 1-teoria/
│   ├── 01-introduccion-rnns.md
│   ├── 02-problema-secuencias-largas.md
│   ├── 03-lstm-memoria-largo-plazo.md
│   └── 04-gru-simplificacion.md
├── 2-practicas/
│   ├── ejercicio-01-rnn-basica/
│   ├── ejercicio-02-lstm-gru/
│   └── ejercicio-03-series-temporales/
├── 3-proyecto/
│   └── predictor-temperatura/
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
| 1 | Introducción a RNNs | [01-introduccion-rnns.md](1-teoria/01-introduccion-rnns.md) | 25 min |
| 2 | Problema de Secuencias Largas | [02-problema-secuencias-largas.md](1-teoria/02-problema-secuencias-largas.md) | 20 min |
| 3 | LSTM: Memoria a Largo Plazo | [03-lstm-memoria-largo-plazo.md](1-teoria/03-lstm-memoria-largo-plazo.md) | 25 min |
| 4 | GRU: Simplificación Efectiva | [04-gru-simplificacion.md](1-teoria/04-gru-simplificacion.md) | 20 min |

### 💻 Prácticas (2.5 horas)

| # | Ejercicio | Carpeta | Duración |
|---|-----------|---------|----------|
| 1 | RNN Básica desde Cero | [ejercicio-01-rnn-basica/](2-practicas/ejercicio-01-rnn-basica/) | 45 min |
| 2 | LSTM y GRU en PyTorch | [ejercicio-02-lstm-gru/](2-practicas/ejercicio-02-lstm-gru/) | 50 min |
| 3 | Series Temporales | [ejercicio-03-series-temporales/](2-practicas/ejercicio-03-series-temporales/) | 55 min |

### 📦 Proyecto (2 horas)

| Proyecto | Descripción | Carpeta |
|----------|-------------|---------|
| Predictor de Temperatura | Predecir temperatura usando datos históricos con LSTM | [predictor-temperatura/](3-proyecto/predictor-temperatura/) |

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

## 🧠 Conceptos Clave

### RNN Vanilla
```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)
```

### LSTM (4 gates)
- **Forget gate**: Qué olvidar del estado de celda
- **Input gate**: Qué nueva información añadir
- **Cell state**: Memoria a largo plazo
- **Output gate**: Qué parte del estado mostrar

### GRU (2 gates)
- **Reset gate**: Cuánto del pasado olvidar
- **Update gate**: Balance entre pasado y presente

---

## 📌 Entregables

Al finalizar la semana debes entregar:

1. **Ejercicios completados** (2-practicas/)
   - [ ] ejercicio-01: RNN básica implementada
   - [ ] ejercicio-02: LSTM y GRU funcionando
   - [ ] ejercicio-03: Predicción de serie temporal

2. **Proyecto semanal** (3-proyecto/)
   - [ ] Predictor de temperatura con LSTM
   - [ ] MAE < 2°C en predicciones
   - [ ] Visualización de predicciones vs real

3. **Autoevaluación**
   - [ ] Completar checklist de verificación
   - [ ] Explicar diferencias entre RNN, LSTM, GRU

---

## 🔗 Navegación

| ⬅️ Anterior | 🏠 Módulo | Siguiente ➡️ |
|-------------|-----------|--------------|
| [Semana 23: CNNs II](../week-23/) | [Deep Learning](../README.md) | [Semana 25: Transformers](../week-25/) |

---

## 💡 Tips para esta Semana

> 🎯 **Consejo**: Las RNNs procesan secuencias paso a paso. Visualiza mentalmente cómo la información fluye a través del tiempo para entender mejor la arquitectura.

- **Empieza simple**: Comprende RNN vanilla antes de LSTM/GRU
- **Dibuja los diagramas**: Las puertas de LSTM son más fáciles de entender visualmente
- **Practica con secuencias cortas**: Antes de series temporales largas
- **Compara arquitecturas**: Entrena RNN, LSTM y GRU en el mismo problema

---

_Semana 24 de 36 | Módulo: Deep Learning | Bootcamp IA: Zero to Hero_
