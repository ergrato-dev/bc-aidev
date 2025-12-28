# 🛡️ Semana 26: Regularización en Deep Learning

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender el problema de overfitting en redes neuronales
- ✅ Implementar Dropout y entender su efecto
- ✅ Aplicar Batch Normalization correctamente
- ✅ Usar Data Augmentation para aumentar datos
- ✅ Configurar Early Stopping y Weight Decay
- ✅ Combinar técnicas de regularización efectivamente

---

## 📚 Requisitos Previos

- Semana 25: Transformers completada
- Comprensión de redes neuronales y backpropagation
- Experiencia con PyTorch o TensorFlow

---

## 🗂️ Estructura de la Semana

```
week-26/
├── README.md
├── rubrica-evaluacion.md
├── 0-assets/
│   ├── 01-overfitting-underfitting.svg
│   ├── 02-dropout-visualization.svg
│   ├── 03-batch-normalization.svg
│   └── 04-data-augmentation.svg
├── 1-teoria/
│   ├── 01-overfitting-problema.md
│   ├── 02-dropout.md
│   ├── 03-batch-normalization.md
│   └── 04-data-augmentation.md
├── 2-practicas/
│   ├── ejercicio-01-dropout/
│   ├── ejercicio-02-batch-norm/
│   └── ejercicio-03-augmentation/
├── 3-proyecto/
│   └── clasificador-regularizado/
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
| 1 | Overfitting y Underfitting | [01-overfitting-problema.md](1-teoria/01-overfitting-problema.md) | 20 min |
| 2 | Dropout | [02-dropout.md](1-teoria/02-dropout.md) | 25 min |
| 3 | Batch Normalization | [03-batch-normalization.md](1-teoria/03-batch-normalization.md) | 25 min |
| 4 | Data Augmentation | [04-data-augmentation.md](1-teoria/04-data-augmentation.md) | 20 min |

### 💻 Prácticas (2.5 horas)

| # | Ejercicio | Carpeta | Duración |
|---|-----------|---------|----------|
| 1 | Dropout en CNNs | [ejercicio-01-dropout/](2-practicas/ejercicio-01-dropout/) | 45 min |
| 2 | Batch Normalization | [ejercicio-02-batch-norm/](2-practicas/ejercicio-02-batch-norm/) | 45 min |
| 3 | Data Augmentation | [ejercicio-03-augmentation/](2-practicas/ejercicio-03-augmentation/) | 60 min |

### 📦 Proyecto (2 horas)

| Proyecto | Descripción | Carpeta |
|----------|-------------|---------|
| Clasificador Regularizado | CNN con todas las técnicas de regularización | [clasificador-regularizado/](3-proyecto/clasificador-regularizado/) |

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

## 📌 Entregables

1. **Ejercicios completados** (2-practicas/)
   - [ ] Dropout aplicado a CNN
   - [ ] Batch Normalization implementado
   - [ ] Data Augmentation configurado

2. **Proyecto semanal** (3-proyecto/)
   - [ ] Modelo sin regularización (baseline)
   - [ ] Modelo con todas las técnicas
   - [ ] Comparación de métricas
   - [ ] Gráficas de overfitting vs regularizado

---

## 🔑 Conceptos Clave

### El Problema del Overfitting

```
Training Accuracy: 99%  →  "¡Excelente!"
Test Accuracy: 60%      →  "Houston, tenemos un problema"
```

### Técnicas de Regularización

| Técnica | Qué hace | Cuándo usar |
|---------|----------|-------------|
| **Dropout** | Apaga neuronas aleatoriamente | Capas fully-connected |
| **Batch Norm** | Normaliza activaciones | Entre capas (CNNs, MLPs) |
| **Data Augmentation** | Genera variaciones de datos | Cuando hay pocos datos |
| **Weight Decay** | Penaliza pesos grandes | Siempre (L2 regularization) |
| **Early Stopping** | Para antes de overfitting | Durante entrenamiento |

---

## 🔗 Navegación

| ⬅️ Anterior | 🏠 Módulo | Siguiente ➡️ |
|-------------|-----------|--------------|
| [Semana 25: Transformers](../week-25/README.md) | [Deep Learning](../README.md) | [Semana 27: Optimización](../week-27/README.md) |

---

## 💡 Tips para esta Semana

> 🎯 **Consejo**: La regularización es un arte. No apliques todo a la vez. Empieza con una técnica, mide su efecto, y luego añade más.

- **Dropout**: Empieza con 0.2-0.3, ajusta según validación
- **Batch Norm**: Colócalo después de la capa lineal, antes de activación
- **Augmentation**: Que las transformaciones tengan sentido para tu dominio
- **Early Stopping**: Paciencia de 5-10 épocas suele funcionar

---

_Semana 26 de 36 | Módulo: Deep Learning | Bootcamp IA: Zero to Hero_
