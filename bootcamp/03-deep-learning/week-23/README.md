# 🧠 Semana 23: CNNs II - ResNet, Transfer Learning, Fine-tuning

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender las conexiones residuales y por qué permiten redes más profundas
- ✅ Implementar bloques residuales (ResNet) desde cero
- ✅ Entender el concepto de Transfer Learning y sus beneficios
- ✅ Aplicar modelos preentrenados de torchvision/timm
- ✅ Realizar fine-tuning efectivo para nuevos datasets
- ✅ Elegir estrategias de congelación de capas según el problema

---

## 📚 Requisitos Previos

- ✅ Semana 22: CNNs I completada
- ✅ Convoluciones, pooling, arquitecturas básicas
- ✅ PyTorch nn.Module y entrenamiento

---

## 🗂️ Estructura de la Semana

```
week-23/
├── README.md                    # Este archivo
├── rubrica-evaluacion.md        # Criterios de evaluación
├── 0-assets/                    # Diagramas y recursos visuales
├── 1-teoria/                    # Material teórico
│   ├── 01-problema-profundidad.md
│   ├── 02-resnet-conexiones-residuales.md
│   ├── 03-transfer-learning.md
│   └── 04-fine-tuning-estrategias.md
├── 2-practicas/                 # Ejercicios guiados
│   ├── ejercicio-01-bloques-residuales/
│   ├── ejercicio-02-transfer-learning/
│   └── ejercicio-03-fine-tuning/
├── 3-proyecto/                  # Proyecto semanal
│   └── clasificador-flores/
├── 4-recursos/                  # Material adicional
│   └── README.md
└── 5-glosario/                  # Términos clave
    └── README.md
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| #   | Tema                        | Archivo                                                                        | Duración |
| --- | --------------------------- | ------------------------------------------------------------------------------ | -------- |
| 1   | El Problema de la Profundidad | [01-problema-profundidad.md](1-teoria/01-problema-profundidad.md)              | 20 min   |
| 2   | ResNet y Conexiones Residuales | [02-resnet-conexiones-residuales.md](1-teoria/02-resnet-conexiones-residuales.md) | 25 min   |
| 3   | Transfer Learning           | [03-transfer-learning.md](1-teoria/03-transfer-learning.md)                    | 25 min   |
| 4   | Fine-tuning: Estrategias    | [04-fine-tuning-estrategias.md](1-teoria/04-fine-tuning-estrategias.md)        | 20 min   |

### 💻 Prácticas (2.5 horas)

| #   | Ejercicio              | Carpeta                                                              | Duración |
| --- | ---------------------- | -------------------------------------------------------------------- | -------- |
| 1   | Bloques Residuales     | [ejercicio-01-bloques-residuales/](2-practicas/ejercicio-01-bloques-residuales/) | 45 min   |
| 2   | Transfer Learning      | [ejercicio-02-transfer-learning/](2-practicas/ejercicio-02-transfer-learning/)   | 50 min   |
| 3   | Fine-tuning Completo   | [ejercicio-03-fine-tuning/](2-practicas/ejercicio-03-fine-tuning/)               | 55 min   |

### 📦 Proyecto (2 horas)

| Proyecto             | Descripción                                      | Carpeta                                               |
| -------------------- | ------------------------------------------------ | ----------------------------------------------------- |
| Clasificador de Flores | Transfer Learning + Fine-tuning en Flowers-102   | [clasificador-flores/](3-proyecto/clasificador-flores/) |

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

### ResNet y Conexiones Residuales

```
                    ┌─────────────┐
         x ────────►│   Conv      │
                    │   BN        │
                    │   ReLU      │
                    │   Conv      │
                    │   BN        │────┐
                    └─────────────┘    │
                           │           │
                           ▼           │
                    ┌─────────────┐    │
                    │     +       │◄───┘ Skip Connection
                    └─────────────┘
                           │
                           ▼
                         ReLU
                           │
                           ▼
                        F(x) + x
```

**Idea clave**: En lugar de aprender $H(x)$, la red aprende el **residuo** $F(x) = H(x) - x$

### Transfer Learning

```
Modelo Preentrenado (ImageNet)      Tu Problema
┌──────────────────────┐           ┌──────────────────────┐
│  Features Generales  │  ────►    │  Features Generales  │
│  (bordes, texturas)  │   Reusar  │  (ya aprendidas)     │
├──────────────────────┤           ├──────────────────────┤
│  Features Específicas│  ────►    │  Features Específicas│
│  (1000 clases)       │  Reemplazar│  (tus N clases)     │
└──────────────────────┘           └──────────────────────┘
```

---

## 📌 Entregables

1. **Ejercicios completados** (2-practicas/)
   - [ ] Bloques residuales implementados
   - [ ] Transfer learning funcionando
   - [ ] Fine-tuning con resultados

2. **Proyecto semanal** (3-proyecto/)
   - [ ] Clasificador de flores con ≥85% accuracy
   - [ ] Comparativa: desde cero vs transfer learning
   - [ ] Código documentado

3. **Autoevaluación**
   - [ ] Completar checklist de verificación
   - [ ] Responder cuestionario conceptual

---

## 💡 Tips de la Semana

> 🎯 **Transfer Learning es tu superpoder**: En el 90% de problemas reales, usar un modelo preentrenado es mejor que entrenar desde cero.

- **Datos pequeños** (< 1K): Congela todo, solo entrena clasificador
- **Datos medianos** (1K-10K): Fine-tune últimas capas
- **Datos grandes** (> 10K): Fine-tune toda la red con LR bajo

---

## 🔗 Navegación

| ⬅️ Anterior                       | 🏠 Módulo                                   | Siguiente ➡️                      |
| --------------------------------- | ------------------------------------------- | --------------------------------- |
| [Semana 22](../week-22/README.md) | [Deep Learning](../README.md)               | [Semana 24](../week-24/README.md) |

---

_Semana 23 de 36 | Módulo: Deep Learning | Bootcamp IA: Zero to Hero_
