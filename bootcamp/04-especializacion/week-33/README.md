# 👁️ Semana 33: Computer Vision Avanzado

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Entender tareas de Computer Vision: clasificación, detección, segmentación
- ✅ Implementar detección de objetos con YOLO
- ✅ Aplicar segmentación de imágenes
- ✅ Usar modelos pre-entrenados de visión
- ✅ Evaluar modelos con métricas estándar (mAP, IoU)

---

## 📚 Requisitos Previos

- Módulo 3: Deep Learning (CNNs)
- NumPy y manipulación de imágenes
- Python intermedio

---

## 🗂️ Estructura de la Semana

```
week-33/
├── README.md
├── rubrica-evaluacion.md
├── 0-assets/
│   ├── 01-cv-tasks.svg
│   ├── 02-yolo-architecture.svg
│   ├── 03-iou-metric.svg
│   └── 04-segmentation-types.svg
├── 1-teoria/
│   ├── 01-introduccion-cv.md
│   ├── 02-deteccion-objetos.md
│   ├── 03-yolo-ultralytics.md
│   └── 04-segmentacion.md
├── 2-practicas/
│   ├── ejercicio-01-clasificacion/
│   ├── ejercicio-02-deteccion-yolo/
│   └── ejercicio-03-segmentacion/
├── 3-proyecto/
│   └── detector-objetos/
├── 4-recursos/
│   └── README.md
└── 5-glosario/
    └── README.md
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| #   | Tema                     | Archivo                                                       | Duración |
| --- | ------------------------ | ------------------------------------------------------------- | -------- |
| 1   | Introducción a CV        | [01-introduccion-cv.md](1-teoria/01-introduccion-cv.md)       | 20 min   |
| 2   | Detección de Objetos     | [02-deteccion-objetos.md](1-teoria/02-deteccion-objetos.md)   | 25 min   |
| 3   | YOLO con Ultralytics     | [03-yolo-ultralytics.md](1-teoria/03-yolo-ultralytics.md)     | 25 min   |
| 4   | Segmentación de Imágenes | [04-segmentacion.md](1-teoria/04-segmentacion.md)             | 20 min   |

### 💻 Prácticas (2.5 horas)

| #   | Ejercicio              | Carpeta                                                              | Duración |
| --- | ---------------------- | -------------------------------------------------------------------- | -------- |
| 1   | Clasificación Imágenes | [ejercicio-01-clasificacion/](2-practicas/ejercicio-01-clasificacion/) | 45 min   |
| 2   | Detección con YOLO     | [ejercicio-02-deteccion-yolo/](2-practicas/ejercicio-02-deteccion-yolo/) | 60 min   |
| 3   | Segmentación           | [ejercicio-03-segmentacion/](2-practicas/ejercicio-03-segmentacion/) | 45 min   |

### 📦 Proyecto (2 horas)

| Proyecto          | Descripción                              | Carpeta                                         |
| ----------------- | ---------------------------------------- | ----------------------------------------------- |
| Detector Objetos  | Sistema de detección en tiempo real      | [detector-objetos/](3-proyecto/detector-objetos/) |

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
   - [ ] Clasificación con modelo pre-entrenado
   - [ ] Detección de objetos con YOLO
   - [ ] Segmentación de imágenes

2. **Proyecto semanal** (3-proyecto/)
   - [ ] Detector de objetos funcional
   - [ ] Capacidad de procesar imágenes y video
   - [ ] Visualización de resultados

---

## 🎯 Competencias a Desarrollar

### Técnicas
- YOLO y detección de objetos
- Segmentación semántica e instancias
- Métricas de evaluación (mAP, IoU)
- Transfer learning en visión

### Transversales
- Análisis visual de resultados
- Optimización de modelos
- Documentación técnica

---

## 🔗 Navegación

| ⬅️ Anterior                      | 🏠 Módulo                          | Siguiente ➡️                      |
| -------------------------------- | ---------------------------------- | --------------------------------- |
| [Semana 32](../week-32/README.md) | [Especialización](../README.md)    | [Semana 34](../week-34/README.md) |

---

## 💡 Tips para esta Semana

> 🎯 **Consejo**: YOLO es increíblemente fácil de usar con Ultralytics. Con pocas líneas de código puedes detectar objetos en imágenes y video.

- **GPU recomendada**: Para entrenamiento, aunque inferencia funciona en CPU
- **Datasets**: Usa COCO o tus propias imágenes
- **Práctica**: Experimenta con diferentes modelos (yolov8n, yolov8s, yolov8m)

---

## 📚 Recursos Rápidos

- 📖 [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- 🎥 [YOLO Explained](https://www.youtube.com/results?search_query=yolo+object+detection)
- 💻 [Roboflow Universe](https://universe.roboflow.com/)

---

_Semana 33 de 36 | Módulo: Especialización | Bootcamp IA: Zero to Hero_
