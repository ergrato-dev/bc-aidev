# 🎓 Semana 28: Proyecto Final de Deep Learning

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Integrar todas las técnicas de Deep Learning aprendidas
- ✅ Desarrollar un proyecto end-to-end de Computer Vision o NLP
- ✅ Aplicar transfer learning con modelos preentrenados
- ✅ Implementar un pipeline completo: datos → modelo → evaluación → deploy
- ✅ Documentar y presentar un proyecto de ML profesionalmente

---

## 📚 Requisitos Previos

- Semanas 19-27 completadas
- Dominio de PyTorch o TensorFlow
- Conocimiento de CNNs, RNNs, Transformers
- Técnicas de regularización y optimización

---

## 🗂️ Estructura de la Semana

```
week-28/
├── README.md                    # Este archivo
├── rubrica-evaluacion.md        # Criterios de evaluación
├── 0-assets/                    # Diagramas y recursos
├── 1-teoria/                    # Guías de proyecto
│   ├── 01-guia-proyecto-cv.md   # Guía para Computer Vision
│   └── 02-guia-proyecto-nlp.md  # Guía para NLP
├── 2-practicas/                 # Mini-ejercicios preparatorios
│   └── ejercicio-01-pipeline/   # Pipeline básico
├── 3-proyecto/                  # Proyectos finales
│   ├── opcion-a-clasificador-imagenes/  # CV: Clasificador de imágenes
│   └── opcion-b-clasificador-texto/     # NLP: Análisis de sentimiento
├── 4-recursos/                  # Material adicional
│   └── README.md
└── 5-glosario/                  # Términos clave
    └── README.md
```

---

## 📝 Contenidos

### 📖 Guías de Proyecto (0.5 horas)

| # | Tema | Archivo | Duración |
|---|------|---------|----------|
| 1 | Guía Proyecto Computer Vision | [01-guia-proyecto-cv.md](1-teoria/01-guia-proyecto-cv.md) | 15 min |
| 2 | Guía Proyecto NLP | [02-guia-proyecto-nlp.md](1-teoria/02-guia-proyecto-nlp.md) | 15 min |

### 💻 Práctica Preparatoria (1.5 horas)

| # | Ejercicio | Carpeta | Duración |
|---|-----------|---------|----------|
| 1 | Pipeline End-to-End | [ejercicio-01-pipeline/](2-practicas/ejercicio-01-pipeline/) | 90 min |

### 📦 Proyecto Final (4 horas)

**Elige UNA opción:**

| Opción | Proyecto | Carpeta | Descripción |
|--------|----------|---------|-------------|
| A | Clasificador de Imágenes | [opcion-a-clasificador-imagenes/](3-proyecto/opcion-a-clasificador-imagenes/) | Transfer learning con ResNet/EfficientNet |
| B | Clasificador de Texto | [opcion-b-clasificador-texto/](3-proyecto/opcion-b-clasificador-texto/) | Fine-tuning con BERT/DistilBERT |

---

## ⏱️ Distribución del Tiempo

```
Total: 6 horas

┌─────────────────────────────────────────────────────────┐
│  📖 Guías         │██░░░░░░░░░░░░░░░░░░░░░░│  0.5h (8%)  │
│  💻 Práctica      │██████░░░░░░░░░░░░░░░░░░│  1.5h (25%) │
│  📦 Proyecto      │████████████████░░░░░░░░│  4.0h (67%) │
└─────────────────────────────────────────────────────────┘
```

### Sugerencia de Planificación

| Día | Actividad | Tiempo |
|-----|-----------|--------|
| Día 1 | Leer guías, elegir proyecto | 0.5h |
| Día 2 | Ejercicio pipeline | 1.5h |
| Día 3 | Proyecto: datos y modelo | 2h |
| Día 4 | Proyecto: entrenamiento y evaluación | 1.5h |
| Día 5 | Documentación y entrega | 0.5h |

---

## 📌 Entregables

### Proyecto Final (elige uno)

**Opción A - Computer Vision:**
- [ ] Clasificador de imágenes funcionando
- [ ] Accuracy en test > 85%
- [ ] Modelo guardado (.pth o .h5)
- [ ] Notebook documentado
- [ ] README con instrucciones

**Opción B - NLP:**
- [ ] Clasificador de texto funcionando
- [ ] Accuracy/F1 en test > 80%
- [ ] Modelo guardado
- [ ] Notebook documentado
- [ ] README con instrucciones

### Documentación Requerida

- [ ] Descripción del problema
- [ ] Dataset utilizado y preprocesamiento
- [ ] Arquitectura del modelo (diagrama)
- [ ] Resultados y métricas
- [ ] Conclusiones y mejoras futuras

---

## 🎯 Criterios de Éxito

| Criterio | Mínimo Aceptable |
|----------|------------------|
| Accuracy/Métrica principal | > 80% (NLP) / > 85% (CV) |
| Código | Limpio, documentado, reproducible |
| Modelo | Guardado y cargable |
| Documentación | Completa y clara |

---

## 🔗 Navegación

| ⬅️ Anterior | 🏠 Módulo | Siguiente ➡️ |
|-------------|-----------|--------------|
| [Semana 27](../week-27/README.md) | [Deep Learning](../README.md) | [Especialización](../../04-especializacion/README.md) |

---

## 💡 Tips para el Proyecto

> 🎯 **Consejo**: Empieza simple y mejora iterativamente. Un modelo básico funcionando es mejor que uno complejo que no funciona.

### Para Computer Vision
- Usa **transfer learning** (ResNet, EfficientNet preentrenados)
- Aplica **data augmentation** agresivo
- Fine-tune solo las últimas capas primero

### Para NLP
- Usa modelos **preentrenados de Hugging Face**
- **DistilBERT** es más rápido que BERT completo
- Tokeniza correctamente según el modelo

### General
- **Guarda checkpoints** durante el entrenamiento
- Usa **Early Stopping** para evitar overfitting
- **Documenta** mientras desarrollas, no al final

---

## 🏆 Rúbrica Rápida

| Componente | Peso |
|------------|------|
| Funcionalidad del modelo | 35% |
| Calidad del código | 20% |
| Documentación | 20% |
| Métricas alcanzadas | 15% |
| Presentación/README | 10% |

---

_Semana 28 de 36 | Módulo: Deep Learning | Bootcamp IA: Zero to Hero_
