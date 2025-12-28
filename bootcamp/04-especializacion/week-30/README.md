# 🤗 Semana 30: Hugging Face Transformers

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Usar la librería Hugging Face Transformers
- ✅ Aplicar pipelines para tareas comunes de NLP
- ✅ Cargar y usar modelos pre-entrenados
- ✅ Trabajar con tokenizers de Hugging Face
- ✅ Implementar clasificación, NER y generación de texto
- ✅ Usar modelos en español y multilingües

---

## 📚 Requisitos Previos

- Semana 29: NLP Fundamentos completada
- Conocimientos de tokenización y embeddings
- Python intermedio

---

## 🗂️ Estructura de la Semana

```
week-30/
├── README.md                    # Este archivo
├── rubrica-evaluacion.md        # Criterios de evaluación
├── 0-assets/                    # Diagramas y recursos visuales
├── 1-teoria/                    # Material teórico
│   ├── 01-introduccion-hf.md
│   ├── 02-pipelines.md
│   ├── 03-tokenizers.md
│   └── 04-modelos-pretrained.md
├── 2-practicas/                 # Ejercicios guiados
│   ├── ejercicio-01-pipelines/
│   ├── ejercicio-02-tokenizers/
│   └── ejercicio-03-modelos/
├── 3-proyecto/                  # Proyecto semanal
│   └── analizador-sentimientos/
├── 4-recursos/                  # Material adicional
│   └── README.md
└── 5-glosario/                  # Términos clave
    └── README.md
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| #   | Tema                      | Archivo                                                           | Duración |
| --- | ------------------------- | ----------------------------------------------------------------- | -------- |
| 1   | Introducción a Hugging Face | [01-introduccion-hf.md](1-teoria/01-introduccion-hf.md)         | 20 min   |
| 2   | Pipelines                 | [02-pipelines.md](1-teoria/02-pipelines.md)                       | 25 min   |
| 3   | Tokenizers                | [03-tokenizers.md](1-teoria/03-tokenizers.md)                     | 25 min   |
| 4   | Modelos Pre-entrenados    | [04-modelos-pretrained.md](1-teoria/04-modelos-pretrained.md)     | 20 min   |

### 💻 Prácticas (2.5 horas)

| #   | Ejercicio            | Carpeta                                                          | Duración |
| --- | -------------------- | ---------------------------------------------------------------- | -------- |
| 1   | Pipelines NLP        | [ejercicio-01-pipelines/](2-practicas/ejercicio-01-pipelines/)   | 45 min   |
| 2   | Tokenizers           | [ejercicio-02-tokenizers/](2-practicas/ejercicio-02-tokenizers/) | 45 min   |
| 3   | Modelos y Inferencia | [ejercicio-03-modelos/](2-practicas/ejercicio-03-modelos/)       | 60 min   |

### 📦 Proyecto (2 horas)

| Proyecto                 | Descripción                                           | Carpeta                                                           |
| ------------------------ | ----------------------------------------------------- | ----------------------------------------------------------------- |
| Analizador de Sentimientos | Sistema de análisis de sentimientos multilingüe | [analizador-sentimientos/](3-proyecto/analizador-sentimientos/) |

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

Al finalizar la semana debes entregar:

1. **Ejercicios completados** (2-practicas/)
   - [ ] ejercicio-01: Pipelines funcionando
   - [ ] ejercicio-02: Tokenizers implementados
   - [ ] ejercicio-03: Modelos cargados y usados

2. **Proyecto semanal** (3-proyecto/)
   - [ ] Analizador de sentimientos funcional
   - [ ] Soporte para múltiples idiomas
   - [ ] Interfaz de línea de comandos

3. **Autoevaluación**
   - [ ] Completar checklist de verificación
   - [ ] Documentar modelos utilizados

---

## 🎯 Competencias a Desarrollar

### Técnicas

- Uso de APIs de Hugging Face
- Carga de modelos pre-entrenados
- Procesamiento de texto con tokenizers modernos
- Inferencia con transformers

### Transversales

- Lectura de documentación técnica
- Selección de modelos apropiados
- Evaluación de resultados

---

## 🔗 Navegación

| ⬅️ Anterior                    | 🏠 Inicio                              | Siguiente ➡️                    |
| ----------------------------- | ------------------------------------- | ------------------------------ |
| [Semana 29](../week-29/README.md) | [Módulo 04](../README.md) | [Semana 31](../week-31/README.md) |

---

## 💡 Tips para esta Semana

> 🎯 **Consejo**: Los modelos de Hugging Face pueden ser grandes. La primera vez que uses un modelo, se descargará automáticamente. Ten paciencia y buena conexión a internet.

- **Usa modelos pequeños para pruebas**: `distilbert-base-uncased` es más rápido que `bert-base-uncased`
- **Explora el Hub**: [huggingface.co/models](https://huggingface.co/models) tiene miles de modelos
- **Caché de modelos**: Se guardan en `~/.cache/huggingface/`
- **GPU opcional**: Los ejemplos funcionan en CPU, pero GPU acelera mucho

---

## 📚 Recursos Rápidos

- 🤗 [Hugging Face Hub](https://huggingface.co/)
- 📖 [Transformers Docs](https://huggingface.co/docs/transformers)
- 🎓 [HF Course](https://huggingface.co/learn/nlp-course)
- 💬 [HF Forums](https://discuss.huggingface.co/)

---

_Semana 30 de 36 | Módulo: Especialización | Bootcamp IA: Zero to Hero_
