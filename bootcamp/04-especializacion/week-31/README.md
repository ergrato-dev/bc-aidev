# 🤖 Semana 31: Large Language Models (LLMs)

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender las arquitecturas de LLMs (GPT, BERT, T5)
- ✅ Dominar técnicas de prompt engineering efectivo
- ✅ Aplicar fine-tuning a modelos pre-entrenados
- ✅ Usar LoRA y PEFT para entrenamiento eficiente
- ✅ Implementar generación de texto controlada
- ✅ Evaluar y comparar diferentes LLMs

---

## 📚 Requisitos Previos

- Semana 30: Hugging Face Transformers completada
- Conocimiento de tokenizers y pipelines
- PyTorch básico
- GPU recomendada para fine-tuning

---

## 🗂️ Estructura de la Semana

```
week-31/
├── README.md                    # Este archivo
├── rubrica-evaluacion.md        # Criterios de evaluación
├── 0-assets/                    # Diagramas y recursos visuales
│   ├── 01-llm-landscape.svg
│   ├── 02-gpt-architecture.svg
│   ├── 03-prompt-engineering.svg
│   └── 04-fine-tuning.svg
├── 1-teoria/                    # Material teórico
│   ├── 01-introduccion-llms.md
│   ├── 02-arquitecturas.md
│   ├── 03-prompt-engineering.md
│   └── 04-fine-tuning.md
├── 2-practicas/                 # Ejercicios guiados
│   ├── ejercicio-01-prompts/
│   ├── ejercicio-02-generacion/
│   └── ejercicio-03-fine-tuning/
├── 3-proyecto/                  # Proyecto semanal
│   └── asistente-especializado/
├── 4-recursos/                  # Material adicional
│   └── README.md
└── 5-glosario/                  # Términos clave
    └── README.md
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| #   | Tema                  | Archivo                                                     | Duración |
| --- | --------------------- | ----------------------------------------------------------- | -------- |
| 1   | Introducción a LLMs   | [01-introduccion-llms.md](1-teoria/01-introduccion-llms.md) | 20 min   |
| 2   | Arquitecturas GPT/BERT| [02-arquitecturas.md](1-teoria/02-arquitecturas.md)         | 25 min   |
| 3   | Prompt Engineering    | [03-prompt-engineering.md](1-teoria/03-prompt-engineering.md)| 25 min   |
| 4   | Fine-tuning y PEFT    | [04-fine-tuning.md](1-teoria/04-fine-tuning.md)             | 20 min   |

### 💻 Prácticas (2.5 horas)

| #   | Ejercicio                | Carpeta                                                          | Duración |
| --- | ------------------------ | ---------------------------------------------------------------- | -------- |
| 1   | Prompt Engineering       | [ejercicio-01-prompts/](2-practicas/ejercicio-01-prompts/)       | 45 min   |
| 2   | Generación de Texto      | [ejercicio-02-generacion/](2-practicas/ejercicio-02-generacion/) | 45 min   |
| 3   | Fine-tuning con LoRA     | [ejercicio-03-fine-tuning/](2-practicas/ejercicio-03-fine-tuning/)| 60 min   |

### 📦 Proyecto (2 horas)

| Proyecto                | Descripción                                     | Carpeta                                                          |
| ----------------------- | ----------------------------------------------- | ---------------------------------------------------------------- |
| Asistente Especializado | Chatbot con personalidad usando prompt tuning   | [asistente-especializado/](3-proyecto/asistente-especializado/)  |

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
   - [ ] ejercicio-01: Prompts optimizados para diferentes tareas
   - [ ] ejercicio-02: Generación de texto con parámetros controlados
   - [ ] ejercicio-03: Modelo fine-tuned funcionando

2. **Proyecto semanal** (3-proyecto/)
   - [ ] Asistente especializado funcionando
   - [ ] Sistema de prompts documentado
   - [ ] Ejemplos de interacción

3. **Autoevaluación**
   - [ ] Completar checklist de verificación
   - [ ] Responder cuestionario de conocimientos

---

## 🎯 Competencias a Desarrollar

### Técnicas
- Diseño de prompts efectivos
- Configuración de generación de texto
- Fine-tuning eficiente con PEFT
- Evaluación de modelos generativos

### Transversales
- Pensamiento crítico para evaluar outputs
- Creatividad en diseño de prompts
- Documentación de experimentos

---

## 🔗 Navegación

| ⬅️ Anterior                     | 🏠 Módulo                                  | Siguiente ➡️                    |
| ------------------------------- | ------------------------------------------ | ------------------------------- |
| [Semana 30](../week-30/README.md) | [Especialización](../README.md)           | [Semana 32](../week-32/README.md) |

---

## 💡 Tips para esta Semana

> 🎯 **Consejo**: Los LLMs son potentes pero impredecibles. Experimenta con diferentes prompts y parámetros de generación para entender su comportamiento.

- **Itera en los prompts**: Pequeños cambios pueden tener grandes efectos
- **Documenta tus experimentos**: Guarda qué prompts funcionan y cuáles no
- **Empieza simple**: Antes de fine-tuning, optimiza tus prompts
- **Cuidado con alucinaciones**: Los LLMs pueden inventar información

---

## 📚 Recursos Rápidos

- 📖 [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- 📖 [Hugging Face PEFT](https://huggingface.co/docs/peft)
- 🎥 [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- 💻 [LangChain Prompts](https://python.langchain.com/docs/modules/model_io/prompts/)

---

_Semana 31 de 36 | Módulo: Especialización | Bootcamp IA: Zero to Hero_
