# 🔧 Ejercicio 01: Pipelines de Hugging Face

## 🎯 Objetivo

Aprender a usar los pipelines de Hugging Face para tareas NLP comunes.

---

## 📋 Descripción

En este ejercicio explorarás los diferentes pipelines disponibles: análisis de sentimientos, NER, question answering, generación de texto y zero-shot classification.

---

## 🔧 Pasos del Ejercicio

### Paso 1: Instalación y Setup

Asegúrate de tener las librerías instaladas:

```bash
pip install transformers torch
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

### Paso 2: Análisis de Sentimientos

El pipeline más básico para clasificar texto como positivo o negativo:

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")
```

### Paso 3: Reconocimiento de Entidades (NER)

Identificar personas, lugares y organizaciones en texto:

```python
ner = pipeline("ner", aggregation_strategy="simple")
entities = ner("Apple Inc. was founded by Steve Jobs")
```

### Paso 4: Preguntas y Respuestas

Extraer respuestas de un contexto dado:

```python
qa = pipeline("question-answering")
result = qa(question="What is Python?", context="Python is a programming language.")
```

### Paso 5: Generación de Texto

Generar texto continuando un prompt:

```python
generator = pipeline("text-generation", model="gpt2")
result = generator("The future of AI is", max_length=50)
```

### Paso 6: Zero-Shot Classification

Clasificar texto sin entrenamiento previo en esas categorías:

```python
classifier = pipeline("zero-shot-classification")
result = classifier("I need to buy groceries", candidate_labels=["shopping", "work", "travel"])
```

---

## 📁 Estructura

```
ejercicio-01-pipelines/
├── README.md
└── starter/
    └── main.py
```

---

## ▶️ Ejecución

```bash
cd bootcamp/04-especializacion/week-30/2-practicas/ejercicio-01-pipelines
python starter/main.py
```

---

## ✅ Criterios de Éxito

- [ ] Pipeline de sentiment-analysis funciona
- [ ] NER identifica entidades correctamente
- [ ] QA extrae respuestas del contexto
- [ ] Generación de texto produce output coherente
- [ ] Zero-shot clasifica sin entrenamiento previo

---

## 🔗 Recursos

- [Pipelines Documentation](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [Task Summary](https://huggingface.co/docs/transformers/task_summary)
