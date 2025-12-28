# 🎯 Ejercicio 03: Q&A sobre Documentos

## 🎯 Objetivo

Construir un pipeline RAG completo para responder preguntas sobre documentos.

---

## 📋 Descripción

En este ejercicio implementarás un sistema RAG end-to-end: cargar documentos, chunking, indexar en vector DB, buscar contexto relevante y generar respuestas.

---

## 🔧 Requisitos

```bash
pip install chromadb sentence-transformers openai python-dotenv
```

---

## 🔧 Pasos del Ejercicio

### Paso 1: Preparar Documentos

```python
documents = [
    "Python fue creado por Guido van Rossum...",
    "Machine Learning es una rama de la IA..."
]
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

### Paso 2: Chunking de Texto

```python
def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
```

### Paso 3: Indexar en ChromaDB

Crear colección e insertar chunks con metadatos.

### Paso 4: Búsqueda de Contexto

Recuperar chunks relevantes para una pregunta.

### Paso 5: Generar Respuesta

Usar LLM para responder basándose en el contexto.

---

## 📁 Estructura

```
ejercicio-03-qa-documents/
├── README.md
└── starter/
    └── main.py
```

---

## ▶️ Ejecución

```bash
cd bootcamp/04-especializacion/week-32/2-practicas/ejercicio-03-qa-documents
python starter/main.py
```

---

## ✅ Criterios de Éxito

- [ ] Implemento chunking efectivo
- [ ] Indexo documentos en ChromaDB
- [ ] Recupero contexto relevante
- [ ] Genero respuestas coherentes

---

## 🔗 Recursos

- [RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [OpenAI API](https://platform.openai.com/docs/guides/text-generation)
