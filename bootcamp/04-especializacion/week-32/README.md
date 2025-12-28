# 🔍 Semana 32: RAG - Retrieval Augmented Generation

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Entender qué es RAG y por qué es fundamental para LLMs
- ✅ Implementar pipelines de embeddings y búsqueda semántica
- ✅ Trabajar con bases de datos vectoriales (ChromaDB, Pinecone)
- ✅ Construir sistemas de Q&A sobre documentos propios
- ✅ Optimizar retrieval con chunking y reranking

---

## 📚 Requisitos Previos

- Semana 31: LLMs y prompt engineering
- Conocimiento de embeddings (Semana 29-30)
- Python con manejo de APIs

---

## 🗂️ Estructura de la Semana

```
week-32/
├── README.md                    # Este archivo
├── rubrica-evaluacion.md        # Criterios de evaluación
├── 0-assets/                    # Diagramas y recursos visuales
│   ├── 01-rag-architecture.svg
│   ├── 02-embeddings-space.svg
│   ├── 03-chunking-strategies.svg
│   └── 04-vector-databases.svg
├── 1-teoria/                    # Material teórico
│   ├── 01-introduccion-rag.md
│   ├── 02-embeddings-vectores.md
│   ├── 03-vector-databases.md
│   └── 04-optimizacion-rag.md
├── 2-practicas/                 # Ejercicios guiados
│   ├── ejercicio-01-embeddings/
│   ├── ejercicio-02-chromadb/
│   └── ejercicio-03-qa-documents/
├── 3-proyecto/                  # Proyecto semanal
│   └── asistente-documentos/
├── 4-recursos/                  # Material adicional
│   └── README.md
└── 5-glosario/                  # Términos clave
    └── README.md
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| #   | Tema                    | Archivo                                                           | Duración |
| --- | ----------------------- | ----------------------------------------------------------------- | -------- |
| 1   | Introducción a RAG      | [01-introduccion-rag.md](1-teoria/01-introduccion-rag.md)         | 25 min   |
| 2   | Embeddings y Vectores   | [02-embeddings-vectores.md](1-teoria/02-embeddings-vectores.md)   | 25 min   |
| 3   | Bases de Datos Vectoriales | [03-vector-databases.md](1-teoria/03-vector-databases.md)      | 20 min   |
| 4   | Optimización de RAG     | [04-optimizacion-rag.md](1-teoria/04-optimizacion-rag.md)         | 20 min   |

### 💻 Prácticas (2.5 horas)

| #   | Ejercicio                | Carpeta                                                              | Duración |
| --- | ------------------------ | -------------------------------------------------------------------- | -------- |
| 1   | Embeddings Semánticos    | [ejercicio-01-embeddings/](2-practicas/ejercicio-01-embeddings/)     | 45 min   |
| 2   | ChromaDB                 | [ejercicio-02-chromadb/](2-practicas/ejercicio-02-chromadb/)         | 50 min   |
| 3   | Q&A sobre Documentos     | [ejercicio-03-qa-documents/](2-practicas/ejercicio-03-qa-documents/) | 55 min   |

### 📦 Proyecto (2 horas)

| Proyecto               | Descripción                                         | Carpeta                                                   |
| ---------------------- | --------------------------------------------------- | --------------------------------------------------------- |
| Asistente de Documentos | Sistema RAG completo para Q&A sobre PDFs/textos    | [asistente-documentos/](3-proyecto/asistente-documentos/) |

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

### ¿Qué es RAG?

**Retrieval Augmented Generation** combina:
1. **Retrieval**: Buscar información relevante en una base de conocimiento
2. **Augmentation**: Inyectar esa información en el prompt
3. **Generation**: El LLM genera respuesta basada en el contexto

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   PREGUNTA   │───▶│   RETRIEVAL  │───▶│  DOCUMENTOS  │
│   del user   │    │   (buscar)   │    │  relevantes  │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
┌──────────────┐    ┌──────────────┐           │
│   RESPUESTA  │◀───│     LLM      │◀──────────┘
│   final      │    │  (generar)   │   + prompt original
└──────────────┘    └──────────────┘
```

### ¿Por qué RAG?

| Problema del LLM | Solución con RAG |
|------------------|------------------|
| Conocimiento desactualizado | Datos en tiempo real |
| Alucinaciones | Fuentes verificables |
| Sin datos privados | Tu propia base de conocimiento |
| Contexto limitado | Retrieval selectivo |

---

## 🛠️ Stack Tecnológico

| Tecnología | Versión | Uso |
|------------|---------|-----|
| sentence-transformers | Latest | Embeddings |
| ChromaDB | 0.4+ | Vector DB local |
| Pinecone | Latest | Vector DB cloud |
| LangChain | Latest | Orquestación |
| PyPDF2 / pdfplumber | Latest | Procesamiento PDF |

---

## 📌 Entregables

Al finalizar la semana debes entregar:

1. **Ejercicios completados** (2-practicas/)
   - [ ] ejercicio-01: Embeddings y similitud semántica
   - [ ] ejercicio-02: CRUD con ChromaDB
   - [ ] ejercicio-03: Pipeline Q&A básico

2. **Proyecto semanal** (3-proyecto/)
   - [ ] Sistema RAG funcional
   - [ ] Soporte para múltiples documentos
   - [ ] Interfaz de chat
   - [ ] Código documentado

3. **Autoevaluación**
   - [ ] Completar checklist de verificación
   - [ ] Documentar decisiones de diseño

---

## 🔗 Navegación

| ⬅️ Anterior                     | 🏠 Módulo                                    | Siguiente ➡️                     |
| ------------------------------ | ------------------------------------------- | -------------------------------- |
| [Semana 31](../week-31/README.md) | [Especialización](../README.md) | [Semana 33](../week-33/README.md) |

---

## 💡 Tips para esta Semana

> 🎯 **Consejo**: RAG es el patrón más usado en producción para aplicaciones LLM. Domínalo bien porque lo usarás constantemente.

- **Empieza simple**: Un documento, queries básicas
- **Itera el chunking**: El tamaño de chunks afecta mucho la calidad
- **Evalúa retrieval**: Antes de culpar al LLM, verifica qué recuperas
- **Cachea embeddings**: Son costosos de calcular

---

_Semana 32 de 36 | Módulo: Especialización | Bootcamp IA: Zero to Hero_
