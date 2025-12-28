# 📦 Proyecto: Asistente de Documentos con RAG

## 🎯 Objetivo

Construir un asistente inteligente que responda preguntas sobre documentos usando RAG (Retrieval Augmented Generation).

---

## 📋 Descripción

Crearás un sistema RAG completo que:
- Carga y procesa documentos de texto
- Los indexa en una base de datos vectorial
- Responde preguntas basándose en el contenido
- Cita las fuentes utilizadas

---

## 🔧 Requisitos

```bash
pip install chromadb sentence-transformers rich
```

Opcional (para usar LLM real):
```bash
pip install openai python-dotenv
```

---

## 📁 Estructura

```
asistente-documentos/
├── README.md
├── starter/
│   └── main.py
└── solution/
    └── main.py
```

---

## 🎯 Funcionalidades a Implementar

### Nivel Básico
1. **Cargar documentos** desde archivos de texto o strings
2. **Chunking** de documentos largos
3. **Indexar** en ChromaDB
4. **Buscar** contexto relevante
5. **Generar** respuestas

### Nivel Intermedio
6. **Metadatos** enriquecidos (fecha, autor, categoría)
7. **Filtrado** por fuente o categoría
8. **Reranking** de resultados
9. **Historial** de conversación

### Nivel Avanzado
10. **Integración** con OpenAI/Anthropic
11. **Evaluación** automática de calidad
12. **Interfaz** de usuario con Rich

---

## ▶️ Ejecución

```bash
# Con solución
python solution/main.py

# Tu implementación
python starter/main.py
```

---

## 📝 Instrucciones

### 1. Implementa la clase `DocumentProcessor`

```python
class DocumentProcessor:
    def load_document(self, text: str, source: str) -> None:
        """Carga un documento."""
        # TODO: Implementar
        pass
    
    def chunk_documents(self, chunk_size: int = 300) -> list:
        """Divide documentos en chunks."""
        # TODO: Implementar
        pass
```

### 2. Implementa la clase `VectorStore`

```python
class VectorStore:
    def add_chunks(self, chunks: list, metadatas: list) -> None:
        """Añade chunks a la colección."""
        # TODO: Implementar
        pass
    
    def search(self, query: str, n_results: int = 3) -> list:
        """Busca chunks relevantes."""
        # TODO: Implementar
        pass
```

### 3. Implementa la clase `RAGAssistant`

```python
class RAGAssistant:
    def answer(self, question: str) -> dict:
        """Responde una pregunta usando RAG."""
        # TODO: Implementar
        pass
```

---

## ✅ Criterios de Evaluación

### Conocimiento (30%)
- [ ] Explica correctamente el pipeline RAG
- [ ] Entiende chunking y su importancia
- [ ] Conoce métricas de evaluación

### Desempeño (40%)
- [ ] Implementa chunking efectivo
- [ ] Usa ChromaDB correctamente
- [ ] Búsqueda semántica funcional

### Producto (30%)
- [ ] Código limpio y documentado
- [ ] Respuestas coherentes
- [ ] Citas de fuentes correctas

---

## 📚 Documentos de Prueba

El proyecto incluye documentos de ejemplo sobre Python, Machine Learning y RAG. Puedes añadir tus propios documentos para probar.

---

## 💡 Tips

1. **Chunk size**: 200-500 caracteres suele funcionar bien
2. **Overlap**: 10-20% del chunk size
3. **Top-k**: 3-5 documentos de contexto
4. **Prompt**: Sé específico sobre cómo usar el contexto

---

## 🔗 Recursos

- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [ChromaDB Guide](https://docs.trychroma.com/guides)
- [LangChain RAG](https://python.langchain.com/docs/tutorials/rag/)
