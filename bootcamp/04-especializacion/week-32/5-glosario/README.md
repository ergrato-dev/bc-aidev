# 📖 Glosario - Semana 32: RAG y Vector Databases

## A

### ANN (Approximate Nearest Neighbor)
Algoritmo que encuentra vecinos aproximados en lugar de exactos, sacrificando precisión por velocidad. Usado en búsqueda vectorial a gran escala.

## B

### BM25
Algoritmo de ranking basado en frecuencia de términos. Usado en búsqueda tradicional (keyword search) y complementa búsqueda semántica en sistemas híbridos.

## C

### Chunk
Fragmento de un documento más grande. En RAG, los documentos se dividen en chunks para indexar y recuperar porciones relevantes.

```python
def chunk_text(text: str, size: int = 300) -> list[str]:
    return [text[i:i+size] for i in range(0, len(text), size)]
```

### ChromaDB
Base de datos vectorial open source, embebida, diseñada para ser simple de usar con aplicaciones de IA.

### Contexto
Información recuperada que se proporciona al LLM junto con la pregunta del usuario para generar respuestas más precisas.

### Cosine Similarity
Métrica que mide la similitud entre dos vectores basándose en el ángulo entre ellos.

$$\text{cos}(\theta) = \frac{A \cdot B}{||A|| \times ||B||}$$

## D

### Dense Retrieval
Búsqueda basada en embeddings densos (vectores de alta dimensionalidad). Contrasta con sparse retrieval (basado en términos).

### Distance Metrics
Funciones para medir distancia entre vectores:
- **L2 (Euclidean)**: Distancia directa
- **Cosine**: Basada en ángulo
- **Dot Product**: Producto punto

## E

### Embedding
Representación vectorial densa de texto, imágenes u otros datos. Captura significado semántico en un espacio vectorial.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("Hello world")  # shape: (384,)
```

### Embedding Model
Modelo que convierte texto a vectores. Ejemplos: all-MiniLM-L6-v2, OpenAI ada-002, Cohere embed.

## F

### Few-shot Learning
Técnica donde se proporcionan ejemplos en el prompt para guiar la respuesta del modelo.

## G

### Ground Truth
Respuestas correctas conocidas usadas para evaluar la calidad del sistema RAG.

### Grounding
Proceso de anclar las respuestas del LLM en información factual recuperada.

## H

### Hallucination
Cuando un LLM genera información incorrecta o inventada. RAG reduce este problema proporcionando contexto factual.

### HNSW (Hierarchical Navigable Small World)
Algoritmo de indexación vectorial que organiza datos en una estructura de grafo jerárquico. Muy eficiente para búsqueda ANN.

### Hybrid Search
Combinación de búsqueda semántica (vectores) con búsqueda por palabras clave (BM25) para mejores resultados.

## I

### Indexing
Proceso de almacenar embeddings en una estructura optimizada para búsqueda rápida.

### IVF (Inverted File Index)
Técnica de indexación que agrupa vectores en clusters para acelerar la búsqueda.

## K

### K-Nearest Neighbors (KNN)
Algoritmo que encuentra los k vectores más cercanos a una query.

## L

### Latent Space
Espacio vectorial donde los embeddings representan conceptos semánticos. Puntos cercanos tienen significados similares.

### LLM (Large Language Model)
Modelo de lenguaje grande que genera texto. En RAG, el LLM usa el contexto recuperado para responder.

## M

### Metadata
Información adicional asociada a documentos (autor, fecha, categoría). Permite filtrar resultados.

```python
collection.add(
    documents=["texto"],
    metadatas=[{"author": "Juan", "date": "2024"}]
)
```

### MTEB (Massive Text Embedding Benchmark)
Benchmark para evaluar modelos de embeddings en múltiples tareas.

## O

### Overlap
Solapamiento entre chunks consecutivos. Ayuda a mantener contexto en los bordes.

```python
chunk_size = 300
overlap = 50
# Chunks: [0:300], [250:550], [500:800]...
```

## P

### Pinecone
Base de datos vectorial cloud gestionada. Escalable y con características enterprise.

### Prompt Engineering
Técnica de diseñar prompts efectivos para obtener mejores respuestas del LLM.

### Prompt Augmentation
Proceso de enriquecer el prompt con contexto recuperado antes de enviarlo al LLM.

## Q

### Query
Texto de búsqueda del usuario que se convierte en embedding para buscar documentos similares.

### Qdrant
Base de datos vectorial escrita en Rust, conocida por su velocidad y eficiencia.

## R

### RAG (Retrieval Augmented Generation)
Técnica que combina recuperación de información con generación de texto para producir respuestas más precisas y fundamentadas.

**Pipeline:**
1. Query → Embedding
2. Vector Search → Top-K Documents
3. Prompt + Context → LLM
4. LLM → Response

### Recall@K
Métrica que mide qué proporción de documentos relevantes están en los top-k resultados.

$$\text{Recall@K} = \frac{\text{Relevant in top-K}}{\text{Total Relevant}}$$

### Reranking
Proceso de reordenar resultados de búsqueda usando un modelo más sofisticado después de la recuperación inicial.

### Retriever
Componente que busca y recupera documentos relevantes de la base de conocimiento.

## S

### Semantic Search
Búsqueda basada en significado en lugar de coincidencia exacta de palabras.

### Sentence Transformers
Biblioteca Python para generar embeddings de oraciones usando modelos transformer.

### Similarity Score
Puntuación que indica qué tan similar es un documento a la query (0 a 1 típicamente).

### Sparse Retrieval
Búsqueda basada en vectores dispersos (ej. TF-IDF, BM25). Complementa dense retrieval.

## T

### Top-K
Los k documentos más relevantes recuperados para una query.

### Transformer
Arquitectura de red neuronal base para modelos de embeddings y LLMs modernos.

## U

### Upsert
Operación que inserta un documento si no existe, o lo actualiza si ya existe.

## V

### Vector Database
Base de datos especializada en almacenar y buscar embeddings de alta dimensionalidad.

**Características clave:**
- Indexación eficiente (HNSW, IVF)
- Búsqueda de similitud rápida
- Filtrado por metadata
- Escalabilidad

### Vector Index
Estructura de datos optimizada para búsqueda de similitud vectorial.

### Vector Space
Espacio matemático donde cada documento está representado como un punto (vector).

## W

### Weaviate
Base de datos vectorial con soporte para búsqueda híbrida y módulos de ML integrados.

### Window Size
Tamaño del contexto que el modelo puede procesar. Importante al diseñar chunks.

---

## 🔗 Referencias Rápidas

| Concepto | Fórmula/Código |
|----------|----------------|
| Cosine Similarity | `np.dot(a,b) / (norm(a) * norm(b))` |
| Chunk Overlap | `overlap = chunk_size * 0.15` |
| Score from Distance | `score = 1 / (1 + distance)` |

---

## 🔗 Navegación

| ⬅️ Teoría | 🏠 Semana | Prácticas ➡️ |
|-----------|-----------|--------------|
| [1-teoria](../1-teoria/) | [README](../README.md) | [2-practicas](../2-practicas/) |
