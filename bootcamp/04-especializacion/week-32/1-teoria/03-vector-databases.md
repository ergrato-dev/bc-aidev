# 🗄️ Bases de Datos Vectoriales

![Comparación de Vector Databases](../0-assets/04-vector-databases.svg)

## 🎯 Objetivos de Aprendizaje

- Entender qué son las bases de datos vectoriales
- Conocer las opciones disponibles (ChromaDB, Pinecone, etc.)
- Implementar operaciones CRUD con vectores
- Usar filtros y metadata en queries

---

## 📋 Contenido

### 1. ¿Qué es una Vector Database?

Una **base de datos vectorial** está optimizada para almacenar, indexar y buscar vectores de alta dimensión eficientemente.

```
┌─────────────────────────────────────────────────────────────────┐
│                    VECTOR DATABASE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Base de Datos Tradicional        Vector Database              │
│   ─────────────────────────        ───────────────              │
│                                                                 │
│   SELECT * FROM docs               query(                       │
│   WHERE title LIKE '%python%'        vector=[0.1, 0.2, ...],   │
│                                      n_results=5                │
│   → Búsqueda exacta                )                            │
│   → Por keywords                   → Búsqueda por similitud     │
│                                    → Por significado            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Comparativa de Vector Databases

| Database | Tipo | Mejor para | Características |
|----------|------|------------|-----------------|
| **ChromaDB** | Embebida | Desarrollo, prototipos | Fácil, Python-native |
| **Pinecone** | Cloud | Producción | Serverless, escalable |
| **Weaviate** | Self-hosted/Cloud | Empresas | GraphQL, módulos ML |
| **Qdrant** | Self-hosted/Cloud | Alto rendimiento | Rust, muy rápido |
| **Milvus** | Self-hosted | Big data | Escalabilidad masiva |
| **FAISS** | Librería | Investigación | Meta, muy optimizado |
| **pgvector** | Extensión | PostgreSQL users | SQL + vectores |

### 3. ChromaDB en Profundidad

ChromaDB es ideal para aprender y prototipar:

```python
import chromadb

# Cliente en memoria (desarrollo)
client = chromadb.Client()

# Cliente persistente (guarda en disco)
client = chromadb.PersistentClient(path="./chroma_db")
```

#### Crear Colección

```python
# Crear colección con función de embedding automática
collection = client.create_collection(
    name="my_documents",
    metadata={"description": "Documentos de ejemplo"}
)

# O con embedding personalizado
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="my_documents",
    embedding_function=ef
)
```

#### Insertar Documentos

```python
collection.add(
    documents=[
        "Python es un lenguaje de programación",
        "JavaScript se usa para desarrollo web",
        "SQL es para bases de datos relacionales"
    ],
    metadatas=[
        {"topic": "programming", "level": "beginner"},
        {"topic": "web", "level": "intermediate"},
        {"topic": "database", "level": "beginner"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

# Con embeddings pre-calculados
collection.add(
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["doc1", "doc2"],
    ids=["id1", "id2"]
)
```

#### Query Semántico

```python
# Búsqueda por texto (embedding automático)
results = collection.query(
    query_texts=["lenguaje para análisis de datos"],
    n_results=2
)

print(results['documents'])     # Documentos encontrados
print(results['distances'])     # Distancias (menor = más similar)
print(results['metadatas'])     # Metadata de cada doc
print(results['ids'])           # IDs de documentos

# Búsqueda por embedding
results = collection.query(
    query_embeddings=[query_vector],
    n_results=3
)
```

#### Filtros con Metadata

```python
# Filtrar por metadata
results = collection.query(
    query_texts=["programación"],
    n_results=5,
    where={"topic": "programming"}
)

# Operadores de comparación
results = collection.query(
    query_texts=["curso"],
    where={"level": {"$in": ["beginner", "intermediate"]}}
)

# Combinar condiciones
results = collection.query(
    query_texts=["tutorial"],
    where={
        "$and": [
            {"topic": {"$eq": "web"}},
            {"level": {"$ne": "advanced"}}
        ]
    }
)
```

#### Operadores Disponibles

| Operador | Descripción | Ejemplo |
|----------|-------------|---------|
| `$eq` | Igual | `{"field": {"$eq": "value"}}` |
| `$ne` | No igual | `{"field": {"$ne": "value"}}` |
| `$gt` | Mayor que | `{"count": {"$gt": 10}}` |
| `$gte` | Mayor o igual | `{"count": {"$gte": 10}}` |
| `$lt` | Menor que | `{"count": {"$lt": 10}}` |
| `$lte` | Menor o igual | `{"count": {"$lte": 10}}` |
| `$in` | En lista | `{"field": {"$in": ["a", "b"]}}` |
| `$nin` | No en lista | `{"field": {"$nin": ["a", "b"]}}` |

#### Actualizar y Eliminar

```python
# Actualizar documento
collection.update(
    ids=["doc1"],
    documents=["Nuevo contenido actualizado"],
    metadatas=[{"topic": "updated", "level": "advanced"}]
)

# Eliminar por ID
collection.delete(ids=["doc1", "doc2"])

# Eliminar por filtro
collection.delete(where={"topic": "deprecated"})
```

### 4. Pinecone (Cloud)

Para producción, Pinecone ofrece escalabilidad:

```python
from pinecone import Pinecone

# Inicializar
pc = Pinecone(api_key="your-api-key")

# Crear índice
pc.create_index(
    name="my-index",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# Conectar al índice
index = pc.Index("my-index")

# Insertar vectores
index.upsert(
    vectors=[
        {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"topic": "ai"}},
        {"id": "vec2", "values": [0.3, 0.4, ...], "metadata": {"topic": "ml"}}
    ]
)

# Query
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    include_metadata=True,
    filter={"topic": {"$eq": "ai"}}
)
```

### 5. Índices y Algoritmos

Las vector DBs usan algoritmos especializados para búsqueda eficiente:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALGORITMOS DE INDEXACIÓN                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   FLAT (Fuerza Bruta)                                           │
│   └── Compara con TODOS los vectores                            │
│       ✅ 100% accuracy                                          │
│       ❌ O(n) - lento para millones de docs                     │
│                                                                 │
│   HNSW (Hierarchical Navigable Small World)                     │
│   └── Grafo navegable multi-capa                                │
│       ✅ Muy rápido (~O(log n))                                 │
│       ✅ Alta precisión (~95%+)                                 │
│       ❌ Más memoria                                            │
│                                                                 │
│   IVF (Inverted File Index)                                     │
│   └── Agrupa vectores en clusters                               │
│       ✅ Balance velocidad/precisión                            │
│       ⚠️ Requiere entrenamiento                                 │
│                                                                 │
│   PQ (Product Quantization)                                     │
│   └── Comprime vectores                                         │
│       ✅ Muy compacto en memoria                                │
│       ❌ Menor precisión                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6. Patrones de Uso

#### Patrón: Colecciones por Dominio

```python
# Separar conocimiento por tema
docs_collection = client.get_or_create_collection("documentation")
faq_collection = client.get_or_create_collection("faq")
support_collection = client.get_or_create_collection("support_tickets")

# Query específico por dominio
results = faq_collection.query(
    query_texts=["¿Cómo resetear contraseña?"],
    n_results=3
)
```

#### Patrón: Metadata Rica

```python
collection.add(
    documents=["Contenido del artículo..."],
    metadatas=[{
        "source": "blog",
        "author": "Juan",
        "date": "2024-01-15",
        "category": "tutorial",
        "language": "es",
        "word_count": 1500
    }],
    ids=["article-001"]
)
```

---

## 🔑 Puntos Clave

1. **Vector DBs** optimizadas para búsqueda por similitud
2. **ChromaDB** ideal para desarrollo y prototipos
3. **Metadata + filtros** para búsquedas precisas
4. **HNSW** es el algoritmo más usado (rápido + preciso)

---

## ✅ Checklist de Verificación

- [ ] Puedo crear colecciones en ChromaDB
- [ ] Sé insertar documentos con metadata
- [ ] Domino queries con filtros
- [ ] Entiendo los algoritmos de indexación
