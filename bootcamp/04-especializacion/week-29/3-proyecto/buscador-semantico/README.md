# 🔍 Proyecto: Buscador Semántico

## 🎯 Objetivo

Construir un buscador semántico que encuentre documentos relevantes usando similaridad de embeddings.

---

## 📋 Descripción

En este proyecto crearás un motor de búsqueda semántica completo que:

1. Preprocesa y tokeniza documentos
2. Genera embeddings para cada documento
3. Permite buscar documentos por similaridad semántica
4. Muestra resultados ordenados por relevancia

---

## 🏗️ Arquitectura

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Documentos    │────▶│  Preprocesamiento │────▶│   Embeddings    │
│   (corpus)      │     │  + Tokenización   │     │   (vectores)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Resultados    │◀────│    Ranking por   │◀────│     Query       │
│   ordenados     │     │    similaridad   │     │   embedding     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## 📁 Estructura

```
buscador-semantico/
├── README.md
├── starter/
│   └── main.py      # Plantilla para implementar
└── solution/
    └── main.py      # Solución completa
```

---

## 🔧 Requisitos Funcionales

### Clase `SemanticSearchEngine`

Implementar los siguientes métodos:

1. **`__init__(model_name)`**: Inicializar con modelo de embeddings
2. **`preprocess(text)`**: Limpiar y normalizar texto
3. **`get_embedding(text)`**: Obtener embedding de texto
4. **`index_documents(documents)`**: Indexar corpus de documentos
5. **`search(query, top_k)`**: Buscar documentos similares
6. **`add_document(document)`**: Añadir documento al índice

---

## 📊 Dataset de Ejemplo

```python
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with many layers",
    "Natural language processing analyzes human language",
    "Computer vision enables machines to interpret images",
    "Reinforcement learning trains agents through rewards",
    "Python is widely used for data science projects",
    "TensorFlow and PyTorch are popular deep learning frameworks",
    "Word embeddings represent words as dense vectors",
]
```

---

## ▶️ Ejecución

```bash
cd bootcamp/04-especializacion/week-29/3-proyecto/buscador-semantico

# Ejecutar tu implementación
python starter/main.py

# Ver la solución
python solution/main.py
```

---

## 📝 Ejemplo de Uso

```python
# Crear buscador
engine = SemanticSearchEngine('glove-wiki-gigaword-50')

# Indexar documentos
engine.index_documents(documents)

# Buscar
results = engine.search("neural networks and AI", top_k=3)

# Mostrar resultados
for doc, score in results:
    print(f"[{score:.4f}] {doc}")
```

**Salida esperada:**
```
[0.8234] Deep learning uses neural networks with many layers
[0.7891] Machine learning is a subset of artificial intelligence
[0.6543] TensorFlow and PyTorch are popular deep learning frameworks
```

---

## ✅ Criterios de Evaluación

### Funcionalidad (40%)
- [ ] Preprocesamiento funciona correctamente
- [ ] Embeddings se calculan para documentos
- [ ] Búsqueda retorna resultados ordenados
- [ ] Manejo de palabras fuera de vocabulario

### Código (30%)
- [ ] Código bien organizado y modular
- [ ] Type hints en funciones
- [ ] Docstrings descriptivos
- [ ] Nombres descriptivos de variables

### Extras (30%)
- [ ] Búsqueda interactiva en terminal
- [ ] Soporte para múltiples modelos
- [ ] Métricas de evaluación (tiempo de búsqueda)
- [ ] Persistencia del índice

---

## 💡 Hints

1. **Modelo pequeño para pruebas**: Usa `glove-wiki-gigaword-50` (50 dims)
2. **Caché de embeddings**: Guarda embeddings calculados para no recalcular
3. **Normalización**: Considera normalizar vectores para búsqueda más rápida
4. **Manejo de OOV**: Ignora palabras que no están en el vocabulario

---

## 🔗 Recursos

- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [Pre-trained Word Vectors](https://github.com/RaRe-Technologies/gensim-data)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

---

## 🚀 Extensiones Opcionales

1. **TF-IDF Weighting**: Ponderar palabras por importancia
2. **BM25 Hybrid**: Combinar con búsqueda tradicional
3. **GUI Simple**: Interfaz web con Streamlit
4. **Evaluación**: Implementar métricas como MRR o NDCG
