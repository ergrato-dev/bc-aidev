# 🎯 Ejercicio 03: Word Embeddings

## 🎯 Objetivo

Trabajar con word embeddings pre-entrenados y calcular similaridad semántica.

---

## 📋 Descripción

En este ejercicio aprenderás a cargar embeddings pre-entrenados, realizar operaciones vectoriales, y calcular similaridad entre palabras y documentos.

---

## 🔧 Pasos del Ejercicio

### Paso 1: Cargar Embeddings

Usamos Gensim para cargar modelos pre-entrenados:

```python
import gensim.downloader as api
model = api.load('glove-wiki-gigaword-50')  # 50 dimensiones
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

### Paso 2: Explorar Vectores

Cada palabra tiene un vector denso asociado:

```python
vector = model['king']
print(vector.shape)  # (50,)
print(vector[:5])    # Primeros 5 valores
```

### Paso 3: Similaridad Coseno

Implementar la fórmula de similaridad:

```python
import numpy as np

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2)
```

### Paso 4: Palabras Similares

Encontrar las palabras más cercanas:

```python
similar = model.most_similar('king', topn=5)
# [('queen', 0.89), ('prince', 0.85), ...]
```

### Paso 5: Analogías

La famosa operación rey - hombre + mujer = reina:

```python
result = model.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=1
)
```

### Paso 6: Embedding de Documentos

Promedio de embeddings de palabras:

```python
def document_embedding(text, model):
    words = text.lower().split()
    vectors = [model[w] for w in words if w in model]
    return np.mean(vectors, axis=0)
```

---

## 📁 Estructura

```
ejercicio-03-embeddings/
├── README.md
└── starter/
    └── main.py
```

---

## ▶️ Ejecución

```bash
cd bootcamp/04-especializacion/week-29/2-practicas/ejercicio-03-embeddings
python starter/main.py
```

---

## ✅ Criterios de Éxito

- [ ] Embeddings cargados correctamente
- [ ] Similaridad coseno implementada
- [ ] Analogías vectoriales funcionan
- [ ] Embedding de documento calculado
- [ ] Comparación de documentos funciona

---

## 🔗 Recursos

- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
- [Pre-trained Models](https://github.com/RaRe-Technologies/gensim-data)
