# 🎯 Ejercicio 01: Embeddings Semánticos

## 🎯 Objetivo

Dominar la generación de embeddings y búsqueda por similitud semántica.

---

## 📋 Descripción

En este ejercicio aprenderás a generar embeddings con sentence-transformers, calcular similitud coseno y construir un buscador semántico básico.

---

## 🔧 Requisitos

```bash
pip install sentence-transformers numpy
```

---

## 🔧 Pasos del Ejercicio

### Paso 1: Cargar Modelo de Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("Hola mundo")
print(f"Dimensiones: {embedding.shape}")  # (384,)
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

### Paso 2: Generar Embeddings de Múltiples Textos

```python
texts = ["Python es genial", "JavaScript es popular"]
embeddings = model.encode(texts)
```

### Paso 3: Calcular Similitud Coseno

```python
from numpy.linalg import norm
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))
```

### Paso 4: Búsqueda Semántica

Encontrar documentos similares a una query.

### Paso 5: Comparar Modelos

Evaluar diferentes modelos de embedding.

---

## 📁 Estructura

```
ejercicio-01-embeddings/
├── README.md
└── starter/
    └── main.py
```

---

## ▶️ Ejecución

```bash
cd bootcamp/04-especializacion/week-32/2-practicas/ejercicio-01-embeddings
python starter/main.py
```

---

## ✅ Criterios de Éxito

- [ ] Genero embeddings correctamente
- [ ] Calculo similitud coseno
- [ ] Implemento búsqueda semántica
- [ ] Entiendo las dimensiones de los vectores

---

## 🔗 Recursos

- [Sentence Transformers](https://www.sbert.net/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
