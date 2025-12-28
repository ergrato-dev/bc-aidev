# 🎯 Ejercicio 02: ChromaDB CRUD

## 🎯 Objetivo

Dominar operaciones CRUD con ChromaDB para almacenar y buscar embeddings.

---

## 📋 Descripción

En este ejercicio aprenderás a usar ChromaDB, una base de datos vectorial local, para crear colecciones, insertar documentos, buscar por similitud y aplicar filtros de metadatos.

---

## 🔧 Requisitos

```bash
pip install chromadb sentence-transformers
```

---

## 🔧 Pasos del Ejercicio

### Paso 1: Crear Cliente y Colección

```python
import chromadb

client = chromadb.Client()  # In-memory
collection = client.create_collection("mi_coleccion")
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

### Paso 2: Insertar Documentos

```python
collection.add(
    documents=["texto1", "texto2"],
    ids=["id1", "id2"],
    metadatas=[{"tipo": "a"}, {"tipo": "b"}]
)
```

### Paso 3: Búsqueda por Similitud

```python
results = collection.query(
    query_texts=["mi búsqueda"],
    n_results=3
)
```

### Paso 4: Filtros de Metadata

```python
results = collection.query(
    query_texts=["búsqueda"],
    where={"tipo": "a"},
    n_results=3
)
```

### Paso 5: Actualizar y Eliminar

Operaciones de actualización y borrado.

---

## 📁 Estructura

```
ejercicio-02-chromadb/
├── README.md
└── starter/
    └── main.py
```

---

## ▶️ Ejecución

```bash
cd bootcamp/04-especializacion/week-32/2-practicas/ejercicio-02-chromadb
python starter/main.py
```

---

## ✅ Criterios de Éxito

- [ ] Creo colecciones en ChromaDB
- [ ] Inserto documentos con metadatos
- [ ] Realizo búsquedas semánticas
- [ ] Aplico filtros de metadata

---

## 🔗 Recursos

- [ChromaDB Docs](https://docs.trychroma.com/)
- [ChromaDB GitHub](https://github.com/chroma-core/chroma)
