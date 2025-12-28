"""
Ejercicio 03: Word Embeddings
============================

Aprende a trabajar con embeddings pre-entrenados y calcular similaridad.

Instrucciones:
1. Lee cada sección y descomenta el código
2. Ejecuta el script después de cada paso
3. Experimenta con diferentes palabras y analogías

Nota: La primera ejecución descargará el modelo (~66MB)
"""

from typing import List

import numpy as np

# ============================================
# PASO 1: Cargar Embeddings Pre-entrenados
# ============================================
print("--- Paso 1: Cargar Embeddings ---")

# Gensim permite descargar modelos pre-entrenados fácilmente
# glove-wiki-gigaword-50 es un modelo pequeño (50 dims) para pruebas

# Descomenta las siguientes líneas:
# import gensim.downloader as api
#
# print('Descargando modelo (puede tardar la primera vez)...')
# model = api.load('glove-wiki-gigaword-50')
# print(f'Modelo cargado: {model}')
# print(f'Vocabulario: {len(model.key_to_index)} palabras')
# print(f'Dimensiones: {model.vector_size}')

print()


# ============================================
# PASO 2: Explorar Vectores
# ============================================
print("--- Paso 2: Explorar Vectores ---")

# Cada palabra tiene un vector denso de 50 dimensiones

# Descomenta las siguientes líneas:
# word = 'king'
# vector = model[word]
#
# print(f'Palabra: "{word}"')
# print(f'Forma del vector: {vector.shape}')
# print(f'Primeros 10 valores: {vector[:10]}')
# print(f'Norma del vector: {np.linalg.norm(vector):.4f}')

print()


# ============================================
# PASO 3: Similaridad Coseno
# ============================================
print("--- Paso 3: Similaridad Coseno ---")


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calcula similaridad coseno entre dos vectores.

    Fórmula: cos(θ) = (A · B) / (||A|| × ||B||)

    Returns:
        Valor entre -1 y 1 (1 = idénticos, 0 = ortogonales)
    """
    # Descomenta las siguientes líneas:
    # dot_product = np.dot(v1, v2)
    # norm_v1 = np.linalg.norm(v1)
    # norm_v2 = np.linalg.norm(v2)
    #
    # if norm_v1 == 0 or norm_v2 == 0:
    #     return 0.0
    #
    # return dot_product / (norm_v1 * norm_v2)
    pass  # Elimina esta línea cuando descomentes


# Descomenta las siguientes líneas:
# # Comparar palabras relacionadas vs no relacionadas
# pairs = [
#     ('king', 'queen'),    # Relacionadas
#     ('cat', 'dog'),       # Relacionadas (animales)
#     ('cat', 'computer'),  # No relacionadas
#     ('good', 'bad'),      # Antónimos (curiosamente similares)
# ]
#
# print('Similaridad entre pares de palabras:')
# for w1, w2 in pairs:
#     if w1 in model and w2 in model:
#         sim = cosine_similarity(model[w1], model[w2])
#         print(f'  {w1} <-> {w2}: {sim:.4f}')

print()


# ============================================
# PASO 4: Palabras Más Similares
# ============================================
print("--- Paso 4: Palabras Más Similares ---")

# Gensim tiene método built-in para encontrar vecinos

# Descomenta las siguientes líneas:
# words_to_check = ['king', 'computer', 'python', 'happy']
#
# for word in words_to_check:
#     if word in model:
#         similar = model.most_similar(word, topn=5)
#         print(f'\nPalabras similares a "{word}":')
#         for sim_word, score in similar:
#             print(f'  {sim_word}: {score:.4f}')

print()


# ============================================
# PASO 5: Analogías Vectoriales
# ============================================
print("--- Paso 5: Analogías Vectoriales ---")

# La famosa propiedad de los embeddings:
# king - man + woman ≈ queen

# Descomenta las siguientes líneas:
# print('Analogías (A - B + C = ?):')
#
# analogies = [
#     # (A, B, C) -> A - B + C = ?
#     ('king', 'man', 'woman'),      # queen
#     ('paris', 'france', 'spain'),  # madrid
#     ('bigger', 'big', 'small'),    # smaller
# ]
#
# for a, b, c in analogies:
#     if a in model and b in model and c in model:
#         result = model.most_similar(
#             positive=[a, c],
#             negative=[b],
#             topn=3
#         )
#         print(f'\n  {a} - {b} + {c} = ?')
#         for word, score in result:
#             print(f'    → {word}: {score:.4f}')

print()


# ============================================
# PASO 6: Embedding de Documentos
# ============================================
print("--- Paso 6: Embedding de Documentos ---")


def document_embedding(text: str, model) -> np.ndarray:
    """
    Calcula embedding de documento como promedio de palabras.

    Args:
        text: Texto del documento
        model: Modelo de embeddings

    Returns:
        Vector promedio del documento
    """
    # Descomenta las siguientes líneas:
    # words = text.lower().split()
    # vectors = []
    #
    # for word in words:
    #     if word in model:
    #         vectors.append(model[word])
    #
    # if not vectors:
    #     return np.zeros(model.vector_size)
    #
    # return np.mean(vectors, axis=0)
    pass  # Elimina esta línea cuando descomentes


# Descomenta las siguientes líneas:
# doc1 = "the cat sat on the mat"
# doc2 = "the dog lay on the rug"
# doc3 = "python is a programming language"
#
# if model:
#     emb1 = document_embedding(doc1, model)
#     emb2 = document_embedding(doc2, model)
#     emb3 = document_embedding(doc3, model)
#
#     print('Embeddings de documentos calculados')
#     print(f'  doc1: "{doc1}"')
#     print(f'  doc2: "{doc2}"')
#     print(f'  doc3: "{doc3}"')
#
#     sim_12 = cosine_similarity(emb1, emb2)
#     sim_13 = cosine_similarity(emb1, emb3)
#     sim_23 = cosine_similarity(emb2, emb3)
#
#     print(f'\nSimilaridad entre documentos:')
#     print(f'  doc1 <-> doc2: {sim_12:.4f} (esperado: alto)')
#     print(f'  doc1 <-> doc3: {sim_13:.4f} (esperado: bajo)')
#     print(f'  doc2 <-> doc3: {sim_23:.4f} (esperado: bajo)')

print()


# ============================================
# PASO 7: Búsqueda Semántica Simple
# ============================================
print("--- Paso 7: Búsqueda Semántica ---")


def semantic_search(query: str, documents: List[str], model, top_k: int = 3):
    """
    Busca documentos más similares a una consulta.

    Args:
        query: Texto de búsqueda
        documents: Lista de documentos
        model: Modelo de embeddings
        top_k: Número de resultados

    Returns:
        Lista de (índice, documento, score)
    """
    # Descomenta las siguientes líneas:
    # query_emb = document_embedding(query, model)
    #
    # scores = []
    # for i, doc in enumerate(documents):
    #     doc_emb = document_embedding(doc, model)
    #     sim = cosine_similarity(query_emb, doc_emb)
    #     scores.append((i, doc, sim))
    #
    # # Ordenar por similaridad descendente
    # scores.sort(key=lambda x: x[2], reverse=True)
    # return scores[:top_k]
    pass  # Elimina esta línea cuando descomentes


# Descomenta las siguientes líneas:
# documents = [
#     "machine learning is a subset of artificial intelligence",
#     "the cat is sleeping on the couch",
#     "deep learning uses neural networks",
#     "python is great for data science",
#     "the dog is playing in the garden",
# ]
#
# query = "AI and neural networks"
#
# if model:
#     results = semantic_search(query, documents, model)
#
#     print(f'Query: "{query}"')
#     print(f'\nResultados más relevantes:')
#     for idx, doc, score in results:
#         print(f'  [{score:.4f}] {doc}')

print()
print("=" * 50)
print("¡Ejercicio completado!")
print("Ahora sabes trabajar con word embeddings.")
