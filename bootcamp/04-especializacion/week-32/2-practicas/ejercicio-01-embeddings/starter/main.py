"""
Ejercicio 01: Embeddings Semánticos
===================================

Aprende a generar embeddings y calcular similitud semántica.

Instrucciones:
1. Lee cada sección y descomenta el código
2. Ejecuta el script después de cada paso
3. Experimenta con diferentes textos
"""

import numpy as np

# ============================================
# PASO 1: Setup y Cargar Modelo
# ============================================
print("--- Paso 1: Setup ---")

# Descomenta las siguientes líneas:
# from sentence_transformers import SentenceTransformer
# import warnings
# warnings.filterwarnings('ignore')
#
# # Cargar modelo de embeddings
# print('Cargando modelo all-MiniLM-L6-v2...')
# model = SentenceTransformer('all-MiniLM-L6-v2')
# print('✓ Modelo cargado')
#
# # Generar un embedding simple
# text = "Python es un lenguaje de programación"
# embedding = model.encode(text)
#
# print(f'\nTexto: "{text}"')
# print(f'Shape del embedding: {embedding.shape}')
# print(f'Tipo: {type(embedding)}')
# print(f'Primeros 5 valores: {embedding[:5]}')

print()


# ============================================
# PASO 2: Embeddings de Múltiples Textos
# ============================================
print("--- Paso 2: Múltiples Textos ---")

# Descomenta las siguientes líneas:
# documents = [
#     "Python es ideal para data science",
#     "JavaScript domina el desarrollo web",
#     "SQL es esencial para bases de datos",
#     "Machine learning es una rama de la IA",
#     "Los gatos son mascotas populares"
# ]
#
# # Generar embeddings en batch (más eficiente)
# embeddings = model.encode(documents, show_progress_bar=True)
#
# print(f'\nDocumentos: {len(documents)}')
# print(f'Shape de embeddings: {embeddings.shape}')
# # (5, 384) = 5 documentos, 384 dimensiones cada uno

print()


# ============================================
# PASO 3: Similitud Coseno
# ============================================
print("--- Paso 3: Similitud Coseno ---")

# Descomenta las siguientes líneas:
# from numpy.linalg import norm
#
# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     """
#     Calcula la similitud coseno entre dos vectores.
#
#     Valores:
#     - 1.0: Idénticos
#     - 0.0: Ortogonales (sin relación)
#     - -1.0: Opuestos
#     """
#     return np.dot(a, b) / (norm(a) * norm(b))
#
# # Comparar documentos
# print('Comparando similitudes:')
#
# # Similar (ambos sobre programación)
# sim_python_js = cosine_similarity(embeddings[0], embeddings[1])
# print(f'  Python ↔ JavaScript: {sim_python_js:.4f}')
#
# # Similar (data science relacionado con ML)
# sim_python_ml = cosine_similarity(embeddings[0], embeddings[3])
# print(f'  Python ↔ ML: {sim_python_ml:.4f}')
#
# # Diferente (programación vs mascotas)
# sim_python_cats = cosine_similarity(embeddings[0], embeddings[4])
# print(f'  Python ↔ Gatos: {sim_python_cats:.4f}')

print()


# ============================================
# PASO 4: Matriz de Similitud
# ============================================
print("--- Paso 4: Matriz de Similitud ---")

# Descomenta las siguientes líneas:
# def similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
#     """Calcula matriz de similitud entre todos los embeddings."""
#     n = len(embeddings)
#     matrix = np.zeros((n, n))
#
#     for i in range(n):
#         for j in range(n):
#             matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])
#
#     return matrix
#
# # Calcular matriz
# sim_matrix = similarity_matrix(embeddings)
#
# print('Matriz de similitud:')
# print('     ', end='')
# for i in range(len(documents)):
#     print(f'D{i}    ', end='')
# print()
#
# for i, row in enumerate(sim_matrix):
#     print(f'D{i}  ', end='')
#     for val in row:
#         print(f'{val:.2f}  ', end='')
#     print()

print()


# ============================================
# PASO 5: Búsqueda Semántica
# ============================================
print("--- Paso 5: Búsqueda Semántica ---")

# Descomenta las siguientes líneas:
# class SemanticSearch:
#     """Buscador semántico simple."""
#
#     def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
#         self.model = SentenceTransformer(model_name)
#         self.documents = []
#         self.embeddings = None
#
#     def index(self, documents: list[str]):
#         """Indexa documentos."""
#         self.documents = documents
#         self.embeddings = self.model.encode(documents)
#         print(f'✓ Indexados {len(documents)} documentos')
#
#     def search(self, query: str, top_k: int = 3) -> list[tuple]:
#         """
#         Busca documentos similares a la query.
#
#         Returns:
#             Lista de (documento, score)
#         """
#         query_embedding = self.model.encode(query)
#
#         # Calcular similitudes
#         scores = []
#         for i, doc_emb in enumerate(self.embeddings):
#             score = cosine_similarity(query_embedding, doc_emb)
#             scores.append((self.documents[i], score))
#
#         # Ordenar por score (mayor primero)
#         scores.sort(key=lambda x: x[1], reverse=True)
#
#         return scores[:top_k]
#
# # Crear buscador
# searcher = SemanticSearch()
# searcher.index(documents)
#
# # Buscar
# query = "análisis de datos"
# print(f'\nQuery: "{query}"')
# print('Resultados:')
#
# results = searcher.search(query, top_k=3)
# for doc, score in results:
#     print(f'  {score:.4f}: {doc}')

print()


# ============================================
# PASO 6: Búsqueda con Más Documentos
# ============================================
print("--- Paso 6: Corpus Expandido ---")

# Descomenta las siguientes líneas:
# # Corpus más grande
# corpus = [
#     # Programación
#     "Python es un lenguaje de programación versátil",
#     "JavaScript permite crear aplicaciones web interactivas",
#     "Java es popular en desarrollo empresarial",
#     "C++ se usa para programación de sistemas",
#     "Rust ofrece seguridad de memoria",
#
#     # Data Science
#     "Machine learning predice patrones en datos",
#     "Deep learning usa redes neuronales profundas",
#     "Pandas facilita el análisis de datos en Python",
#     "NumPy es fundamental para cálculo numérico",
#     "Scikit-learn tiene algoritmos de ML",
#
#     # Bases de datos
#     "SQL es el lenguaje para bases de datos relacionales",
#     "MongoDB es una base de datos NoSQL",
#     "PostgreSQL es una BD relacional avanzada",
#     "Redis es una base de datos en memoria",
#
#     # Otros
#     "Los perros son las mascotas más leales",
#     "El café es una bebida estimulante",
#     "El fútbol es el deporte más popular del mundo"
# ]
#
# searcher2 = SemanticSearch()
# searcher2.index(corpus)
#
# # Múltiples queries
# queries = [
#     "cómo analizar datos",
#     "lenguaje para web",
#     "almacenar información",
#     "animales domésticos"
# ]
#
# for query in queries:
#     print(f'\n🔍 Query: "{query}"')
#     results = searcher2.search(query, top_k=2)
#     for doc, score in results:
#         print(f'   {score:.3f}: {doc[:50]}...')

print()


# ============================================
# PASO 7: Normalización de Embeddings
# ============================================
print("--- Paso 7: Normalización ---")

# Descomenta las siguientes líneas:
# # Verificar si los embeddings están normalizados
# embedding_test = model.encode("test")
# norma = np.linalg.norm(embedding_test)
# print(f'Norma del embedding: {norma:.4f}')
#
# if abs(norma - 1.0) < 0.01:
#     print('✓ Embeddings normalizados (norma ≈ 1)')
#     print('  → Similitud coseno = producto punto')
# else:
#     print('Embeddings no normalizados')
#     print('  → Usar cosine_similarity explícita')
#
# # Normalizar manualmente si es necesario
# def normalize(v: np.ndarray) -> np.ndarray:
#     """Normaliza un vector a norma 1."""
#     return v / np.linalg.norm(v)
#
# normalized = normalize(embedding_test)
# print(f'\nNorma después de normalizar: {np.linalg.norm(normalized):.4f}')

print()


# ============================================
# PASO 8: Diferentes Modelos
# ============================================
print("--- Paso 8: Comparar Modelos ---")

# Descomenta las siguientes líneas:
# # Nota: Este paso es opcional, requiere descargar más modelos
#
# models_info = {
#     'all-MiniLM-L6-v2': {'dim': 384, 'speed': 'Rápido', 'quality': 'Buena'},
#     'all-mpnet-base-v2': {'dim': 768, 'speed': 'Medio', 'quality': 'Mejor'},
#     'paraphrase-MiniLM-L6-v2': {'dim': 384, 'speed': 'Rápido', 'quality': 'Paráfrasis'},
# }
#
# print('Modelos de embedding populares:')
# print('-' * 50)
# for name, info in models_info.items():
#     print(f'{name}')
#     print(f'  Dimensiones: {info["dim"]}')
#     print(f'  Velocidad: {info["speed"]}')
#     print(f'  Calidad: {info["quality"]}')
#     print()
#
# # Para cargar otro modelo:
# # model_mpnet = SentenceTransformer('all-mpnet-base-v2')
# # embedding_mpnet = model_mpnet.encode("test")
# # print(f'all-mpnet-base-v2 shape: {embedding_mpnet.shape}')  # (768,)

print()


# ============================================
# PASO 9: Batch Processing Eficiente
# ============================================
print("--- Paso 9: Batch Processing ---")

# Descomenta las siguientes líneas:
# import time
#
# # Crear corpus grande
# large_corpus = [f"Documento número {i} sobre tema {i % 5}" for i in range(100)]
#
# # Método lento: uno por uno
# start = time.time()
# slow_embeddings = [model.encode(doc) for doc in large_corpus]
# slow_time = time.time() - start
#
# # Método rápido: batch
# start = time.time()
# fast_embeddings = model.encode(large_corpus, batch_size=32)
# fast_time = time.time() - start
#
# print(f'Tiempo uno por uno: {slow_time:.2f}s')
# print(f'Tiempo en batch: {fast_time:.2f}s')
# print(f'Speedup: {slow_time/fast_time:.1f}x')

print()


# ============================================
# PASO 10: Guardar y Cargar Embeddings
# ============================================
print("--- Paso 10: Persistencia ---")

# Descomenta las siguientes líneas:
# # Los embeddings son costosos de calcular
# # Es buena práctica guardarlos para reutilizar
#
# # Guardar
# np.save('embeddings.npy', embeddings)
# print('✓ Embeddings guardados en embeddings.npy')
#
# # Cargar
# loaded_embeddings = np.load('embeddings.npy')
# print(f'✓ Embeddings cargados: {loaded_embeddings.shape}')
#
# # Verificar que son iguales
# are_equal = np.allclose(embeddings, loaded_embeddings)
# print(f'✓ Verificación: {"Iguales" if are_equal else "Diferentes"}')
#
# # Limpiar archivo de prueba
# import os
# os.remove('embeddings.npy')
# print('✓ Archivo de prueba eliminado')

print()
print("=" * 50)
print("¡Ejercicio completado!")
print("Ahora dominas embeddings semánticos.")
