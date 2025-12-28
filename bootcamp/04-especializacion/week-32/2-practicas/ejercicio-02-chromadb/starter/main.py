"""
Ejercicio 02: ChromaDB CRUD
===========================

Aprende operaciones CRUD con la base de datos vectorial ChromaDB.

Instrucciones:
1. Lee cada sección y descomenta el código
2. Ejecuta el script después de cada paso
3. Experimenta con diferentes queries
"""

# ============================================
# PASO 1: Setup ChromaDB
# ============================================
print('--- Paso 1: Setup ChromaDB ---')

# Descomenta las siguientes líneas:
# import chromadb
# from chromadb.utils import embedding_functions
# import warnings
# warnings.filterwarnings('ignore')
# 
# # Crear cliente en memoria (efímero)
# client = chromadb.Client()
# print('✓ Cliente ChromaDB creado (in-memory)')
# 
# # Para persistencia, usar:
# # client = chromadb.PersistentClient(path="./chroma_db")

print()


# ============================================
# PASO 2: Crear Colección
# ============================================
print('--- Paso 2: Crear Colección ---')

# Descomenta las siguientes líneas:
# # Configurar función de embedding
# # ChromaDB usa all-MiniLM-L6-v2 por defecto
# default_ef = embedding_functions.DefaultEmbeddingFunction()
# 
# # Crear colección
# collection = client.create_collection(
#     name="documentos",
#     embedding_function=default_ef,
#     metadata={"description": "Colección de prueba"}
# )
# 
# print(f'✓ Colección creada: {collection.name}')
# print(f'  Documentos: {collection.count()}')

print()


# ============================================
# PASO 3: Insertar Documentos (Add)
# ============================================
print('--- Paso 3: Insertar Documentos ---')

# Descomenta las siguientes líneas:
# # Documentos de ejemplo
# documents = [
#     "Python es un lenguaje de programación versátil",
#     "JavaScript se usa principalmente para desarrollo web",
#     "Machine learning es una rama de la inteligencia artificial",
#     "Las bases de datos almacenan información estructurada",
#     "Docker permite contenerizar aplicaciones",
#     "Kubernetes orquesta contenedores en producción",
#     "SQL es el lenguaje estándar para bases de datos relacionales",
#     "NoSQL ofrece flexibilidad en el esquema de datos"
# ]
# 
# # IDs únicos (obligatorios)
# ids = [f"doc_{i}" for i in range(len(documents))]
# 
# # Metadatos opcionales pero muy útiles
# metadatas = [
#     {"categoria": "programacion", "nivel": "basico"},
#     {"categoria": "programacion", "nivel": "basico"},
#     {"categoria": "ia", "nivel": "intermedio"},
#     {"categoria": "datos", "nivel": "basico"},
#     {"categoria": "devops", "nivel": "intermedio"},
#     {"categoria": "devops", "nivel": "avanzado"},
#     {"categoria": "datos", "nivel": "basico"},
#     {"categoria": "datos", "nivel": "intermedio"}
# ]
# 
# # Insertar
# collection.add(
#     documents=documents,
#     ids=ids,
#     metadatas=metadatas
# )
# 
# print(f'✓ Documentos insertados: {collection.count()}')

print()


# ============================================
# PASO 4: Búsqueda Básica (Query)
# ============================================
print('--- Paso 4: Búsqueda Básica ---')

# Descomenta las siguientes líneas:
# # Búsqueda semántica
# query = "lenguajes de programación"
# 
# results = collection.query(
#     query_texts=[query],
#     n_results=3
# )
# 
# print(f'🔍 Query: "{query}"')
# print('\nResultados:')
# 
# for i, (doc, distance, metadata) in enumerate(zip(
#     results['documents'][0],
#     results['distances'][0],
#     results['metadatas'][0]
# )):
#     # ChromaDB usa distancia L2 por defecto
#     # Menor distancia = más similar
#     print(f'  {i+1}. [dist={distance:.4f}] {doc[:50]}...')
#     print(f'      metadata: {metadata}')

print()


# ============================================
# PASO 5: Filtros con Where
# ============================================
print('--- Paso 5: Filtros con Where ---')

# Descomenta las siguientes líneas:
# # Buscar solo en categoría específica
# results_filtered = collection.query(
#     query_texts=["herramientas de desarrollo"],
#     n_results=3,
#     where={"categoria": "devops"}
# )
# 
# print('🔍 Query: "herramientas de desarrollo" (solo devops)')
# print('\nResultados:')
# 
# for doc, distance in zip(
#     results_filtered['documents'][0],
#     results_filtered['distances'][0]
# ):
#     print(f'  [dist={distance:.4f}] {doc}')

print()


# ============================================
# PASO 6: Filtros Avanzados
# ============================================
print('--- Paso 6: Filtros Avanzados ---')

# Descomenta las siguientes líneas:
# # Operadores disponibles: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
# # Operadores lógicos: $and, $or
# 
# # Ejemplo: nivel intermedio o avanzado
# results_advanced = collection.query(
#     query_texts=["tecnología moderna"],
#     n_results=5,
#     where={
#         "$or": [
#             {"nivel": "intermedio"},
#             {"nivel": "avanzado"}
#         ]
#     }
# )
# 
# print('🔍 Query con filtro $or (intermedio o avanzado):')
# for doc, meta in zip(
#     results_advanced['documents'][0],
#     results_advanced['metadatas'][0]
# ):
#     print(f'  [{meta["nivel"]}] {doc[:40]}...')
# 
# # Ejemplo: categoría datos Y nivel básico
# results_and = collection.query(
#     query_texts=["almacenamiento"],
#     n_results=3,
#     where={
#         "$and": [
#             {"categoria": "datos"},
#             {"nivel": "basico"}
#         ]
#     }
# )
# 
# print('\n🔍 Query con filtro $and (datos + básico):')
# for doc in results_and['documents'][0]:
#     print(f'  {doc}')

print()


# ============================================
# PASO 7: Where Document (Filtro de Texto)
# ============================================
print('--- Paso 7: Filtro de Texto ---')

# Descomenta las siguientes líneas:
# # Filtrar documentos que contienen palabra específica
# results_contains = collection.query(
#     query_texts=["desarrollo"],
#     n_results=5,
#     where_document={"$contains": "Python"}
# )
# 
# print('🔍 Query: "desarrollo" (documentos con "Python"):')
# for doc in results_contains['documents'][0]:
#     print(f'  {doc}')
# 
# # Nota: where_document filtra DESPUÉS de la búsqueda vectorial

print()


# ============================================
# PASO 8: Obtener por ID (Get)
# ============================================
print('--- Paso 8: Obtener por ID ---')

# Descomenta las siguientes líneas:
# # Obtener documentos específicos
# specific_docs = collection.get(
#     ids=["doc_0", "doc_2", "doc_4"]
# )
# 
# print('Documentos por ID:')
# for doc_id, doc, meta in zip(
#     specific_docs['ids'],
#     specific_docs['documents'],
#     specific_docs['metadatas']
# ):
#     print(f'  {doc_id}: {doc[:40]}... [{meta["categoria"]}]')
# 
# # Obtener con filtro de metadata (sin query)
# devops_docs = collection.get(
#     where={"categoria": "devops"}
# )
# 
# print(f'\nDocumentos de DevOps: {len(devops_docs["ids"])}')
# for doc in devops_docs['documents']:
#     print(f'  - {doc}')

print()


# ============================================
# PASO 9: Actualizar Documentos (Update)
# ============================================
print('--- Paso 9: Actualizar Documentos ---')

# Descomenta las siguientes líneas:
# # Actualizar metadata
# collection.update(
#     ids=["doc_0"],
#     metadatas=[{"categoria": "programacion", "nivel": "intermedio", "actualizado": True}]
# )
# 
# # Verificar actualización
# updated_doc = collection.get(ids=["doc_0"])
# print(f'Documento actualizado:')
# print(f'  ID: {updated_doc["ids"][0]}')
# print(f'  Metadata: {updated_doc["metadatas"][0]}')
# 
# # Actualizar documento completo
# collection.update(
#     ids=["doc_0"],
#     documents=["Python es el lenguaje líder para IA y Data Science"],
#     metadatas=[{"categoria": "ia", "nivel": "intermedio", "actualizado": True}]
# )
# 
# updated_doc = collection.get(ids=["doc_0"])
# print(f'\nDocumento re-actualizado:')
# print(f'  Texto: {updated_doc["documents"][0]}')
# print(f'  Metadata: {updated_doc["metadatas"][0]}')

print()


# ============================================
# PASO 10: Upsert (Insert or Update)
# ============================================
print('--- Paso 10: Upsert ---')

# Descomenta las siguientes líneas:
# # Upsert: Inserta si no existe, actualiza si existe
# collection.upsert(
#     ids=["doc_0", "doc_new"],
#     documents=[
#         "Python domina el mundo de la IA",
#         "Rust es un lenguaje de sistemas seguro"
#     ],
#     metadatas=[
#         {"categoria": "ia", "nivel": "avanzado"},
#         {"categoria": "programacion", "nivel": "avanzado"}
#     ]
# )
# 
# print(f'Documentos después de upsert: {collection.count()}')
# 
# # Verificar nuevo documento
# new_doc = collection.get(ids=["doc_new"])
# print(f'Nuevo documento: {new_doc["documents"][0]}')

print()


# ============================================
# PASO 11: Eliminar Documentos (Delete)
# ============================================
print('--- Paso 11: Eliminar Documentos ---')

# Descomenta las siguientes líneas:
# print(f'Documentos antes de eliminar: {collection.count()}')
# 
# # Eliminar por ID
# collection.delete(ids=["doc_new"])
# print(f'Después de eliminar doc_new: {collection.count()}')
# 
# # Eliminar por filtro
# # collection.delete(where={"nivel": "basico"})
# # Cuidado: esto eliminaría todos los documentos con nivel básico

print()


# ============================================
# PASO 12: Múltiples Colecciones
# ============================================
print('--- Paso 12: Múltiples Colecciones ---')

# Descomenta las siguientes líneas:
# # Crear otra colección
# collection_faq = client.create_collection(name="faq")
# 
# faqs = [
#     "¿Cómo instalar Python? Descarga desde python.org",
#     "¿Qué es pip? Es el gestor de paquetes de Python",
#     "¿Cómo crear un entorno virtual? Usa venv o conda"
# ]
# 
# collection_faq.add(
#     documents=faqs,
#     ids=[f"faq_{i}" for i in range(len(faqs))]
# )
# 
# # Listar colecciones
# print('Colecciones existentes:')
# for coll in client.list_collections():
#     print(f'  - {coll.name}: {coll.count()} documentos')
# 
# # Obtener colección existente
# existing = client.get_collection("documentos")
# print(f'\nRecuperada colección: {existing.name}')

print()


# ============================================
# PASO 13: Incluir Embeddings en Resultados
# ============================================
print('--- Paso 13: Incluir Embeddings ---')

# Descomenta las siguientes líneas:
# # Por defecto, query no devuelve embeddings
# results_with_emb = collection.query(
#     query_texts=["inteligencia artificial"],
#     n_results=2,
#     include=["documents", "metadatas", "distances", "embeddings"]
# )
# 
# print('Resultados con embeddings:')
# for i, emb in enumerate(results_with_emb['embeddings'][0]):
#     print(f'  Doc {i}: embedding shape = {len(emb)} dimensiones')
#     print(f'         primeros 5 valores: {emb[:5]}')

print()


# ============================================
# PASO 14: Persistencia
# ============================================
print('--- Paso 14: Persistencia ---')

# Descomenta las siguientes líneas:
# import os
# import shutil
# 
# # Crear cliente persistente
# persist_path = "./chroma_test_db"
# 
# # Limpiar si existe
# if os.path.exists(persist_path):
#     shutil.rmtree(persist_path)
# 
# persistent_client = chromadb.PersistentClient(path=persist_path)
# 
# # Crear colección y agregar datos
# persistent_collection = persistent_client.create_collection("persistent_docs")
# persistent_collection.add(
#     documents=["Datos persistentes en disco"],
#     ids=["persist_1"]
# )
# 
# print(f'✓ Datos guardados en {persist_path}')
# print(f'  Archivos: {os.listdir(persist_path)}')
# 
# # En otra sesión, los datos seguirían disponibles:
# # client2 = chromadb.PersistentClient(path=persist_path)
# # coll2 = client2.get_collection("persistent_docs")
# # print(coll2.count())  # 1
# 
# # Limpiar
# shutil.rmtree(persist_path)
# print('✓ Directorio de prueba eliminado')

print()


# ============================================
# PASO 15: Resumen de Operaciones
# ============================================
print('--- Paso 15: Resumen ---')

# Descomenta las siguientes líneas:
# print("""
# RESUMEN DE OPERACIONES CHROMADB
# ================================
# 
# CREAR:
#   client = chromadb.Client()                    # In-memory
#   client = chromadb.PersistentClient(path)      # Persistente
#   collection = client.create_collection(name)   # Nueva colección
#   
# INSERTAR:
#   collection.add(documents, ids, metadatas)     # Insertar nuevos
#   collection.upsert(documents, ids, metadatas)  # Insert/Update
# 
# LEER:
#   collection.query(query_texts, n_results)      # Búsqueda semántica
#   collection.get(ids)                           # Por IDs
#   collection.get(where={...})                   # Por filtro
# 
# ACTUALIZAR:
#   collection.update(ids, documents, metadatas)  # Actualizar existentes
# 
# ELIMINAR:
#   collection.delete(ids)                        # Por IDs
#   collection.delete(where={...})                # Por filtro
# 
# FILTROS:
#   where={"campo": "valor"}                      # Igualdad
#   where={"campo": {"$gt": 5}}                   # Mayor que
#   where={"$or": [{...}, {...}]}                 # OR lógico
#   where={"$and": [{...}, {...}]}                # AND lógico
# """)

print()
print('=' * 50)
print('¡Ejercicio completado!')
print('Ahora dominas ChromaDB.')
