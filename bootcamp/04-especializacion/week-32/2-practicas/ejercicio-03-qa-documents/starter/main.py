"""
Ejercicio 03: Q&A sobre Documentos
==================================

Construye un pipeline RAG completo para responder preguntas.

Instrucciones:
1. Lee cada sección y descomenta el código
2. Ejecuta el script después de cada paso
3. Experimenta con diferentes documentos y preguntas
"""

# ============================================
# PASO 1: Setup y Documentos de Ejemplo
# ============================================
print('--- Paso 1: Setup ---')

# Descomenta las siguientes líneas:
# import chromadb
# from sentence_transformers import SentenceTransformer
# import warnings
# warnings.filterwarnings('ignore')
# 
# # Documentos de ejemplo (simularemos un knowledge base)
# DOCUMENTS = {
#     "python_history": """
#     Python es un lenguaje de programación creado por Guido van Rossum.
#     Fue lanzado por primera vez en 1991. El nombre viene del grupo de 
#     comedia Monty Python. Python es conocido por su sintaxis clara y 
#     legible. Es un lenguaje interpretado y de alto nivel. Python soporta
#     múltiples paradigmas: programación orientada a objetos, programación
#     funcional y programación procedural. La versión Python 3.0 fue
#     lanzada en 2008 con cambios importantes.
#     """,
#     
#     "machine_learning": """
#     Machine Learning es una rama de la inteligencia artificial. Permite
#     a las computadoras aprender de datos sin ser programadas explícitamente.
#     Hay tres tipos principales: aprendizaje supervisado, no supervisado
#     y por refuerzo. El aprendizaje supervisado usa datos etiquetados.
#     El no supervisado encuentra patrones en datos sin etiquetas.
#     El aprendizaje por refuerzo aprende mediante recompensas y castigos.
#     Scikit-learn es una biblioteca popular para ML en Python.
#     """,
#     
#     "deep_learning": """
#     Deep Learning es un subcampo del Machine Learning. Utiliza redes
#     neuronales artificiales con múltiples capas. Las CNNs son usadas
#     para procesamiento de imágenes. Las RNNs son usadas para secuencias
#     y texto. Los Transformers revolucionaron el NLP desde 2017. GPT y
#     BERT son modelos basados en Transformers. TensorFlow y PyTorch son
#     los frameworks más populares para Deep Learning.
#     """,
#     
#     "rag_explanation": """
#     RAG significa Retrieval Augmented Generation. Es una técnica que
#     combina búsqueda con generación de texto. Primero, se recuperan
#     documentos relevantes usando búsqueda semántica. Luego, estos
#     documentos se usan como contexto para un LLM. RAG reduce las
#     alucinaciones porque el modelo se basa en hechos reales. Es útil
#     para chatbots, asistentes de documentación y Q&A empresarial.
#     """
# }
# 
# print(f'Documentos cargados: {len(DOCUMENTS)}')
# for name in DOCUMENTS:
#     print(f'  - {name}: {len(DOCUMENTS[name])} caracteres')

print()


# ============================================
# PASO 2: Chunking de Texto
# ============================================
print('--- Paso 2: Chunking ---')

# Descomenta las siguientes líneas:
# def chunk_text(
#     text: str, 
#     chunk_size: int = 200, 
#     overlap: int = 50
# ) -> list[str]:
#     """
#     Divide texto en chunks con overlap.
#     
#     Args:
#         text: Texto a dividir
#         chunk_size: Tamaño de cada chunk en caracteres
#         overlap: Caracteres de solapamiento entre chunks
#     
#     Returns:
#         Lista de chunks
#     """
#     # Limpiar texto
#     text = ' '.join(text.split())  # Normalizar espacios
#     
#     chunks = []
#     start = 0
#     
#     while start < len(text):
#         end = start + chunk_size
#         chunk = text[start:end]
#         
#         # Si no es el último chunk, intentar cortar en espacio
#         if end < len(text):
#             last_space = chunk.rfind(' ')
#             if last_space > chunk_size // 2:
#                 chunk = chunk[:last_space]
#                 end = start + last_space
#         
#         chunks.append(chunk.strip())
#         start = end - overlap
#     
#     return chunks
# 
# # Probar chunking
# test_chunks = chunk_text(DOCUMENTS["python_history"], chunk_size=150, overlap=30)
# print(f'Documento "python_history" dividido en {len(test_chunks)} chunks:')
# for i, chunk in enumerate(test_chunks):
#     print(f'  Chunk {i}: "{chunk[:50]}..." ({len(chunk)} chars)')

print()


# ============================================
# PASO 3: Procesar Todos los Documentos
# ============================================
print('--- Paso 3: Procesar Documentos ---')

# Descomenta las siguientes líneas:
# def process_documents(
#     documents: dict[str, str],
#     chunk_size: int = 200,
#     overlap: int = 50
# ) -> tuple[list[str], list[str], list[dict]]:
#     """
#     Procesa diccionario de documentos en chunks con metadatos.
#     
#     Returns:
#         (chunks, ids, metadatas)
#     """
#     all_chunks = []
#     all_ids = []
#     all_metadatas = []
#     
#     for doc_name, doc_text in documents.items():
#         chunks = chunk_text(doc_text, chunk_size, overlap)
#         
#         for i, chunk in enumerate(chunks):
#             all_chunks.append(chunk)
#             all_ids.append(f"{doc_name}_chunk_{i}")
#             all_metadatas.append({
#                 "source": doc_name,
#                 "chunk_index": i,
#                 "total_chunks": len(chunks)
#             })
#     
#     return all_chunks, all_ids, all_metadatas
# 
# # Procesar
# chunks, ids, metadatas = process_documents(DOCUMENTS)
# print(f'Total de chunks generados: {len(chunks)}')
# print(f'Distribución por documento:')
# from collections import Counter
# for source, count in Counter(m['source'] for m in metadatas).items():
#     print(f'  {source}: {count} chunks')

print()


# ============================================
# PASO 4: Indexar en ChromaDB
# ============================================
print('--- Paso 4: Indexar en ChromaDB ---')

# Descomenta las siguientes líneas:
# # Crear cliente y colección
# client = chromadb.Client()
# 
# # Eliminar colección si existe (para re-ejecuciones)
# try:
#     client.delete_collection("knowledge_base")
# except:
#     pass
# 
# collection = client.create_collection(
#     name="knowledge_base",
#     metadata={"description": "Knowledge base para Q&A"}
# )
# 
# # Indexar todos los chunks
# collection.add(
#     documents=chunks,
#     ids=ids,
#     metadatas=metadatas
# )
# 
# print(f'✓ Indexados {collection.count()} chunks en ChromaDB')

print()


# ============================================
# PASO 5: Función de Búsqueda de Contexto
# ============================================
print('--- Paso 5: Búsqueda de Contexto ---')

# Descomenta las siguientes líneas:
# def search_context(
#     query: str,
#     collection,
#     n_results: int = 3
# ) -> list[dict]:
#     """
#     Busca chunks relevantes para una query.
#     
#     Returns:
#         Lista de diccionarios con chunk, source y score
#     """
#     results = collection.query(
#         query_texts=[query],
#         n_results=n_results,
#         include=["documents", "metadatas", "distances"]
#     )
#     
#     context_items = []
#     for doc, meta, dist in zip(
#         results['documents'][0],
#         results['metadatas'][0],
#         results['distances'][0]
#     ):
#         context_items.append({
#             'text': doc,
#             'source': meta['source'],
#             'score': 1 / (1 + dist)  # Convertir distancia a score
#         })
#     
#     return context_items
# 
# # Probar búsqueda
# query = "¿Quién creó Python?"
# context = search_context(query, collection)
# 
# print(f'🔍 Query: "{query}"')
# print('\nContexto recuperado:')
# for i, item in enumerate(context):
#     print(f'  {i+1}. [{item["source"]}] (score: {item["score"]:.3f})')
#     print(f'     "{item["text"][:80]}..."')

print()


# ============================================
# PASO 6: Construir Prompt Aumentado
# ============================================
print('--- Paso 6: Prompt Aumentado ---')

# Descomenta las siguientes líneas:
# def build_augmented_prompt(
#     query: str,
#     context_items: list[dict]
# ) -> str:
#     """
#     Construye prompt con contexto para el LLM.
#     """
#     context_text = "\n\n".join([
#         f"[Fuente: {item['source']}]\n{item['text']}"
#         for item in context_items
#     ])
#     
#     prompt = f"""Responde la siguiente pregunta basándote ÚNICAMENTE en el contexto proporcionado.
# Si la información no está en el contexto, di "No tengo información sobre eso".
# 
# CONTEXTO:
# {context_text}
# 
# PREGUNTA: {query}
# 
# RESPUESTA:"""
#     
#     return prompt
# 
# # Construir prompt
# prompt = build_augmented_prompt(query, context)
# print('Prompt aumentado:')
# print('-' * 50)
# print(prompt)
# print('-' * 50)

print()


# ============================================
# PASO 7: Simulador de LLM (sin API)
# ============================================
print('--- Paso 7: Simulador de Respuesta ---')

# Descomenta las siguientes líneas:
# def simulate_llm_response(query: str, context_items: list[dict]) -> str:
#     """
#     Simula respuesta de LLM extrayendo información del contexto.
#     (En producción, usarías OpenAI, Anthropic, etc.)
#     """
#     # Extraer información relevante del contexto
#     context_text = ' '.join([item['text'].lower() for item in context_items])
#     query_lower = query.lower()
#     
#     # Respuestas simuladas basadas en keywords
#     if 'python' in query_lower and 'cre' in query_lower:
#         if 'guido van rossum' in context_text:
#             return "Según el contexto, Python fue creado por Guido van Rossum y lanzado en 1991."
#     
#     if 'machine learning' in query_lower or 'ml' in query_lower:
#         if 'tipos' in query_lower or 'cuáles' in query_lower:
#             return "Según el contexto, hay tres tipos: supervisado, no supervisado y por refuerzo."
#     
#     if 'rag' in query_lower:
#         if 'retrieval' in context_text:
#             return "RAG (Retrieval Augmented Generation) combina búsqueda semántica con generación de texto para reducir alucinaciones."
#     
#     if 'deep learning' in query_lower:
#         if 'transformer' in context_text:
#             return "Según el contexto, Deep Learning usa redes neuronales profundas. Los Transformers revolucionaron el NLP."
#     
#     return "Basándome en el contexto proporcionado: " + context_items[0]['text'][:100]
# 
# # Probar simulador
# response = simulate_llm_response(query, context)
# print(f'🤖 Respuesta: {response}')

print()


# ============================================
# PASO 8: Pipeline RAG Completo
# ============================================
print('--- Paso 8: Pipeline RAG ---')

# Descomenta las siguientes líneas:
# class SimpleRAG:
#     """Pipeline RAG simple sin LLM externo."""
#     
#     def __init__(self, collection):
#         self.collection = collection
#     
#     def query(self, question: str, n_context: int = 3) -> dict:
#         """
#         Ejecuta el pipeline RAG completo.
#         
#         Returns:
#             Dict con respuesta, contexto y metadatos
#         """
#         # 1. Buscar contexto
#         context = search_context(question, self.collection, n_context)
#         
#         # 2. Generar respuesta
#         response = simulate_llm_response(question, context)
#         
#         # 3. Preparar resultado
#         return {
#             'question': question,
#             'answer': response,
#             'context': context,
#             'sources': list(set(item['source'] for item in context))
#         }
# 
# # Crear instancia
# rag = SimpleRAG(collection)
# 
# # Probar con varias preguntas
# questions = [
#     "¿Quién creó Python y cuándo?",
#     "¿Cuáles son los tipos de Machine Learning?",
#     "¿Qué es RAG y para qué sirve?",
#     "¿Qué son los Transformers en Deep Learning?"
# ]
# 
# print('Pipeline RAG - Respuestas:')
# print('=' * 50)
# 
# for q in questions:
#     result = rag.query(q)
#     print(f'\n❓ {result["question"]}')
#     print(f'💡 {result["answer"]}')
#     print(f'📚 Fuentes: {", ".join(result["sources"])}')

print()


# ============================================
# PASO 9: Filtrado por Fuente
# ============================================
print('--- Paso 9: Filtrado por Fuente ---')

# Descomenta las siguientes líneas:
# def search_with_filter(
#     query: str,
#     collection,
#     source_filter: str = None,
#     n_results: int = 3
# ) -> list[dict]:
#     """Búsqueda con filtro de fuente opcional."""
#     
#     query_params = {
#         'query_texts': [query],
#         'n_results': n_results,
#         'include': ["documents", "metadatas", "distances"]
#     }
#     
#     if source_filter:
#         query_params['where'] = {"source": source_filter}
#     
#     results = collection.query(**query_params)
#     
#     return [
#         {
#             'text': doc,
#             'source': meta['source'],
#             'score': 1 / (1 + dist)
#         }
#         for doc, meta, dist in zip(
#             results['documents'][0],
#             results['metadatas'][0],
#             results['distances'][0]
#         )
#     ]
# 
# # Buscar solo en deep_learning
# query = "¿Qué frameworks se usan?"
# 
# print(f'🔍 Query: "{query}"')
# 
# print('\nSin filtro:')
# for item in search_with_filter(query, collection):
#     print(f'  [{item["source"]}] {item["text"][:50]}...')
# 
# print('\nSolo deep_learning:')
# for item in search_with_filter(query, collection, source_filter="deep_learning"):
#     print(f'  [{item["source"]}] {item["text"][:50]}...')

print()


# ============================================
# PASO 10: Evaluación Simple
# ============================================
print('--- Paso 10: Evaluación ---')

# Descomenta las siguientes líneas:
# # Test cases con respuestas esperadas
# test_cases = [
#     {
#         'question': '¿Cuándo fue lanzado Python?',
#         'expected_keywords': ['1991'],
#         'expected_source': 'python_history'
#     },
#     {
#         'question': '¿Qué significa RAG?',
#         'expected_keywords': ['retrieval', 'generation'],
#         'expected_source': 'rag_explanation'
#     }
# ]
# 
# def evaluate_rag(rag_system, test_cases: list[dict]) -> dict:
#     """Evalúa el sistema RAG con test cases."""
#     results = {
#         'total': len(test_cases),
#         'correct_source': 0,
#         'keyword_hits': 0
#     }
#     
#     for test in test_cases:
#         result = rag_system.query(test['question'])
#         
#         # Verificar fuente
#         if test['expected_source'] in result['sources']:
#             results['correct_source'] += 1
#         
#         # Verificar keywords en respuesta
#         answer_lower = result['answer'].lower()
#         if any(kw.lower() in answer_lower for kw in test['expected_keywords']):
#             results['keyword_hits'] += 1
#     
#     results['source_accuracy'] = results['correct_source'] / results['total']
#     results['keyword_accuracy'] = results['keyword_hits'] / results['total']
#     
#     return results
# 
# # Evaluar
# eval_results = evaluate_rag(rag, test_cases)
# print('Resultados de evaluación:')
# print(f'  Precisión de fuentes: {eval_results["source_accuracy"]:.0%}')
# print(f'  Precisión de keywords: {eval_results["keyword_accuracy"]:.0%}')

print()


# ============================================
# PASO 11: Agregar Nuevos Documentos
# ============================================
print('--- Paso 11: Agregar Documentos ---')

# Descomenta las siguientes líneas:
# def add_document(
#     collection,
#     doc_name: str,
#     doc_text: str,
#     chunk_size: int = 200,
#     overlap: int = 50
# ):
#     """Agrega un nuevo documento al knowledge base."""
#     chunks = chunk_text(doc_text, chunk_size, overlap)
#     
#     new_ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
#     new_metadatas = [
#         {"source": doc_name, "chunk_index": i, "total_chunks": len(chunks)}
#         for i in range(len(chunks))
#     ]
#     
#     collection.add(
#         documents=chunks,
#         ids=new_ids,
#         metadatas=new_metadatas
#     )
#     
#     print(f'✓ Agregado "{doc_name}": {len(chunks)} chunks')
# 
# # Agregar nuevo documento
# new_doc = """
# Hugging Face es una plataforma para Machine Learning. Ofrece miles de
# modelos pre-entrenados gratuitos. Su biblioteca Transformers facilita
# el uso de modelos como BERT, GPT y T5. También tiene datasets y spaces
# para demos interactivas. Es muy popular en la comunidad de NLP.
# """
# 
# add_document(collection, "hugging_face", new_doc)
# 
# # Verificar que se agregó
# print(f'\nTotal documentos ahora: {collection.count()}')
# 
# # Probar query sobre el nuevo documento
# result = rag.query("¿Qué es Hugging Face?")
# print(f'\n❓ ¿Qué es Hugging Face?')
# print(f'💡 {result["answer"]}')

print()


# ============================================
# PASO 12: Pipeline con OpenAI (Opcional)
# ============================================
print('--- Paso 12: Con OpenAI (Opcional) ---')

# Descomenta las siguientes líneas SI tienes API key de OpenAI:
# import os
# from openai import OpenAI
# 
# # Configurar API key
# # export OPENAI_API_KEY="tu-api-key"
# api_key = os.getenv("OPENAI_API_KEY")
# 
# if api_key:
#     client = OpenAI(api_key=api_key)
#     
#     def generate_with_openai(query: str, context_items: list[dict]) -> str:
#         """Genera respuesta usando GPT."""
#         prompt = build_augmented_prompt(query, context_items)
#         
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "Eres un asistente que responde basándose en el contexto dado."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=200,
#             temperature=0.7
#         )
#         
#         return response.choices[0].message.content
#     
#     # Probar
#     context = search_context("¿Qué es Deep Learning?", collection)
#     answer = generate_with_openai("¿Qué es Deep Learning?", context)
#     print(f'Respuesta GPT: {answer}')
# else:
#     print('No hay OPENAI_API_KEY configurada')
#     print('Para usar OpenAI: export OPENAI_API_KEY="sk-..."')

print()
print('=' * 50)
print('¡Ejercicio completado!')
print('Has construido un pipeline RAG funcional.')
print('Próximo paso: Proyecto completo con documentos reales')
