"""
Ejercicio 01: Pipelines de Hugging Face
=======================================

Aprende a usar los pipelines para tareas NLP comunes.

Instrucciones:
1. Lee cada sección y descomenta el código
2. Ejecuta el script después de cada paso
3. Experimenta con diferentes textos

Nota: La primera ejecución descargará modelos (~250MB+)
"""

# ============================================
# PASO 1: Setup e Imports
# ============================================
print("--- Paso 1: Setup ---")

# Descomenta las siguientes líneas:
# from transformers import pipeline
# import warnings
# warnings.filterwarnings('ignore')
#
# print('✓ Transformers importado correctamente')

print()


# ============================================
# PASO 2: Análisis de Sentimientos
# ============================================
print("--- Paso 2: Análisis de Sentimientos ---")

# El pipeline más básico: clasificar texto como positivo/negativo
# Modelo por defecto: distilbert-base-uncased-finetuned-sst-2-english

# Descomenta las siguientes líneas:
# print('Cargando pipeline de sentiment-analysis...')
# sentiment = pipeline('sentiment-analysis')
#
# # Probar con un texto
# result = sentiment('I absolutely love this product!')
# print(f'Resultado: {result}')
#
# # Probar con múltiples textos
# texts = [
#     "This movie was fantastic!",
#     "I hate waiting in long lines.",
#     "The weather is okay today.",
#     "What a terrible experience.",
# ]
#
# results = sentiment(texts)
# print('\nAnálisis de múltiples textos:')
# for text, res in zip(texts, results):
#     emoji = '😊' if res['label'] == 'POSITIVE' else '😞'
#     print(f"  {emoji} {res['label']:8} ({res['score']:.2%}) | {text}")

print()


# ============================================
# PASO 3: Reconocimiento de Entidades (NER)
# ============================================
print("--- Paso 3: Named Entity Recognition ---")

# NER identifica entidades como personas, lugares, organizaciones
# aggregation_strategy="simple" agrupa tokens de la misma entidad

# Descomenta las siguientes líneas:
# print('Cargando pipeline de NER...')
# ner = pipeline('ner', aggregation_strategy='simple')
#
# text = "Apple Inc. was founded by Steve Jobs in California in 1976."
# entities = ner(text)
#
# print(f'Texto: "{text}"')
# print('\nEntidades encontradas:')
# for entity in entities:
#     print(f"  {entity['word']:20} → {entity['entity_group']:5} ({entity['score']:.2%})")
#
# # Probar con otro texto
# text2 = "Elon Musk is the CEO of Tesla and SpaceX, based in Texas."
# entities2 = ner(text2)
#
# print(f'\nTexto: "{text2}"')
# print('Entidades:')
# for entity in entities2:
#     print(f"  {entity['word']:20} → {entity['entity_group']:5} ({entity['score']:.2%})")

print()


# ============================================
# PASO 4: Preguntas y Respuestas
# ============================================
print("--- Paso 4: Question Answering ---")

# QA extrae respuestas de un contexto dado
# El modelo busca el span de texto que responde la pregunta

# Descomenta las siguientes líneas:
# print('Cargando pipeline de QA...')
# qa = pipeline('question-answering')
#
# context = """
# Python is a high-level, general-purpose programming language.
# Its design philosophy emphasizes code readability. Python was
# created by Guido van Rossum and first released in 1991.
# It supports multiple programming paradigms, including structured,
# object-oriented and functional programming.
# """
#
# questions = [
#     "Who created Python?",
#     "When was Python first released?",
#     "What does Python emphasize?",
# ]
#
# print('Contexto sobre Python...\n')
# for question in questions:
#     result = qa(question=question, context=context)
#     print(f'Q: {question}')
#     print(f'A: {result["answer"]} (confidence: {result["score"]:.2%})\n')

print()


# ============================================
# PASO 5: Generación de Texto
# ============================================
print("--- Paso 5: Generación de Texto ---")

# Genera texto continuando un prompt
# GPT-2 es autoregresivo: predice el siguiente token

# Descomenta las siguientes líneas:
# print('Cargando pipeline de text-generation (GPT-2)...')
# generator = pipeline('text-generation', model='gpt2')
#
# prompts = [
#     "Artificial intelligence will",
#     "The future of technology is",
#     "Machine learning helps us",
# ]
#
# for prompt in prompts:
#     print(f'\nPrompt: "{prompt}"')
#     result = generator(
#         prompt,
#         max_length=40,
#         num_return_sequences=1,
#         do_sample=True,
#         temperature=0.7,
#         pad_token_id=50256  # GPT-2 EOS token
#     )
#     generated = result[0]['generated_text']
#     print(f'Generated: {generated}')

print()


# ============================================
# PASO 6: Zero-Shot Classification
# ============================================
print("--- Paso 6: Zero-Shot Classification ---")

# Clasifica texto en categorías sin entrenamiento previo
# Usa NLI (Natural Language Inference) internamente

# Descomenta las siguientes líneas:
# print('Cargando pipeline de zero-shot-classification...')
# zero_shot = pipeline('zero-shot-classification')
#
# texts_to_classify = [
#     "I need to buy groceries for dinner tonight",
#     "The stock market crashed by 5% today",
#     "Barcelona won the Champions League final",
#     "Python 3.12 introduces new features for developers",
# ]
#
# labels = ["shopping", "finance", "sports", "technology", "entertainment"]
#
# print(f'Categorías disponibles: {labels}\n')
#
# for text in texts_to_classify:
#     result = zero_shot(text, candidate_labels=labels)
#     top_label = result['labels'][0]
#     top_score = result['scores'][0]
#     print(f'"{text[:50]}..."')
#     print(f'  → {top_label} ({top_score:.2%})\n')

print()


# ============================================
# PASO 7: Comparación de Modelos
# ============================================
print("--- Paso 7: Usar Modelos Específicos ---")

# Puedes especificar qué modelo usar en cada pipeline

# Descomenta las siguientes líneas:
# # Modelo multilingüe para sentimientos
# print('Cargando modelo multilingüe...')
# sentiment_multi = pipeline(
#     'sentiment-analysis',
#     model='nlptown/bert-base-multilingual-uncased-sentiment'
# )
#
# # Probar en diferentes idiomas
# texts_multi = [
#     "I love this product!",           # Inglés
#     "¡Me encanta este producto!",     # Español
#     "J'adore ce produit!",            # Francés
#     "Ich liebe dieses Produkt!",      # Alemán
# ]
#
# print('\nSentiment multilingüe:')
# for text in texts_multi:
#     result = sentiment_multi(text)
#     print(f'  {result[0]["label"]:8} | {text}')

print()
print("=" * 50)
print("¡Ejercicio completado!")
print("Ahora sabes usar los pipelines de Hugging Face.")
