"""
Ejercicio 01: Prompt Engineering
================================

Aprende técnicas de prompt engineering: zero-shot, few-shot, CoT.

Instrucciones:
1. Lee cada sección y descomenta el código
2. Ejecuta el script después de cada paso
3. Compara resultados entre técnicas

Nota: Usaremos GPT-2 local. Para mejores resultados, usar GPT-3.5+
"""

# ============================================
# PASO 1: Setup
# ============================================
print("--- Paso 1: Setup ---")

# Descomenta las siguientes líneas:
# from transformers import pipeline, set_seed
# import warnings
# warnings.filterwarnings('ignore')
#
# # Fijar seed para reproducibilidad
# set_seed(42)
#
# # Cargar modelo de generación
# print('Cargando modelo GPT-2...')
# generator = pipeline(
#     'text-generation',
#     model='gpt2',
#     device=-1  # CPU
# )
#
# def generate(prompt: str, max_tokens: int = 50) -> str:
#     """Genera texto dado un prompt."""
#     result = generator(
#         prompt,
#         max_new_tokens=max_tokens,
#         num_return_sequences=1,
#         do_sample=True,
#         temperature=0.7,
#         pad_token_id=generator.tokenizer.eos_token_id
#     )
#     # Extraer solo lo generado (después del prompt)
#     generated = result[0]['generated_text'][len(prompt):]
#     return generated.strip()
#
# print('✓ Modelo cargado')

print()


# ============================================
# PASO 2: Zero-Shot Prompting
# ============================================
print("--- Paso 2: Zero-Shot Prompting ---")

# Clasificación sin ejemplos

# Descomenta las siguientes líneas:
# print('\n📌 Zero-Shot: Clasificación de Sentimiento\n')
#
# texts = [
#     "I absolutely love this product!",
#     "This is terrible, waste of money.",
#     "It's okay, nothing special.",
# ]
#
# zero_shot_template = """Task: Classify the sentiment as POSITIVE, NEGATIVE, or NEUTRAL.
#
# Text: "{text}"
# Sentiment:"""
#
# print('Resultados Zero-Shot:')
# for text in texts:
#     prompt = zero_shot_template.format(text=text)
#     result = generate(prompt, max_tokens=10)
#     print(f'  "{text[:40]}..."')
#     print(f'  → {result[:20]}')
#     print()

print()


# ============================================
# PASO 3: Few-Shot Prompting
# ============================================
print("--- Paso 3: Few-Shot Prompting ---")

# Clasificación con ejemplos

# Descomenta las siguientes líneas:
# print('\n📌 Few-Shot: Clasificación con Ejemplos\n')
#
# few_shot_template = """Classify the sentiment of each text.
#
# Text: "This is amazing, best purchase ever!" → POSITIVE
# Text: "Horrible experience, never again." → NEGATIVE
# Text: "It works fine, does the job." → NEUTRAL
#
# Text: "{text}" →"""
#
# print('Resultados Few-Shot:')
# for text in texts:
#     prompt = few_shot_template.format(text=text)
#     result = generate(prompt, max_tokens=5)
#     print(f'  "{text[:40]}..."')
#     print(f'  → {result[:15]}')
#     print()

print()


# ============================================
# PASO 4: Chain-of-Thought (CoT)
# ============================================
print("--- Paso 4: Chain-of-Thought ---")

# Razonamiento paso a paso

# Descomenta las siguientes líneas:
# print('\n📌 Chain-of-Thought: Problemas Matemáticos\n')
#
# # Sin CoT
# simple_prompt = """Problem: If John has 5 apples and buys 3 bags with 4 apples each,
# how many apples does he have?
# Answer:"""
#
# print('Sin CoT:')
# result_no_cot = generate(simple_prompt, max_tokens=20)
# print(f'  {result_no_cot}')
#
# # Con CoT
# cot_prompt = """Problem: If Mary has 3 oranges and buys 2 boxes with 6 oranges each,
# how many oranges does she have?
#
# Let's think step by step:
# 1. Mary starts with 3 oranges
# 2. She buys 2 boxes × 6 oranges = 12 oranges
# 3. Total: 3 + 12 = 15 oranges
# Answer: 15 oranges
#
# Problem: If John has 5 apples and buys 3 bags with 4 apples each,
# how many apples does he have?
#
# Let's think step by step:"""
#
# print('\nCon CoT:')
# result_cot = generate(cot_prompt, max_tokens=60)
# print(f'  {result_cot}')

print()


# ============================================
# PASO 5: Structured Output
# ============================================
print("--- Paso 5: Structured Output ---")

# Forzar formato JSON

# Descomenta las siguientes líneas:
# print('\n📌 Structured Output: Extracción a JSON\n')
#
# extraction_prompt = """Extract information from the text in JSON format.
#
# Text: "Microsoft was founded by Bill Gates in 1975 in Albuquerque."
# JSON: {"company": "Microsoft", "founder": "Bill Gates", "year": 1975, "location": "Albuquerque"}
#
# Text: "Amazon was started by Jeff Bezos in 1994 in Seattle."
# JSON: {"company": "Amazon", "founder": "Jeff Bezos", "year": 1994, "location": "Seattle"}
#
# Text: "Google was created by Larry Page and Sergey Brin in 1998 in California."
# JSON:"""
#
# result = generate(extraction_prompt, max_tokens=50)
# print(f'Resultado: {result}')

print()


# ============================================
# PASO 6: Role Prompting
# ============================================
print("--- Paso 6: Role Prompting ---")

# Asignar personalidad/expertise

# Descomenta las siguientes líneas:
# print('\n📌 Role Prompting: Expert Persona\n')
#
# role_prompt = """You are a friendly Python programming tutor with 10 years of experience.
# You explain concepts simply and always include code examples.
# You encourage students and make learning fun.
#
# Student: What is a list in Python?
# Tutor:"""
#
# result = generate(role_prompt, max_tokens=80)
# print(f'Respuesta del tutor:\n{result}')

print()


# ============================================
# PASO 7: Prompt Templates
# ============================================
print("--- Paso 7: Prompt Templates ---")

# Crear templates reutilizables

# Descomenta las siguientes líneas:
# class PromptTemplate:
#     """Template reutilizable para prompts."""
#
#     def __init__(self, template: str):
#         self.template = template
#
#     def format(self, **kwargs) -> str:
#         return self.template.format(**kwargs)
#
#
# # Template para clasificación
# classification_template = PromptTemplate("""
# Task: {task}
# Categories: {categories}
#
# Examples:
# {examples}
#
# Text: "{text}"
# Category:""")
#
# # Usar el template
# prompt = classification_template.format(
#     task="Classify the topic of the text",
#     categories="technology, sports, politics, entertainment",
#     examples='Text: "The new iPhone was announced" → technology\nText: "The team won the championship" → sports',
#     text="The president signed a new law today"
# )
#
# print('Template formateado:')
# print(prompt)
# print()
# result = generate(prompt, max_tokens=10)
# print(f'Resultado: {result}')

print()


# ============================================
# PASO 8: Iteración y Mejora
# ============================================
print("--- Paso 8: Iteración y Mejora ---")

# Proceso de mejora de prompts

# Descomenta las siguientes líneas:
# print('\n📌 Iteración de Prompts\n')
#
# # Versión 1: Muy simple
# v1 = "Summarize: The quick brown fox jumps over the lazy dog."
#
# # Versión 2: Con instrucciones
# v2 = """Summarize the following text in one sentence.
# Text: The quick brown fox jumps over the lazy dog.
# Summary:"""
#
# # Versión 3: Con formato y restricciones
# v3 = """Task: Summarize the text in exactly 5 words or less.
#
# Text: The quick brown fox jumps over the lazy dog.
# Summary (max 5 words):"""
#
# print('Comparando versiones:')
# for i, prompt in enumerate([v1, v2, v3], 1):
#     result = generate(prompt, max_tokens=20)
#     print(f'\nV{i}: {result[:50]}')

print()


# ============================================
# PASO 9: Manejo de Errores
# ============================================
print("--- Paso 9: Manejo de Edge Cases ---")

# Manejar casos problemáticos

# Descomenta las siguientes líneas:
# print('\n📌 Manejo de Edge Cases\n')
#
# # Prompt con fallback para preguntas fuera de alcance
# safe_prompt = """You are a helpful assistant specialized in Python programming.
#
# Rules:
# 1. Only answer questions about Python
# 2. If asked about other topics, say "I can only help with Python questions"
# 3. Be concise and include code examples
#
# User: What is the capital of France?
# Assistant:"""
#
# result = generate(safe_prompt, max_tokens=30)
# print(f'Respuesta segura: {result}')
#
# # Prompt sobre Python
# python_prompt = """You are a helpful assistant specialized in Python programming.
#
# Rules:
# 1. Only answer questions about Python
# 2. If asked about other topics, say "I can only help with Python questions"
# 3. Be concise and include code examples
#
# User: How do I read a file in Python?
# Assistant:"""
#
# result2 = generate(python_prompt, max_tokens=50)
# print(f'\nRespuesta Python: {result2}')

print()


# ============================================
# PASO 10: Evaluación de Prompts
# ============================================
print("--- Paso 10: Evaluación ---")

# Evaluar calidad de prompts

# Descomenta las siguientes líneas:
# def evaluate_classification(prompt_template: str, test_cases: list) -> dict:
#     """Evalúa un prompt de clasificación."""
#     correct = 0
#     total = len(test_cases)
#
#     for text, expected in test_cases:
#         prompt = prompt_template.format(text=text)
#         result = generate(prompt, max_tokens=10).strip().upper()
#
#         # Verificar si la respuesta contiene la categoría esperada
#         is_correct = expected.upper() in result
#         correct += int(is_correct)
#
#         print(f'  Text: "{text[:30]}..."')
#         print(f'  Expected: {expected}, Got: {result[:15]}')
#         print(f'  {"✓" if is_correct else "✗"}\n')
#
#     accuracy = correct / total * 100
#     return {'accuracy': accuracy, 'correct': correct, 'total': total}
#
# # Casos de prueba
# test_cases = [
#     ("This is the best day ever!", "POSITIVE"),
#     ("I hate this so much", "NEGATIVE"),
#     ("It's fine I guess", "NEUTRAL"),
# ]
#
# # Evaluar
# print('\n📊 Evaluación de Prompt:\n')
# template = 'Sentiment (POSITIVE/NEGATIVE/NEUTRAL): "{text}" →'
# metrics = evaluate_classification(template, test_cases)
# print(f'Accuracy: {metrics["accuracy"]:.1f}%')

print()
print("=" * 50)
print("¡Ejercicio completado!")
print("Ahora dominas técnicas de prompt engineering.")
