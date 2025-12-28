"""
Ejercicio 02: Generación de Texto
=================================

Aprende a controlar la generación con parámetros.

Instrucciones:
1. Lee cada sección y descomenta el código
2. Ejecuta el script después de cada paso
3. Compara outputs con diferentes parámetros
"""

import torch

# ============================================
# PASO 1: Setup
# ============================================
print("--- Paso 1: Setup ---")

# Descomenta las siguientes líneas:
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import warnings
# warnings.filterwarnings('ignore')
#
# # Cargar modelo y tokenizer
# model_name = 'gpt2'
# print(f'Cargando {model_name}...')
#
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
#
# # GPT-2 no tiene pad_token, usar eos_token
# tokenizer.pad_token = tokenizer.eos_token
#
# # Modo evaluación
# model.eval()
#
# def generate_text(prompt: str, **kwargs) -> str:
#     """Genera texto con parámetros configurables."""
#     inputs = tokenizer(prompt, return_tensors='pt')
#
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs['input_ids'],
#             attention_mask=inputs['attention_mask'],
#             pad_token_id=tokenizer.eos_token_id,
#             **kwargs
#         )
#
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)
#
# print('✓ Modelo cargado')

print()


# ============================================
# PASO 2: Temperature
# ============================================
print("--- Paso 2: Temperature ---")

# Temperature controla aleatoriedad
# Baja = determinista, Alta = creativo/caótico

# Descomenta las siguientes líneas:
# prompt = "The future of artificial intelligence is"
# print(f'Prompt: "{prompt}"\n')
#
# temperatures = [0.1, 0.5, 0.7, 1.0, 1.5]
#
# for temp in temperatures:
#     print(f'Temperature = {temp}:')
#     result = generate_text(
#         prompt,
#         max_new_tokens=30,
#         temperature=temp,
#         do_sample=True
#     )
#     # Mostrar solo la parte generada
#     generated = result[len(prompt):].strip()
#     print(f'  "{generated[:60]}..."\n')

print()


# ============================================
# PASO 3: Top-K Sampling
# ============================================
print("--- Paso 3: Top-K Sampling ---")

# Top-K limita a los K tokens más probables

# Descomenta las siguientes líneas:
# prompt = "Machine learning can be used for"
# print(f'Prompt: "{prompt}"\n')
#
# top_k_values = [1, 10, 50, 100]
#
# for k in top_k_values:
#     print(f'Top-K = {k}:')
#     result = generate_text(
#         prompt,
#         max_new_tokens=30,
#         top_k=k,
#         do_sample=True,
#         temperature=0.8
#     )
#     generated = result[len(prompt):].strip()
#     print(f'  "{generated[:60]}..."\n')

print()


# ============================================
# PASO 4: Top-P (Nucleus) Sampling
# ============================================
print("--- Paso 4: Top-P Sampling ---")

# Top-P selecciona tokens hasta acumular probabilidad P

# Descomenta las siguientes líneas:
# prompt = "In a world where robots"
# print(f'Prompt: "{prompt}"\n')
#
# top_p_values = [0.5, 0.8, 0.9, 0.95, 1.0]
#
# for p in top_p_values:
#     print(f'Top-P = {p}:')
#     result = generate_text(
#         prompt,
#         max_new_tokens=30,
#         top_p=p,
#         do_sample=True,
#         temperature=0.8
#     )
#     generated = result[len(prompt):].strip()
#     print(f'  "{generated[:60]}..."\n')

print()


# ============================================
# PASO 5: Combinando Top-K y Top-P
# ============================================
print("--- Paso 5: Combinando Parámetros ---")

# Se pueden combinar para mejor control

# Descomenta las siguientes líneas:
# prompt = "The secret to happiness is"
# print(f'Prompt: "{prompt}"\n')
#
# configs = [
#     {'top_k': 50, 'top_p': 1.0, 'temperature': 0.7},
#     {'top_k': 50, 'top_p': 0.9, 'temperature': 0.7},
#     {'top_k': 10, 'top_p': 0.9, 'temperature': 0.7},
# ]
#
# for cfg in configs:
#     print(f'Config: {cfg}')
#     result = generate_text(
#         prompt,
#         max_new_tokens=30,
#         do_sample=True,
#         **cfg
#     )
#     generated = result[len(prompt):].strip()
#     print(f'  "{generated[:60]}..."\n')

print()


# ============================================
# PASO 6: Repetition Penalty
# ============================================
print("--- Paso 6: Repetition Penalty ---")

# Penaliza tokens que ya aparecieron

# Descomenta las siguientes líneas:
# prompt = "The cat sat on the mat. The cat"
# print(f'Prompt: "{prompt}"\n')
#
# penalties = [1.0, 1.2, 1.5, 2.0]
#
# for penalty in penalties:
#     print(f'Repetition penalty = {penalty}:')
#     result = generate_text(
#         prompt,
#         max_new_tokens=40,
#         repetition_penalty=penalty,
#         do_sample=True,
#         temperature=0.7
#     )
#     generated = result[len(prompt):].strip()
#     print(f'  "{generated[:70]}..."\n')

print()


# ============================================
# PASO 7: Control de Longitud
# ============================================
print("--- Paso 7: Control de Longitud ---")

# max_new_tokens, min_length, max_length

# Descomenta las siguientes líneas:
# prompt = "Python is a programming language that"
# print(f'Prompt: "{prompt}"\n')
#
# # Generación corta
# print('Corta (max_new_tokens=20):')
# result = generate_text(prompt, max_new_tokens=20, do_sample=True, temperature=0.7)
# print(f'  "{result[len(prompt):].strip()}"\n')
#
# # Generación larga
# print('Larga (max_new_tokens=80):')
# result = generate_text(prompt, max_new_tokens=80, do_sample=True, temperature=0.7)
# print(f'  "{result[len(prompt):].strip()[:150]}..."\n')
#
# # Con longitud mínima
# print('Con mínimo (min_length=50):')
# result = generate_text(
#     prompt,
#     max_new_tokens=100,
#     min_length=50,
#     do_sample=True,
#     temperature=0.7
# )
# print(f'  Tokens generados: {len(tokenizer.encode(result)) - len(tokenizer.encode(prompt))}')

print()


# ============================================
# PASO 8: Beam Search
# ============================================
print("--- Paso 8: Beam Search ---")

# Explora múltiples secuencias en paralelo
# Más determinista que sampling

# Descomenta las siguientes líneas:
# prompt = "The quick brown fox"
# print(f'Prompt: "{prompt}"\n')
#
# # Sin beam search (greedy)
# print('Greedy (num_beams=1):')
# result = generate_text(prompt, max_new_tokens=20, num_beams=1)
# print(f'  "{result[len(prompt):].strip()}"\n')
#
# # Con beam search
# beam_sizes = [2, 4, 8]
# for beams in beam_sizes:
#     print(f'Beam search (num_beams={beams}):')
#     result = generate_text(
#         prompt,
#         max_new_tokens=20,
#         num_beams=beams,
#         early_stopping=True
#     )
#     print(f'  "{result[len(prompt):].strip()}"\n')

print()


# ============================================
# PASO 9: Múltiples Secuencias
# ============================================
print("--- Paso 9: Múltiples Secuencias ---")

# Generar varias opciones a la vez

# Descomenta las siguientes líneas:
# prompt = "Ideas for a new app:"
# print(f'Prompt: "{prompt}"\n')
#
# inputs = tokenizer(prompt, return_tensors='pt')
#
# with torch.no_grad():
#     outputs = model.generate(
#         inputs['input_ids'],
#         max_new_tokens=30,
#         num_return_sequences=3,
#         do_sample=True,
#         temperature=0.9,
#         top_p=0.9,
#         pad_token_id=tokenizer.eos_token_id
#     )
#
# print('Secuencias generadas:')
# for i, output in enumerate(outputs, 1):
#     text = tokenizer.decode(output, skip_special_tokens=True)
#     generated = text[len(prompt):].strip()
#     print(f'  {i}. "{generated[:60]}..."')

print()


# ============================================
# PASO 10: Configuración Óptima por Tarea
# ============================================
print("--- Paso 10: Configuraciones por Tarea ---")

# Diferentes tareas requieren diferentes configs

# Descomenta las siguientes líneas:
# # Configuraciones recomendadas
# configs = {
#     'creative_writing': {
#         'temperature': 0.9,
#         'top_p': 0.95,
#         'top_k': 50,
#         'repetition_penalty': 1.1
#     },
#     'factual_qa': {
#         'temperature': 0.3,
#         'top_p': 0.9,
#         'top_k': 40,
#         'repetition_penalty': 1.0
#     },
#     'code_generation': {
#         'temperature': 0.2,
#         'top_p': 0.95,
#         'top_k': 50,
#         'repetition_penalty': 1.1
#     },
#     'summarization': {
#         'temperature': 0.5,
#         'top_p': 0.9,
#         'num_beams': 4,
#         'repetition_penalty': 1.2
#     }
# }
#
# print('Configuraciones recomendadas por tarea:')
# for task, cfg in configs.items():
#     print(f'\n  📌 {task}:')
#     for key, value in cfg.items():
#         print(f'      {key}: {value}')
#
# # Probar creative writing
# print('\n\nEjemplo - Creative Writing:')
# prompt = "Once upon a time in a magical forest"
# result = generate_text(
#     prompt,
#     max_new_tokens=50,
#     do_sample=True,
#     **configs['creative_writing']
# )
# print(f'  "{result[len(prompt):].strip()[:100]}..."')

print()
print("=" * 50)
print("¡Ejercicio completado!")
print("Ahora controlas la generación de texto.")
