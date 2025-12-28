"""
Ejercicio 02: Tokenizers de Hugging Face
========================================

Aprende a usar tokenizers: cargar, tokenizar, padding, attention masks.

Instrucciones:
1. Lee cada sección y descomenta el código
2. Ejecuta el script después de cada paso
3. Observa las diferencias entre métodos
"""

# ============================================
# PASO 1: Cargar Tokenizer
# ============================================
print("--- Paso 1: Cargar Tokenizer ---")

# AutoTokenizer detecta automáticamente el tipo de tokenizer

# Descomenta las siguientes líneas:
# from transformers import AutoTokenizer
#
# # Cargar tokenizer de BERT
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#
# print(f'Tokenizer cargado: {type(tokenizer).__name__}')
# print(f'Vocabulario: {tokenizer.vocab_size:,} tokens')
# print(f'Max length: {tokenizer.model_max_length}')

print()


# ============================================
# PASO 2: Métodos de Tokenización
# ============================================
print("--- Paso 2: Métodos de Tokenización ---")

# Hay varios métodos para tokenizar texto

# Descomenta las siguientes líneas:
# text = "Hello, how are you doing today?"
#
# # Método 1: tokenize() - Solo tokens (strings)
# tokens = tokenizer.tokenize(text)
# print(f'tokenize(): {tokens}')
#
# # Método 2: encode() - Tokens + IDs + tokens especiales
# ids = tokenizer.encode(text)
# print(f'encode(): {ids}')
#
# # Método 3: __call__() - Encoding completo (dict)
# encoding = tokenizer(text)
# print(f'__call__(): {encoding.keys()}')
# print(f'  input_ids: {encoding["input_ids"]}')
# print(f'  attention_mask: {encoding["attention_mask"]}')

print()


# ============================================
# PASO 3: Decodificación
# ============================================
print("--- Paso 3: Decodificación ---")

# Convertir IDs de vuelta a texto

# Descomenta las siguientes líneas:
# ids = [101, 7592, 1010, 2129, 2024, 2017, 2725, 2651, 1029, 102]
#
# # Decodificar con tokens especiales
# decoded = tokenizer.decode(ids)
# print(f'Con especiales: {decoded}')
#
# # Decodificar sin tokens especiales
# decoded_clean = tokenizer.decode(ids, skip_special_tokens=True)
# print(f'Sin especiales: {decoded_clean}')
#
# # Convertir IDs individuales a tokens
# tokens = tokenizer.convert_ids_to_tokens(ids)
# print(f'IDs a tokens: {tokens}')

print()


# ============================================
# PASO 4: Tokens Especiales
# ============================================
print("--- Paso 4: Tokens Especiales ---")

# Cada modelo tiene tokens especiales diferentes

# Descomenta las siguientes líneas:
# print('Tokens especiales de BERT:')
# print(f'  [CLS]: {tokenizer.cls_token} → ID {tokenizer.cls_token_id}')
# print(f'  [SEP]: {tokenizer.sep_token} → ID {tokenizer.sep_token_id}')
# print(f'  [PAD]: {tokenizer.pad_token} → ID {tokenizer.pad_token_id}')
# print(f'  [UNK]: {tokenizer.unk_token} → ID {tokenizer.unk_token_id}')
# print(f'  [MASK]: {tokenizer.mask_token} → ID {tokenizer.mask_token_id}')
#
# # Todos los especiales
# print(f'\nMapa completo: {tokenizer.special_tokens_map}')

print()


# ============================================
# PASO 5: Padding
# ============================================
print("--- Paso 5: Padding ---")

# Padding iguala longitudes de secuencias

# Descomenta las siguientes líneas:
# texts = [
#     "Short text",
#     "This is a much longer text that requires more tokens to encode"
# ]
#
# # Sin padding - diferentes longitudes
# print('Sin padding:')
# for text in texts:
#     enc = tokenizer(text)
#     print(f'  {len(enc["input_ids"]):2} tokens: {text[:30]}...')
#
# # Con padding - misma longitud
# encoding = tokenizer(texts, padding=True)
# print(f'\nCon padding=True:')
# for i, ids in enumerate(encoding['input_ids']):
#     print(f'  {len(ids)} tokens: {texts[i][:30]}...')
#
# # Padding a longitud específica
# encoding = tokenizer(texts, padding='max_length', max_length=20)
# print(f'\nCon max_length=20:')
# for i, ids in enumerate(encoding['input_ids']):
#     print(f'  {len(ids)} tokens')

print()


# ============================================
# PASO 6: Attention Mask
# ============================================
print("--- Paso 6: Attention Mask ---")

# attention_mask indica tokens reales (1) vs padding (0)

# Descomenta las siguientes líneas:
# texts = ["Hello", "Hello world, how are you?"]
#
# encoding = tokenizer(texts, padding=True)
#
# print('Input IDs y Attention Masks:')
# for i in range(len(texts)):
#     ids = encoding['input_ids'][i]
#     mask = encoding['attention_mask'][i]
#     tokens = tokenizer.convert_ids_to_tokens(ids)
#
#     print(f'\nTexto {i+1}: "{texts[i]}"')
#     print(f'  Tokens: {tokens}')
#     print(f'  IDs:    {ids}')
#     print(f'  Mask:   {mask}')
#     print(f'  → {sum(mask)} tokens reales, {len(mask) - sum(mask)} padding')

print()


# ============================================
# PASO 7: Truncation
# ============================================
print("--- Paso 7: Truncation ---")

# Truncation recorta textos muy largos

# Descomenta las siguientes líneas:
# long_text = "This is a very long text. " * 100
# print(f'Texto original: {len(long_text)} caracteres')
#
# # Sin truncation - puede ser muy largo
# enc_no_trunc = tokenizer(long_text)
# print(f'Sin truncation: {len(enc_no_trunc["input_ids"])} tokens')
#
# # Con truncation
# enc_trunc = tokenizer(long_text, truncation=True, max_length=50)
# print(f'Con truncation (max=50): {len(enc_trunc["input_ids"])} tokens')
#
# # Ver qué se conservó
# decoded = tokenizer.decode(enc_trunc['input_ids'], skip_special_tokens=True)
# print(f'Texto truncado: "{decoded[:80]}..."')

print()


# ============================================
# PASO 8: Return Tensors
# ============================================
print("--- Paso 8: Return Tensors ---")

# Convertir a tensores para el modelo

# Descomenta las siguientes líneas:
# import torch
#
# text = "Hello, world!"
#
# # Sin return_tensors - listas de Python
# enc_list = tokenizer(text)
# print(f'Sin return_tensors: {type(enc_list["input_ids"])}')
#
# # Con return_tensors="pt" - PyTorch tensors
# enc_pt = tokenizer(text, return_tensors='pt')
# print(f'Con return_tensors="pt": {type(enc_pt["input_ids"])}')
# print(f'Shape: {enc_pt["input_ids"].shape}')
#
# # Batch de textos
# texts = ["First text", "Second text"]
# batch = tokenizer(texts, padding=True, return_tensors='pt')
# print(f'\nBatch shape: {batch["input_ids"].shape}')

print()


# ============================================
# PASO 9: Tokenización de Pares
# ============================================
print("--- Paso 9: Tokenización de Pares ---")

# Para tareas como QA que necesitan dos textos

# Descomenta las siguientes líneas:
# question = "What is Python?"
# context = "Python is a programming language created by Guido van Rossum."
#
# # Tokenizar par de textos
# encoding = tokenizer(question, context, return_tensors='pt')
#
# print(f'Tokens: {tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])}')
# print(f'\ntoken_type_ids: {encoding["token_type_ids"][0].tolist()}')
# print('  → 0s para la pregunta, 1s para el contexto')

print()


# ============================================
# PASO 10: Comparar Tokenizers
# ============================================
print("--- Paso 10: Comparar Tokenizers ---")

# Diferentes modelos tienen diferentes tokenizers

# Descomenta las siguientes líneas:
# models = [
#     'bert-base-uncased',
#     'gpt2',
#     'roberta-base',
# ]
#
# text = "Hello, I'm learning about tokenizers!"
#
# print(f'Texto: "{text}"\n')
#
# for model_name in models:
#     tok = AutoTokenizer.from_pretrained(model_name)
#     tokens = tok.tokenize(text)
#     print(f'{model_name}:')
#     print(f'  Tokens: {tokens}')
#     print(f'  Count: {len(tokens)}\n')

print()
print("=" * 50)
print("¡Ejercicio completado!")
print("Ahora dominas los tokenizers de Hugging Face.")
