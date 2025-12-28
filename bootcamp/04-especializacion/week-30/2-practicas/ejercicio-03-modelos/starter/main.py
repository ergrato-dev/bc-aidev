"""
Ejercicio 03: Modelos Pre-entrenados
====================================

Aprende a cargar y usar modelos para inferencia manual.

Instrucciones:
1. Lee cada sección y descomenta el código
2. Ejecuta el script después de cada paso
3. Observa la estructura de los outputs

Nota: Se descargarán varios modelos (~500MB+)
"""

import torch

# ============================================
# PASO 1: Cargar Modelo de Clasificación
# ============================================
print("--- Paso 1: Cargar Modelo de Clasificación ---")

# AutoModelForSequenceClassification: clasificar texto completo

# Descomenta las siguientes líneas:
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
#
# print(f'Cargando {model_name}...')
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
#
# print(f'✓ Modelo cargado')
# print(f'  Parámetros: {model.num_parameters():,}')
# print(f'  Etiquetas: {model.config.id2label}')

print()


# ============================================
# PASO 2: Inferencia Paso a Paso
# ============================================
print("--- Paso 2: Inferencia Paso a Paso ---")

# Descomenta las siguientes líneas:
# text = "I absolutely love this movie!"
#
# # Paso 1: Tokenizar
# inputs = tokenizer(text, return_tensors='pt')
# print(f'Input keys: {inputs.keys()}')
# print(f'input_ids shape: {inputs["input_ids"].shape}')
#
# # Paso 2: Inferencia (sin gradientes)
# model.eval()
# with torch.no_grad():
#     outputs = model(**inputs)
#
# # Paso 3: Examinar outputs
# print(f'\nOutput type: {type(outputs).__name__}')
# print(f'Logits shape: {outputs.logits.shape}')
# print(f'Logits: {outputs.logits}')
#
# # Paso 4: Convertir a probabilidades
# probs = torch.softmax(outputs.logits, dim=-1)
# print(f'\nProbabilidades: {probs}')
#
# # Paso 5: Obtener predicción
# pred_id = torch.argmax(probs, dim=-1).item()
# label = model.config.id2label[pred_id]
# confidence = probs[0][pred_id].item()
#
# print(f'\nPredicción: {label} ({confidence:.2%})')

print()


# ============================================
# PASO 3: Clasificar Múltiples Textos
# ============================================
print("--- Paso 3: Clasificar Múltiples Textos ---")

# Descomenta las siguientes líneas:
# texts = [
#     "This is the best day ever!",
#     "I hate this terrible weather.",
#     "The movie was okay, nothing special.",
#     "Absolutely fantastic experience!",
# ]
#
# # Tokenizar batch
# inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
#
# # Inferencia
# with torch.no_grad():
#     outputs = model(**inputs)
#
# # Procesar resultados
# probs = torch.softmax(outputs.logits, dim=-1)
# preds = torch.argmax(probs, dim=-1)
#
# print('Clasificación de textos:')
# for i, text in enumerate(texts):
#     label = model.config.id2label[preds[i].item()]
#     conf = probs[i][preds[i]].item()
#     emoji = '😊' if label == 'POSITIVE' else '😞'
#     print(f'  {emoji} {label:8} ({conf:.2%}) | {text}')

print()


# ============================================
# PASO 4: Token Classification (NER)
# ============================================
print("--- Paso 4: Token Classification (NER) ---")

# AutoModelForTokenClassification: clasificar cada token

# Descomenta las siguientes líneas:
# from transformers import AutoModelForTokenClassification
#
# ner_model_name = 'dslim/bert-base-NER'
# print(f'Cargando {ner_model_name}...')
#
# ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
# ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
#
# text = "Steve Jobs founded Apple in California"
# inputs = ner_tokenizer(text, return_tensors='pt')
#
# with torch.no_grad():
#     outputs = ner_model(**inputs)
#
# # Obtener predicciones por token
# predictions = torch.argmax(outputs.logits, dim=-1)[0]
# tokens = ner_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
#
# print(f'\nTexto: "{text}"')
# print('\nEntidades:')
# for token, pred_id in zip(tokens, predictions):
#     label = ner_model.config.id2label[pred_id.item()]
#     if label != 'O':  # O = Outside (no es entidad)
#         print(f'  {token:15} → {label}')

print()


# ============================================
# PASO 5: Question Answering
# ============================================
print("--- Paso 5: Question Answering ---")

# AutoModelForQuestionAnswering: encontrar respuesta en contexto

# Descomenta las siguientes líneas:
# from transformers import AutoModelForQuestionAnswering
#
# qa_model_name = 'distilbert-base-uncased-distilled-squad'
# print(f'Cargando {qa_model_name}...')
#
# qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
# qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
#
# question = "What is the capital of France?"
# context = "Paris is the capital and largest city of France. It is located in the north of the country."
#
# inputs = qa_tokenizer(question, context, return_tensors='pt')
#
# with torch.no_grad():
#     outputs = qa_model(**inputs)
#
# # Encontrar posiciones de inicio y fin
# start_idx = torch.argmax(outputs.start_logits)
# end_idx = torch.argmax(outputs.end_logits)
#
# # Extraer respuesta
# answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
# answer = qa_tokenizer.decode(answer_tokens)
#
# print(f'\nPregunta: {question}')
# print(f'Contexto: {context}')
# print(f'Respuesta: {answer}')

print()


# ============================================
# PASO 6: Generación de Texto (Causal LM)
# ============================================
print("--- Paso 6: Generación de Texto ---")

# AutoModelForCausalLM: modelos autoregresivos como GPT

# Descomenta las siguientes líneas:
# from transformers import AutoModelForCausalLM
#
# gen_model_name = 'gpt2'
# print(f'Cargando {gen_model_name}...')
#
# gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
# gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)
#
# # GPT-2 necesita pad_token
# gen_tokenizer.pad_token = gen_tokenizer.eos_token
#
# prompt = "The future of artificial intelligence"
# inputs = gen_tokenizer(prompt, return_tensors='pt')
#
# # Generar
# with torch.no_grad():
#     outputs = gen_model.generate(
#         **inputs,
#         max_new_tokens=30,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9,
#         pad_token_id=gen_tokenizer.eos_token_id
#     )
#
# generated = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(f'\nPrompt: "{prompt}"')
# print(f'Generated: {generated}')

print()


# ============================================
# PASO 7: Modelo Base (sin cabeza)
# ============================================
print("--- Paso 7: Modelo Base ---")

# AutoModel: solo encoder, sin cabeza de clasificación

# Descomenta las siguientes líneas:
# from transformers import AutoModel
#
# base_model_name = 'bert-base-uncased'
# print(f'Cargando {base_model_name} (base)...')
#
# base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# base_model = AutoModel.from_pretrained(base_model_name)
#
# text = "Hello, this is a test."
# inputs = base_tokenizer(text, return_tensors='pt')
#
# with torch.no_grad():
#     outputs = base_model(**inputs)
#
# print(f'\nOutputs:')
# print(f'  last_hidden_state shape: {outputs.last_hidden_state.shape}')
# print(f'  → (batch_size, seq_len, hidden_size)')
#
# # El embedding de [CLS] se usa como representación de la oración
# cls_embedding = outputs.last_hidden_state[0, 0, :]
# print(f'\n[CLS] embedding shape: {cls_embedding.shape}')
# print(f'Primeros 10 valores: {cls_embedding[:10]}')

print()


# ============================================
# PASO 8: GPU/CPU
# ============================================
print("--- Paso 8: GPU/CPU ---")

# Descomenta las siguientes líneas:
# # Verificar disponibilidad de GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Dispositivo: {device}')
#
# if torch.cuda.is_available():
#     print(f'GPU: {torch.cuda.get_device_name(0)}')
#     print(f'Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
#
# # Mover modelo a GPU (si disponible)
# # model = model.to(device)
# # inputs = {k: v.to(device) for k, v in inputs.items()}

print()
print("=" * 50)
print("¡Ejercicio completado!")
print("Ahora sabes usar modelos pre-entrenados manualmente.")
