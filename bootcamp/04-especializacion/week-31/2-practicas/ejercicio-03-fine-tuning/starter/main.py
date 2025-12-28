"""
Ejercicio 03: Fine-tuning con LoRA
==================================

Aprende a configurar y ejecutar fine-tuning eficiente.

Instrucciones:
1. Lee cada sección y descomenta el código
2. Este ejercicio es principalmente conceptual sin GPU
3. El código está preparado para ejecutar con GPU

Nota: Fine-tuning real requiere GPU. Aquí vemos la configuración.
"""

import torch

# ============================================
# PASO 1: Setup y Verificación
# ============================================
print("--- Paso 1: Setup ---")

# Descomenta las siguientes líneas:
# import warnings
# warnings.filterwarnings('ignore')
#
# # Verificar GPU
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'Dispositivo: {device}')
#
# if torch.cuda.is_available():
#     print(f'GPU: {torch.cuda.get_device_name(0)}')
#     print(f'Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
# else:
#     print('⚠️  Sin GPU. Fine-tuning será lento o imposible.')
#     print('   Este ejercicio muestra la configuración conceptual.')

print()


# ============================================
# PASO 2: Cargar Modelo Base
# ============================================
print("--- Paso 2: Cargar Modelo Base ---")

# Descomenta las siguientes líneas:
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# # Usamos GPT-2 pequeño para demo
# model_name = 'gpt2'
# print(f'Cargando {model_name}...')
#
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
#
# # Configurar pad token
# tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = tokenizer.pad_token_id
#
# # Contar parámetros
# total_params = sum(p.numel() for p in model.parameters())
# print(f'✓ Modelo cargado: {total_params:,} parámetros')

print()


# ============================================
# PASO 3: Configurar LoRA
# ============================================
print("--- Paso 3: Configurar LoRA ---")

# Descomenta las siguientes líneas:
# from peft import LoraConfig, get_peft_model, TaskType
#
# # Configuración LoRA
# lora_config = LoraConfig(
#     r=8,                      # Rank de las matrices
#     lora_alpha=16,            # Scaling factor (alpha/r)
#     target_modules=['c_attn', 'c_proj'],  # Módulos a adaptar (GPT-2)
#     lora_dropout=0.05,        # Dropout para regularización
#     bias='none',              # No entrenar biases
#     task_type=TaskType.CAUSAL_LM  # Tipo de tarea
# )
#
# print('Configuración LoRA:')
# print(f'  r (rank): {lora_config.r}')
# print(f'  alpha: {lora_config.lora_alpha}')
# print(f'  target_modules: {lora_config.target_modules}')
# print(f'  dropout: {lora_config.lora_dropout}')

print()


# ============================================
# PASO 4: Aplicar LoRA al Modelo
# ============================================
print("--- Paso 4: Aplicar LoRA ---")

# Descomenta las siguientes líneas:
# # Aplicar LoRA
# model = get_peft_model(model, lora_config)
#
# # Ver parámetros entrenables
# model.print_trainable_parameters()
#
# # Desglose
# trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# total = sum(p.numel() for p in model.parameters())
# print(f'\nDesglose:')
# print(f'  Entrenables: {trainable:,}')
# print(f'  Total: {total:,}')
# print(f'  Porcentaje: {trainable/total*100:.4f}%')

print()


# ============================================
# PASO 5: Preparar Dataset
# ============================================
print("--- Paso 5: Preparar Dataset ---")

# Descomenta las siguientes líneas:
# # Dataset de ejemplo (instrucciones simples)
# training_data = [
#     {
#         'instruction': 'Traduce al inglés: Hola mundo',
#         'output': 'Hello world'
#     },
#     {
#         'instruction': 'Traduce al inglés: Buenos días',
#         'output': 'Good morning'
#     },
#     {
#         'instruction': 'Traduce al inglés: Gracias',
#         'output': 'Thank you'
#     },
#     {
#         'instruction': 'Traduce al inglés: Por favor',
#         'output': 'Please'
#     },
#     {
#         'instruction': 'Traduce al inglés: ¿Cómo estás?',
#         'output': 'How are you?'
#     },
# ]
#
# def format_prompt(example: dict) -> str:
#     """Formatea un ejemplo para entrenamiento."""
#     return f"""### Instruction:
# {example['instruction']}
#
# ### Response:
# {example['output']}"""
#
# # Mostrar ejemplo
# print('Ejemplo formateado:')
# print(format_prompt(training_data[0]))

print()


# ============================================
# PASO 6: Tokenizar Dataset
# ============================================
print("--- Paso 6: Tokenizar Dataset ---")

# Descomenta las siguientes líneas:
# from datasets import Dataset
#
# def tokenize_example(example: dict) -> dict:
#     """Tokeniza un ejemplo para entrenamiento."""
#     prompt = format_prompt(example)
#
#     # Tokenizar
#     tokens = tokenizer(
#         prompt,
#         truncation=True,
#         max_length=128,
#         padding='max_length',
#         return_tensors=None
#     )
#
#     # Para Causal LM, labels = input_ids
#     tokens['labels'] = tokens['input_ids'].copy()
#
#     return tokens
#
# # Crear dataset
# dataset = Dataset.from_list(training_data)
#
# # Tokenizar
# tokenized_dataset = dataset.map(
#     tokenize_example,
#     remove_columns=dataset.column_names
# )
#
# print(f'Dataset tokenizado: {len(tokenized_dataset)} ejemplos')
# print(f'Columnas: {tokenized_dataset.column_names}')
# print(f'Forma input_ids[0]: {len(tokenized_dataset[0]["input_ids"])}')

print()


# ============================================
# PASO 7: Configurar Entrenamiento
# ============================================
print("--- Paso 7: Configurar Entrenamiento ---")

# Descomenta las siguientes líneas:
# from transformers import TrainingArguments
#
# training_args = TrainingArguments(
#     output_dir='./lora-output',
#
#     # Épocas y batch
#     num_train_epochs=3,
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=2,
#
#     # Learning rate
#     learning_rate=2e-4,
#     warmup_ratio=0.03,
#     lr_scheduler_type='cosine',
#
#     # Logging
#     logging_steps=5,
#     logging_dir='./logs',
#
#     # Guardado
#     save_strategy='epoch',
#     save_total_limit=2,
#
#     # Optimización
#     fp16=torch.cuda.is_available(),
#     optim='adamw_torch',
#
#     # Otros
#     report_to='none',
#     remove_unused_columns=False,
# )
#
# print('Training Arguments:')
# print(f'  Épocas: {training_args.num_train_epochs}')
# print(f'  Batch size: {training_args.per_device_train_batch_size}')
# print(f'  Learning rate: {training_args.learning_rate}')
# print(f'  FP16: {training_args.fp16}')

print()


# ============================================
# PASO 8: Crear Trainer
# ============================================
print("--- Paso 8: Crear Trainer ---")

# Descomenta las siguientes líneas:
# from transformers import Trainer, DataCollatorForLanguageModeling
#
# # Data collator para causal LM
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False  # False para causal LM
# )
#
# # Crear trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     data_collator=data_collator,
#     tokenizer=tokenizer,
# )
#
# print('✓ Trainer creado')
# print(f'  Dataset: {len(trainer.train_dataset)} ejemplos')

print()


# ============================================
# PASO 9: Entrenar (si hay GPU)
# ============================================
print("--- Paso 9: Entrenar ---")

# Descomenta las siguientes líneas:
# if torch.cuda.is_available():
#     print('Iniciando entrenamiento...')
#     trainer.train()
#     print('✓ Entrenamiento completado')
# else:
#     print('⚠️  Sin GPU. Saltando entrenamiento.')
#     print('   En producción, ejecutarías: trainer.train()')

print()


# ============================================
# PASO 10: Guardar y Cargar Adaptadores
# ============================================
print("--- Paso 10: Guardar y Cargar ---")

# Descomenta las siguientes líneas:
# # Guardar adaptadores LoRA (muy pequeños ~MB)
# adapter_path = './my-lora-adapter'
#
# # Nota: Descomentar si entrenaste
# # model.save_pretrained(adapter_path)
# # tokenizer.save_pretrained(adapter_path)
# # print(f'✓ Adaptadores guardados en {adapter_path}')
#
# # Para cargar después:
# print('\nCódigo para cargar adaptadores:')
# print('''
# from peft import PeftModel
#
# # Cargar modelo base
# base_model = AutoModelForCausalLM.from_pretrained('gpt2')
#
# # Cargar adaptadores LoRA
# model = PeftModel.from_pretrained(base_model, './my-lora-adapter')
#
# # Usar para inferencia
# model.eval()
# ''')

print()


# ============================================
# PASO 11: Inferencia con Modelo Fine-tuned
# ============================================
print("--- Paso 11: Inferencia ---")

# Descomenta las siguientes líneas:
# def generate_response(instruction: str) -> str:
#     """Genera respuesta dado una instrucción."""
#     prompt = f"""### Instruction:
# {instruction}
#
# ### Response:"""
#
#     inputs = tokenizer(prompt, return_tensors='pt')
#
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs['input_ids'],
#             max_new_tokens=50,
#             temperature=0.7,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id
#         )
#
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # Extraer solo la respuesta
#     if '### Response:' in response:
#         response = response.split('### Response:')[1].strip()
#     return response
#
# # Probar
# print('Probando modelo (sin fine-tuning real):')
# test_instruction = 'Traduce al inglés: Buenas noches'
# response = generate_response(test_instruction)
# print(f'  Instrucción: {test_instruction}')
# print(f'  Respuesta: {response[:50]}...')

print()


# ============================================
# RESUMEN
# ============================================
print("=" * 50)
print("RESUMEN - Pasos de Fine-tuning con LoRA")
print("=" * 50)
print(
    """
1. Cargar modelo base pre-entrenado
2. Configurar LoRA (r, alpha, target_modules)
3. Aplicar PEFT al modelo
4. Preparar y tokenizar dataset
5. Configurar TrainingArguments
6. Crear Trainer
7. Entrenar: trainer.train()
8. Guardar adaptadores (~MB vs GB del modelo completo)
9. Cargar con PeftModel.from_pretrained()
10. Inferencia normal

Ventajas de LoRA:
✓ Entrena ~0.1% de parámetros
✓ Adaptadores muy pequeños (~30MB)
✓ Menos overfitting
✓ Múltiples adaptadores para un modelo base
"""
)
print("¡Ejercicio completado!")
