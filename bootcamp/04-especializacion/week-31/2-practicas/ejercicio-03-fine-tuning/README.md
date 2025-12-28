# 🔧 Ejercicio 03: Fine-tuning con LoRA

## 🎯 Objetivo

Aprender a hacer fine-tuning eficiente usando LoRA y PEFT.

---

## 📋 Descripción

En este ejercicio configurarás LoRA, prepararás un dataset, y ejecutarás un ciclo de entrenamiento básico.

---

## 🔧 Requisitos

```bash
pip install transformers peft datasets accelerate bitsandbytes
```

**Nota**: Este ejercicio requiere GPU para entrenamiento real. En CPU solo veremos la configuración.

---

## 🔧 Pasos del Ejercicio

### Paso 1: Cargar Modelo Base

Cargar un modelo pequeño para fine-tuning:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

### Paso 2: Configurar LoRA

Definir qué capas adaptar:

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05
)
model = get_peft_model(model, config)
```

### Paso 3: Preparar Dataset

Formatear datos para entrenamiento:

```python
def format_example(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
```

### Paso 4: Configurar Training

Definir argumentos de entrenamiento:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4
)
```

### Paso 5: Entrenar

Ejecutar el entrenamiento:

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)
trainer.train()
```

### Paso 6: Guardar y Cargar Adaptadores

Guardar solo los pesos LoRA:

```python
model.save_pretrained("./my-lora-adapter")
```

---

## 📁 Estructura

```
ejercicio-03-fine-tuning/
├── README.md
└── starter/
    └── main.py
```

---

## ▶️ Ejecución

```bash
cd bootcamp/04-especializacion/week-31/2-practicas/ejercicio-03-fine-tuning
python starter/main.py
```

---

## ✅ Criterios de Éxito

- [ ] Entiendo la configuración de LoRA
- [ ] Sé preparar un dataset para fine-tuning
- [ ] Comprendo los TrainingArguments principales
- [ ] Puedo guardar y cargar adaptadores
- [ ] Entiendo parámetros entrenables vs congelados

---

## 🔗 Recursos

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [TRL Library](https://huggingface.co/docs/trl)
