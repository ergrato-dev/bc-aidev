# 🔤 Ejercicio 02: Tokenizers de Hugging Face

## 🎯 Objetivo

Dominar el uso de tokenizers: cargar, tokenizar, manejar padding y attention masks.

---

## 📋 Descripción

En este ejercicio aprenderás a usar AutoTokenizer, entender el proceso de tokenización, manejar padding y truncation, y trabajar con tokens especiales.

---

## 🔧 Pasos del Ejercicio

### Paso 1: Cargar Tokenizer

Usar AutoTokenizer para cargar cualquier tokenizer:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

### Paso 2: Tokenización Básica

Diferentes métodos de tokenización:

```python
# Solo tokens (strings)
tokens = tokenizer.tokenize("Hello world")

# Tokens + IDs
ids = tokenizer.encode("Hello world")

# Encoding completo (dict)
encoding = tokenizer("Hello world")
```

### Paso 3: Decodificación

Convertir IDs de vuelta a texto:

```python
text = tokenizer.decode([101, 7592, 2088, 102])
```

### Paso 4: Padding y Truncation

Manejar secuencias de diferentes longitudes:

```python
encoding = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
```

### Paso 5: Attention Mask

Entender qué tokens son reales vs padding:

```python
# attention_mask: 1 = token real, 0 = padding
print(encoding['attention_mask'])
```

### Paso 6: Tokens Especiales

Conocer los tokens especiales del modelo:

```python
print(tokenizer.cls_token)  # [CLS]
print(tokenizer.sep_token)  # [SEP]
print(tokenizer.pad_token)  # [PAD]
```

---

## 📁 Estructura

```
ejercicio-02-tokenizers/
├── README.md
└── starter/
    └── main.py
```

---

## ▶️ Ejecución

```bash
cd bootcamp/04-especializacion/week-30/2-practicas/ejercicio-02-tokenizers
python starter/main.py
```

---

## ✅ Criterios de Éxito

- [ ] Puedo cargar tokenizers con AutoTokenizer
- [ ] Entiendo tokenize, encode, decode
- [ ] Sé aplicar padding y truncation
- [ ] Comprendo attention_mask
- [ ] Conozco los tokens especiales

---

## 🔗 Recursos

- [Tokenizers Documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer)
- [Preprocessing Data](https://huggingface.co/docs/transformers/preprocessing)
