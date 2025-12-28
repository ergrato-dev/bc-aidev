# ✂️ Ejercicio 02: Tokenización

## 🎯 Objetivo

Implementar diferentes estrategias de tokenización y construir un vocabulario.

---

## 📋 Descripción

En este ejercicio aprenderás a dividir texto en tokens usando diferentes estrategias, desde simples splits hasta tokenizadores de NLTK y spaCy.

---

## 🔧 Pasos del Ejercicio

### Paso 1: Tokenización Simple

La forma más básica es dividir por espacios:

```python
text = "Hola mundo cruel"
tokens = text.split()
# ["Hola", "mundo", "cruel"]
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

### Paso 2: Tokenización con Regex

Para manejar mejor la puntuación:

```python
import re
text = "Hola, ¿cómo estás?"
tokens = re.findall(r'\b\w+\b', text)
# ["Hola", "cómo", "estás"]
```

### Paso 3: Tokenización con NLTK

NLTK ofrece tokenizadores más sofisticados:

```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize("Hola, ¿cómo estás?")
# ["Hola", ",", "¿", "cómo", "estás", "?"]
```

### Paso 4: Tokenización por Oraciones

Dividir texto en oraciones:

```python
from nltk.tokenize import sent_tokenize
text = "Hola. ¿Cómo estás? Bien, gracias."
sentences = sent_tokenize(text)
# ["Hola.", "¿Cómo estás?", "Bien, gracias."]
```

### Paso 5: Construir Vocabulario

Crear un mapeo de tokens a índices:

```python
from collections import Counter

def build_vocab(texts, min_freq=1):
    all_tokens = []
    for text in texts:
        all_tokens.extend(text.lower().split())
    
    counts = Counter(all_tokens)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    for token, count in counts.most_common():
        if count >= min_freq:
            vocab[token] = len(vocab)
    
    return vocab
```

### Paso 6: Codificar y Decodificar

Convertir entre texto e índices:

```python
def encode(text, vocab):
    tokens = text.lower().split()
    return [vocab.get(t, vocab['<UNK>']) for t in tokens]

def decode(ids, vocab):
    id_to_token = {v: k for k, v in vocab.items()}
    return [id_to_token.get(i, '<UNK>') for i in ids]
```

---

## 📁 Estructura

```
ejercicio-02-tokenizacion/
├── README.md
└── starter/
    └── main.py
```

---

## ▶️ Ejecución

```bash
cd bootcamp/04-especializacion/week-29/2-practicas/ejercicio-02-tokenizacion
python starter/main.py
```

---

## ✅ Criterios de Éxito

- [ ] Tokenización simple con split funciona
- [ ] Tokenización con regex maneja puntuación
- [ ] Vocabulario se construye correctamente
- [ ] Encode/decode funcionan sin errores
- [ ] Tokens OOV se mapean a `<UNK>`

---

## 🔗 Recursos

- [NLTK Tokenizers](https://www.nltk.org/api/nltk.tokenize.html)
- [spaCy Tokenization](https://spacy.io/usage/linguistic-features#tokenization)
