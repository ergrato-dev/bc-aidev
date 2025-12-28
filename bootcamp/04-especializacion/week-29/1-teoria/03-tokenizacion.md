# ✂️ Tokenización

![Tokenización](../0-assets/03-tokenization-types.svg)

## 🎯 Objetivos

- Comprender qué es la tokenización y su importancia
- Implementar diferentes estrategias de tokenización
- Conocer tokenizadores de subpalabras (BPE, WordPiece)

---

## 📋 ¿Qué es la Tokenización?

**Tokenización** es el proceso de dividir texto en unidades más pequeñas llamadas **tokens**.

```python
texto = "Me encanta aprender NLP"

# Tokenización por palabras
tokens = ["Me", "encanta", "aprender", "NLP"]

# Tokenización por caracteres  
tokens = ["M", "e", " ", "e", "n", "c", "a", "n", "t", "a", ...]
```

---

## 🔤 Tipos de Tokenización

### 1. Por Palabras (Word-level)

```python
# Simple: split por espacios
text = "Hola mundo cruel"
tokens = text.split()
# ["Hola", "mundo", "cruel"]

# Con NLTK (maneja puntuación)
from nltk.tokenize import word_tokenize
tokens = word_tokenize("Hola, ¿cómo estás?")
# ["Hola", ",", "¿", "cómo", "estás", "?"]
```

**Ventajas:**
- Intuitivo
- Preserva palabras completas

**Desventajas:**
- Vocabulario muy grande
- No maneja palabras nuevas (OOV)

### 2. Por Caracteres (Character-level)

```python
text = "Hola"
tokens = list(text)
# ["H", "o", "l", "a"]
```

**Ventajas:**
- Vocabulario pequeño (~100 caracteres)
- Sin problemas de OOV

**Desventajas:**
- Secuencias muy largas
- Pierde información semántica

### 3. Por Subpalabras (Subword)

Equilibrio entre palabras y caracteres.

```python
# Ejemplo conceptual
text = "infelizmente"
tokens = ["in", "feliz", "mente"]

text = "transformers"  
tokens = ["trans", "form", "ers"]
```

---

## 🧩 Algoritmos de Subpalabras

### BPE (Byte Pair Encoding)

Usado por GPT-2, GPT-3, RoBERTa.

**Algoritmo:**
1. Empezar con vocabulario de caracteres
2. Contar pares de tokens más frecuentes
3. Fusionar el par más frecuente
4. Repetir hasta alcanzar tamaño de vocabulario

```python
# Ejemplo simplificado
corpus = ["low", "lower", "newest", "widest"]

# Paso 1: Caracteres
vocab = {'l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'i', 'd'}

# Paso 2: Par más frecuente "es" → fusionar
vocab.add("es")
# "newest" → "new", "es", "t"

# Paso 3: Siguiente par "est" → fusionar
vocab.add("est")
# "newest" → "new", "est"
```

### WordPiece

Usado por BERT, DistilBERT.

Similar a BPE pero usa likelihood en lugar de frecuencia.

```python
# Tokens comienzan con ## si no son inicio de palabra
text = "unbelievable"
tokens = ["un", "##believ", "##able"]

text = "playing"
tokens = ["play", "##ing"]
```

### SentencePiece

Trata el texto como secuencia de caracteres, incluyendo espacios.

```python
# Espacio se representa con ▁
text = "Hello world"
tokens = ["▁Hello", "▁world"]
```

---

## 🛠️ Implementación Práctica

### Tokenizador Simple

```python
import re
from typing import List

class SimpleTokenizer:
    """Tokenizador básico por palabras."""
    
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
    
    def tokenize(self, text: str) -> List[str]:
        """Tokeniza texto en palabras."""
        if self.lowercase:
            text = text.lower()
        # Separar por espacios y puntuación
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """Reconstruye texto desde tokens."""
        return ' '.join(tokens)

# Uso
tokenizer = SimpleTokenizer()
tokens = tokenizer.tokenize("Hola, ¿cómo estás?")
# ["hola", "cómo", "estás"]
```

### Usando Hugging Face Tokenizers

```python
from transformers import AutoTokenizer

# Cargar tokenizer pre-entrenado
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello, how are you doing today?"
tokens = tokenizer.tokenize(text)
# ['hello', ',', 'how', 'are', 'you', 'doing', 'today', '?']

# Convertir a IDs
ids = tokenizer.encode(text)
# [101, 7592, 1010, 2129, 2024, 2017, 2725, 2651, 1029, 102]

# Decodificar
decoded = tokenizer.decode(ids)
# "[CLS] hello, how are you doing today? [SEP]"
```

---

## 🔢 Vocabulario y OOV

### Construir Vocabulario

```python
from collections import Counter

def build_vocab(texts: List[str], min_freq: int = 2) -> dict:
    """Construye vocabulario desde corpus."""
    # Tokenizar y contar
    all_tokens = []
    for text in texts:
        all_tokens.extend(text.lower().split())
    
    counts = Counter(all_tokens)
    
    # Tokens especiales
    vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
    
    # Añadir tokens frecuentes
    for token, count in counts.most_common():
        if count >= min_freq:
            vocab[token] = len(vocab)
    
    return vocab

# Uso
texts = ["hola mundo", "hola amigo", "adiós mundo"]
vocab = build_vocab(texts, min_freq=1)
# {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3, 
#  'hola': 4, 'mundo': 5, 'amigo': 6, 'adiós': 7}
```

### Manejo de OOV

```python
def encode(text: str, vocab: dict) -> List[int]:
    """Convierte texto a IDs usando vocabulario."""
    tokens = text.lower().split()
    unk_id = vocab['<UNK>']
    return [vocab.get(token, unk_id) for token in tokens]

text = "hola extraño mundo"
ids = encode(text, vocab)
# [4, 1, 5]  # "extraño" → <UNK> (1)
```

---

## 📊 Comparación de Estrategias

| Estrategia | Vocab Size | Seq Length | OOV | Ejemplo |
|------------|------------|------------|-----|---------|
| Palabra | 50K-100K+ | Corta | ❌ Sí | GPT-1 |
| Carácter | ~100 | Muy larga | ✅ No | Algunos RNN |
| Subpalabra | 30K-50K | Media | ✅ No | BERT, GPT-2+ |

---

## ✅ Checklist de Verificación

- [ ] Entiendo los diferentes tipos de tokenización
- [ ] Conozco BPE y WordPiece
- [ ] Puedo construir un vocabulario básico
- [ ] Sé manejar tokens fuera de vocabulario (OOV)

---

_Siguiente: [Word Embeddings](04-word-embeddings.md)_
