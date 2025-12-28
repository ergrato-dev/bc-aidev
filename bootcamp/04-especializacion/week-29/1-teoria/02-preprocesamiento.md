# 🧹 Preprocesamiento de Texto

![Preprocesamiento](../0-assets/02-preprocessing-pipeline.svg)

## 🎯 Objetivos

- Implementar técnicas de limpieza de texto
- Aplicar normalización y estandarización
- Crear pipelines de preprocesamiento reutilizables

---

## 📋 ¿Por qué Preprocesar?

El texto crudo contiene mucho "ruido" que dificulta el procesamiento:

```python
# Texto crudo de redes sociales
raw_text = """
@usuario123 Esto es INCREÍBLE!!! 😍😍😍
Visita https://ejemplo.com para más info...
#python #datascience #ia
"""

# Después de preprocesar
clean_text = "esto es increíble visita para más info python datascience ia"
```

---

## 🔧 Técnicas de Preprocesamiento

### 1. Conversión a Minúsculas

```python
text = "Python es GENIAL para NLP"
text_lower = text.lower()
# "python es genial para nlp"
```

**Cuándo NO usar:**
- NER (nombres propios importan)
- Acrónimos con significado (USA, FBI)

### 2. Eliminación de Caracteres Especiales

```python
import re

def remove_special_chars(text: str) -> str:
    """Elimina caracteres especiales manteniendo espacios."""
    # Mantener solo letras, números y espacios
    return re.sub(r'[^a-záéíóúñü\s]', '', text.lower())

text = "¡Hola! ¿Cómo estás? 😊"
clean = remove_special_chars(text)
# "hola cómo estás"
```

### 3. Eliminación de URLs y Menciones

```python
def remove_urls(text: str) -> str:
    """Elimina URLs del texto."""
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_mentions(text: str) -> str:
    """Elimina menciones de redes sociales."""
    return re.sub(r'@\w+', '', text)

def remove_hashtags(text: str) -> str:
    """Elimina hashtags."""
    return re.sub(r'#\w+', '', text)
```

### 4. Eliminación de Números

```python
def remove_numbers(text: str) -> str:
    """Elimina números del texto."""
    return re.sub(r'\d+', '', text)

# O reemplazar por token especial
def replace_numbers(text: str) -> str:
    """Reemplaza números por <NUM>."""
    return re.sub(r'\d+', '<NUM>', text)
```

### 5. Eliminación de Espacios Extra

```python
def normalize_whitespace(text: str) -> str:
    """Normaliza espacios en blanco."""
    return ' '.join(text.split())

text = "Hola    mundo   cruel"
clean = normalize_whitespace(text)
# "Hola mundo cruel"
```

---

## 📝 Normalización de Texto

### Eliminación de Acentos (Opcional)

```python
import unicodedata

def remove_accents(text: str) -> str:
    """Elimina acentos del texto."""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

text = "canción mañana niño"
clean = remove_accents(text)
# "cancion manana nino"
```

**Nota:** Perder acentos puede cambiar significado (año vs ano).

### Lematización vs Stemming

**Stemming** - Cortar sufijos (rápido, impreciso)
```python
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('spanish')
words = ["corriendo", "corrí", "correr", "corremos"]
stems = [stemmer.stem(w) for w in words]
# ["corr", "corr", "corr", "corr"]
```

**Lematización** - Forma base real (lento, preciso)
```python
import spacy

nlp = spacy.load('es_core_news_sm')
doc = nlp("Los gatos están corriendo")
lemmas = [token.lemma_ for token in doc]
# ["el", "gato", "estar", "correr"]
```

---

## 🛑 Stopwords

Palabras muy frecuentes con poco valor semántico.

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('spanish'))
# {'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', ...}

def remove_stopwords(text: str) -> str:
    """Elimina stopwords del texto."""
    words = text.split()
    return ' '.join(w for w in words if w.lower() not in stop_words)

text = "el gato de la casa está en el jardín"
clean = remove_stopwords(text)
# "gato casa está jardín"
```

**Cuándo mantener stopwords:**
- Análisis de sentimiento ("no me gusta" vs "me gusta")
- Modelos de lenguaje
- Cuando el contexto importa

---

## 🔄 Pipeline Completo

```python
import re
from typing import Callable

def create_preprocessing_pipeline(
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    remove_special: bool = True,
    remove_numbers: bool = False,
    remove_stopwords: bool = False
) -> Callable[[str], str]:
    """
    Crea un pipeline de preprocesamiento configurable.
    
    Returns:
        Función que preprocesa texto
    """
    def preprocess(text: str) -> str:
        if remove_urls:
            text = re.sub(r'https?://\S+', '', text)
        
        if remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        if lowercase:
            text = text.lower()
        
        if remove_special:
            text = re.sub(r'[^a-záéíóúñü0-9\s]', '', text)
        
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Normalizar espacios
        text = ' '.join(text.split())
        
        return text
    
    return preprocess

# Uso
preprocess = create_preprocessing_pipeline(
    lowercase=True,
    remove_urls=True,
    remove_special=True
)

raw = "@user Esto es GENIAL! https://t.co/xxx 🎉"
clean = preprocess(raw)
# "esto es genial"
```

---

## ⚠️ Consideraciones

### Preservar Información Útil

```python
# A veces el "ruido" es información
"😍😍😍"  # → Sentimiento muy positivo
"!!!"     # → Énfasis
"jajaja"  # → Humor/sarcasmo
```

### Idioma-Específico

```python
# Español: ñ, acentos
# Alemán: ß, umlauts
# Chino: sin espacios entre palabras
```

### Dominio-Específico

```python
# Médico: mantener términos técnicos
# Legal: preservar formato específico
# Social media: emojis pueden ser importantes
```

---

## ✅ Checklist de Verificación

- [ ] Puedo implementar limpieza básica de texto
- [ ] Entiendo la diferencia entre stemming y lematización
- [ ] Sé cuándo usar o no usar stopwords
- [ ] Puedo crear pipelines de preprocesamiento configurables

---

_Siguiente: [Tokenización](03-tokenizacion.md)_
