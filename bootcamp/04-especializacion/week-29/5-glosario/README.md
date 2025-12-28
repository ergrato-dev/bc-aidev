# 📖 Glosario - Semana 29: NLP Fundamentos

Términos clave de Procesamiento de Lenguaje Natural ordenados alfabéticamente.

---

## B

### BPE (Byte Pair Encoding)
Algoritmo de tokenización por subpalabras que fusiona iterativamente los pares de caracteres más frecuentes. Usado en GPT-2, GPT-3, RoBERTa.

```python
# Ejemplo conceptual
"lowest" → ["low", "est"]
"newer"  → ["new", "er"]
```

### Bag of Words (BoW)
Representación de texto que ignora el orden de las palabras, contando solo su frecuencia.

```python
"el gato come" → {"el": 1, "gato": 1, "come": 1}
```

---

## C

### Corpus
Colección de textos utilizada para entrenar o evaluar modelos de NLP.

### Cosine Similarity (Similaridad Coseno)
Medida de similaridad entre dos vectores basada en el ángulo entre ellos.

$$\text{cos}(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$$

```python
# Valores: 1 = idénticos, 0 = ortogonales, -1 = opuestos
similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

### CBOW (Continuous Bag of Words)
Arquitectura de Word2Vec que predice una palabra dado su contexto.

---

## D

### Dense Vector (Vector Denso)
Vector donde la mayoría de valores son no-cero. Los word embeddings son vectores densos.

```python
# Dense: [0.2, -0.5, 0.8, 0.1, -0.3, ...]
# vs Sparse: [0, 0, 1, 0, 0, 0, 0, ...]
```

---

## E

### Embedding
Representación vectorial densa de baja dimensionalidad que captura propiedades semánticas.

### Embedding Dimension
Número de valores en un vector de embedding. Común: 50, 100, 300 dimensiones.

---

## G

### GloVe (Global Vectors)
Algoritmo de word embeddings que combina estadísticas de co-ocurrencia global con predicción local. Desarrollado por Stanford.

---

## L

### Lemmatization (Lematización)
Reducir palabras a su forma base (lema) usando conocimiento lingüístico.

```python
"corriendo" → "correr"
"mejores"   → "bueno"
```

---

## N

### N-gram
Secuencia contigua de n elementos (palabras o caracteres) de un texto.

```python
# Texto: "el gato come"
# Unigrams (n=1): ["el", "gato", "come"]
# Bigrams (n=2): ["el gato", "gato come"]
# Trigrams (n=3): ["el gato come"]
```

### NLP (Natural Language Processing)
Campo de la IA que estudia la interacción entre computadoras y lenguaje humano.

### Normalization (Normalización)
Proceso de estandarizar texto (minúsculas, eliminar acentos, etc.).

---

## O

### OOV (Out of Vocabulary)
Palabras que no están en el vocabulario del modelo. Se mapean típicamente a un token especial `<UNK>`.

### One-Hot Encoding
Representación sparse donde cada palabra es un vector con un solo 1.

```python
vocab = ["gato", "perro", "casa"]
"gato"  → [1, 0, 0]
"perro" → [0, 1, 0]
"casa"  → [0, 0, 1]
```

---

## P

### Preprocessing (Preprocesamiento)
Limpieza y normalización de texto antes del análisis: minúsculas, eliminar puntuación, etc.

### POS Tagging (Part-of-Speech)
Etiquetar cada palabra con su categoría gramatical (sustantivo, verbo, adjetivo, etc.).

---

## S

### Semantic Similarity (Similaridad Semántica)
Medida de cuán similares son dos textos en significado, no solo en palabras.

### Skip-gram
Arquitectura de Word2Vec que predice palabras del contexto dada una palabra central.

### Sparse Vector (Vector Disperso)
Vector donde la mayoría de valores son cero. One-hot encoding produce vectores sparse.

### Stemming
Reducir palabras a su raíz eliminando sufijos, sin considerar el contexto.

```python
"corriendo" → "corr"
"corredor"  → "corr"
```

### Stopwords (Palabras Vacías)
Palabras muy frecuentes con poco valor semántico (el, la, de, que, etc.).

```python
from nltk.corpus import stopwords
spanish_stops = stopwords.words('spanish')
# ["de", "la", "que", "el", "en", ...]
```

---

## T

### TF-IDF (Term Frequency-Inverse Document Frequency)
Medida que pondera la importancia de una palabra en un documento relativo a un corpus.

$$\text{TF-IDF} = \text{TF}(t,d) \times \log\frac{N}{\text{DF}(t)}$$

### Token
Unidad básica de texto después de tokenización (palabra, subpalabra, o carácter).

### Tokenization (Tokenización)
Proceso de dividir texto en unidades más pequeñas (tokens).

```python
"Hola mundo" → ["Hola", "mundo"]
```

---

## V

### Vocabulary (Vocabulario)
Conjunto de todos los tokens únicos conocidos por un modelo.

```python
vocab = {"<PAD>": 0, "<UNK>": 1, "gato": 2, "perro": 3}
```

---

## W

### Word2Vec
Familia de modelos para generar word embeddings. Incluye Skip-gram y CBOW. Desarrollado por Google (2013).

### Word Embedding
Representación vectorial densa de una palabra que captura su significado semántico.

### WordPiece
Algoritmo de tokenización por subpalabras usado en BERT. Similar a BPE pero usa likelihood.

```python
"unbelievable" → ["un", "##believ", "##able"]
```

---

## 📊 Resumen de Dimensiones Típicas

| Modelo | Dimensiones | Vocabulario |
|--------|-------------|-------------|
| Word2Vec (small) | 100 | ~3M |
| GloVe (6B) | 50, 100, 200, 300 | 400K |
| FastText | 300 | 2M |
| BERT tokenizer | - | 30K tokens |

---

## 🔗 Referencias

- [Stanford NLP Glossary](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html)
- [Hugging Face Glossary](https://huggingface.co/docs/transformers/glossary)
