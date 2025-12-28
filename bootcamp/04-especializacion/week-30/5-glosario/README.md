# 📖 Glosario - Semana 30: Hugging Face Transformers

## A

### Attention (Atención)
Mecanismo que permite al modelo enfocarse en partes relevantes de la entrada. La fórmula del attention es:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Donde Q (Query), K (Key), V (Value) son proyecciones lineales del input.

### AutoModel
Clase de Hugging Face que detecta y carga automáticamente la arquitectura correcta basándose en el checkpoint.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
```

### AutoTokenizer
Clase que carga automáticamente el tokenizer apropiado para un modelo específico.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

---

## B

### BERT (Bidirectional Encoder Representations from Transformers)
Modelo encoder-only pre-entrenado de forma bidireccional. Usa MLM (Masked Language Model) y NSP (Next Sentence Prediction).

### BPE (Byte Pair Encoding)
Algoritmo de tokenización que encuentra subwords frecuentes iterativamente. Usado por GPT-2, RoBERTa.

```
"unhappiness" → ["un", "happiness"] o ["un", "happ", "iness"]
```

### Batch
Grupo de ejemplos procesados juntos. Mejora eficiencia computacional.

```python
inputs = tokenizer(["text1", "text2"], padding=True, return_tensors="pt")
```

---

## C

### Causal Language Modeling (CLM)
Tarea de predecir el siguiente token basándose solo en tokens anteriores. Usado por GPT.

### Checkpoint
Punto de guardado de un modelo entrenado. Incluye pesos, configuración y tokenizer.

### [CLS] Token
Token especial al inicio de la secuencia en BERT. Su representación final se usa para clasificación.

---

## D

### Decoder
Componente que genera output secuencialmente. En Transformers puros, usa masked self-attention.

**Modelos decoder-only**: GPT, GPT-2, LLaMA

### DistilBERT
Versión destilada de BERT con 40% menos parámetros pero 97% del rendimiento.

---

## E

### Embedding
Representación vectorial densa de tokens. Transforma IDs discretos a vectores continuos.

$$\text{token\_id} \rightarrow \mathbf{e} \in \mathbb{R}^{d_{model}}$$

### Encoder
Componente que procesa la entrada completa bidireccionalmente. Genera representaciones contextuales.

**Modelos encoder-only**: BERT, RoBERTa, DistilBERT

---

## F

### Feature Extraction
Usar un modelo pre-entrenado para obtener representaciones de texto sin fine-tuning.

```python
from transformers import pipeline
fe = pipeline("feature-extraction")
embeddings = fe("Hello world")
```

### Fine-tuning
Proceso de entrenar un modelo pre-entrenado en una tarea específica con datos etiquetados.

---

## G

### GPT (Generative Pre-trained Transformer)
Familia de modelos autoregresivos de OpenAI. Decoder-only, entrenados con CLM.

---

## H

### Head (Cabeza)
Capa adicional sobre el modelo base para tareas específicas:
- `ForSequenceClassification`: Clasificación de texto
- `ForTokenClassification`: NER, POS tagging
- `ForQuestionAnswering`: QA extractivo
- `ForCausalLM`: Generación de texto

### Hidden State
Representaciones intermedias en cada capa del modelo.

```python
outputs = model(**inputs, output_hidden_states=True)
hidden_states = outputs.hidden_states  # Tuple de tensores
```

### Hugging Face Hub
Repositorio central de modelos, datasets y spaces de la comunidad. https://huggingface.co/

---

## I

### Input IDs
Secuencia de enteros representando tokens. Entrada principal del modelo.

```python
input_ids = tokenizer.encode("Hello")  # [101, 7592, 102]
```

### Inference
Proceso de usar un modelo entrenado para hacer predicciones sin actualizar pesos.

```python
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
```

---

## L

### Logits
Scores sin normalizar que produce el modelo antes de softmax.

```python
probs = torch.softmax(outputs.logits, dim=-1)
```

---

## M

### Masked Language Modeling (MLM)
Tarea de pre-entrenamiento donde se predicen tokens enmascarados.

```
Input:  "The [MASK] is blue"
Output: "The sky is blue"
```

### Model Hub
Ver "Hugging Face Hub".

### Multi-Head Attention
Múltiples mecanismos de attention en paralelo, cada uno aprendiendo diferentes patrones.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

---

## N

### Named Entity Recognition (NER)
Tarea de identificar y clasificar entidades nombradas en texto.

```
"Apple CEO Tim Cook" → [ORG: Apple, PER: Tim Cook]
```

---

## P

### Padding
Añadir tokens especiales para igualar longitudes de secuencias en un batch.

```python
tokenizer(texts, padding=True)  # Añade [PAD] tokens
```

### Pipeline
Abstracción de alto nivel que combina tokenizer, modelo y post-procesamiento.

```python
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")
```

### Pre-training
Entrenamiento inicial en grandes cantidades de texto sin etiquetar.

---

## Q

### Question Answering (QA)
Tarea de responder preguntas basándose en un contexto.

**Extractivo**: Extrae span del contexto
**Generativo**: Genera respuesta libre

---

## R

### RoBERTa
BERT optimizado con más datos, más tiempo de entrenamiento y sin NSP.

---

## S

### Self-Attention
Attention donde Query, Key y Value vienen de la misma secuencia.

### Sentiment Analysis
Tarea de clasificar polaridad emocional del texto (positivo, negativo, neutral).

### [SEP] Token
Token separador entre segmentos. En BERT separa oraciones.

### Sequence Classification
Clasificar una secuencia completa en categorías.

### Softmax
Función que convierte logits a probabilidades:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

### Special Tokens
Tokens con significado especial: [CLS], [SEP], [PAD], [MASK], [UNK].

---

## T

### T5 (Text-to-Text Transfer Transformer)
Modelo encoder-decoder que formula todas las tareas como text-to-text.

### Token
Unidad básica de texto procesada por el modelo. Puede ser palabra, subword o carácter.

### Token Classification
Clasificar cada token individualmente. Usado en NER, POS tagging.

### Tokenizer
Componente que convierte texto a tokens y viceversa.

```python
tokens = tokenizer.tokenize("Hello")  # ['hello']
ids = tokenizer.encode("Hello")       # [101, 7592, 102]
```

### Transformer
Arquitectura de red neuronal basada completamente en attention, sin recurrencia.

### Truncation
Recortar secuencias que exceden la longitud máxima.

```python
tokenizer(text, truncation=True, max_length=512)
```

---

## V

### Vocabulary
Conjunto de todos los tokens conocidos por un modelo/tokenizer.

```python
vocab_size = tokenizer.vocab_size  # e.g., 30522 para BERT
```

---

## W

### WordPiece
Algoritmo de tokenización usado por BERT. Subwords marcados con `##`.

```
"playing" → ["play", "##ing"]
```

---

## Z

### Zero-Shot Classification
Clasificar texto en categorías no vistas durante entrenamiento.

```python
classifier = pipeline("zero-shot-classification")
result = classifier(
    "I need to buy groceries",
    candidate_labels=["shopping", "work", "travel"]
)
```

---

## 📚 Referencias

- [Hugging Face Glossary](https://huggingface.co/docs/transformers/glossary)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
