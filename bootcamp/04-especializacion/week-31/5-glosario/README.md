# 📖 Glosario - Semana 31: LLMs

Términos clave de Large Language Models, ordenados alfabéticamente.

---

## A

### Attention Mechanism
Mecanismo que permite al modelo enfocarse en partes relevantes del input. En Transformers, calcula pesos de atención entre todos los tokens.

**Fórmula**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Autoregressive
Tipo de modelo que genera tokens uno por uno, condicionando cada token en los anteriores. GPT es autoregresivo.

```
Input:  "The cat"
Output: "The cat sat" → "The cat sat on" → ...
```

---

## B

### BERT (Bidirectional Encoder Representations from Transformers)
Modelo encoder-only de Google que usa atención bidireccional. Excelente para tareas de clasificación y NLU.

### BPE (Byte Pair Encoding)
Algoritmo de tokenización que aprende subpalabras frecuentes del corpus.

```
"unhappiness" → ["un", "happiness"] o ["un", "happ", "iness"]
```

---

## C

### Causal Attention
Atención que solo mira tokens anteriores (izquierda). Usada en modelos de lenguaje generativos.

```
Posición 3 puede ver: [1, 2, 3]
Posición 3 NO ve: [4, 5, ...]
```

### Chain-of-Thought (CoT)
Técnica de prompting que hace al modelo razonar paso a paso antes de dar la respuesta final.

### Context Window
Número máximo de tokens que el modelo puede procesar. GPT-4 tiene ~128K tokens.

---

## D

### Decoder-Only
Arquitectura que solo usa el decoder del Transformer. GPT, LLaMA, Mistral son decoder-only.

### Distillation (Destilación)
Técnica para transferir conocimiento de un modelo grande (teacher) a uno pequeño (student).

---

## E

### Embedding
Representación vectorial densa de tokens. Cada token se mapea a un vector de dimensión fija.

```python
# Token "hello" → vector de 768 dimensiones
embedding = [0.23, -0.15, 0.67, ..., 0.42]
```

### Encoder-Only
Arquitectura que solo usa el encoder. BERT, RoBERTa son encoder-only.

---

## F

### Few-Shot Learning
Capacidad del modelo para realizar tareas nuevas con pocos ejemplos en el prompt.

```
Ejemplo 1: X → Y
Ejemplo 2: A → B
Nuevo: C → ?  (El modelo predice)
```

### Fine-tuning
Proceso de continuar entrenando un modelo pre-entrenado en datos específicos de una tarea.

### FP16 (Float16)
Precisión de punto flotante de 16 bits. Reduce memoria y acelera entrenamiento.

---

## G

### GPT (Generative Pre-trained Transformer)
Familia de modelos de OpenAI basados en arquitectura decoder-only autoregresiva.

### Gradient Accumulation
Técnica para simular batches grandes acumulando gradientes de varios mini-batches.

---

## H

### Hallucination (Alucinación)
Cuando el modelo genera información falsa o inventada que parece plausible.

### Hidden State
Representación interna del modelo en cada capa. Contiene información semántica del input.

---

## I

### In-Context Learning
Capacidad de aprender de ejemplos proporcionados en el prompt sin actualizar pesos.

### Inference
Proceso de usar un modelo entrenado para generar predicciones/texto.

### Instruction Tuning
Fine-tuning específico para seguir instrucciones en lenguaje natural.

---

## K

### KV Cache
Cache de Keys y Values calculados para tokens anteriores. Acelera generación autoregresiva.

---

## L

### LLM (Large Language Model)
Modelo de lenguaje con billones de parámetros entrenado en texto masivo. Ejemplos: GPT-4, LLaMA, Claude.

### LoRA (Low-Rank Adaptation)
Técnica de fine-tuning eficiente que solo entrena matrices de bajo rango añadidas al modelo.

**Fórmula**:
$$W' = W + BA$$

Donde $B \in \mathbb{R}^{d \times r}$ y $A \in \mathbb{R}^{r \times d}$ con $r \ll d$

---

## M

### MLM (Masked Language Modeling)
Objetivo de entrenamiento de BERT. Predecir tokens enmascarados en el input.

```
Input:  "The [MASK] sat on the mat"
Target: "cat"
```

### Multi-Head Attention
Atención ejecutada en paralelo con diferentes proyecciones. Captura relaciones diversas.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

---

## N

### Next Token Prediction
Objetivo de entrenamiento de GPT. Predecir el siguiente token dado el contexto.

---

## O

### Open-Weights
Modelos cuyo pesos están disponibles públicamente. Ejemplo: LLaMA, Mistral.

---

## P

### PEFT (Parameter-Efficient Fine-Tuning)
Familia de técnicas que ajustan solo un pequeño porcentaje de parámetros. Incluye LoRA, QLoRA, adapters.

### Perplexity
Métrica de evaluación de modelos de lenguaje. Menor perplexity = mejor modelo.

$$PPL = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(x_i|x_{<i})\right)$$

### Pre-training
Fase inicial de entrenamiento en grandes cantidades de texto sin supervisión específica.

### Prompt
Texto de entrada que se proporciona al modelo para generar una respuesta.

### Prompt Engineering
Arte de diseñar prompts efectivos para obtener mejores respuestas del modelo.

---

## Q

### QLoRA
LoRA combinado con cuantización a 4 bits. Permite fine-tuning con muy poca memoria.

### Quantization (Cuantización)
Reducir precisión de pesos (32-bit → 8-bit → 4-bit) para reducir memoria.

---

## R

### RLHF (Reinforcement Learning from Human Feedback)
Técnica de alineación que usa feedback humano para mejorar respuestas del modelo.

### Role Prompting
Técnica que asigna un rol/persona específica al modelo en el prompt.

```
"You are an expert Python programmer..."
```

---

## S

### Self-Attention
Atención donde Query, Key, Value vienen de la misma secuencia. Base de Transformers.

### SFT (Supervised Fine-Tuning)
Fine-tuning supervisado con pares (input, output) etiquetados.

### System Prompt
Prompt especial que define el comportamiento general del asistente.

---

## T

### Temperature
Parámetro que controla aleatoriedad en la generación.
- Baja (0.1): Determinista
- Alta (1.0+): Creativo/aleatorio

$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

### Token
Unidad básica de texto que procesa el modelo. Puede ser palabra, subpalabra o carácter.

### Top-K Sampling
Método que limita la generación a los K tokens más probables.

### Top-P (Nucleus) Sampling
Método que selecciona tokens hasta acumular probabilidad P.

### Transformer
Arquitectura neural basada en atención. Base de todos los LLMs modernos.

---

## Z

### Zero-Shot Learning
Capacidad del modelo para realizar tareas sin ejemplos previos en el prompt.

---

## 📊 Comparativa de Arquitecturas

| Característica | GPT (Decoder) | BERT (Encoder) | T5 (Enc-Dec) |
|---------------|---------------|----------------|--------------|
| Atención | Causal | Bidireccional | Ambas |
| Uso principal | Generación | Clasificación | Seq2Seq |
| Pre-training | Next token | MLM | Span corruption |
| Ejemplos | GPT-4, LLaMA | BERT, RoBERTa | T5, BART |

---

## 📈 Métricas Comunes

| Métrica | Uso | Mejor si... |
|---------|-----|-------------|
| Perplexity | Calidad del modelo | Menor |
| BLEU | Traducción/generación | Mayor |
| ROUGE | Resumen | Mayor |
| Accuracy | Clasificación | Mayor |

---

_Glosario Semana 31 - Bootcamp IA_
