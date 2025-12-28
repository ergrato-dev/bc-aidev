# 📖 Glosario - Semana 25: Transformers

Términos clave de la arquitectura Transformer ordenados alfabéticamente.

---

## A

### Add & Norm
Combinación de **conexión residual** (Add) y **Layer Normalization** (Norm). Estabiliza el entrenamiento y permite gradientes más fuertes.

```python
output = LayerNorm(x + Sublayer(x))
```

### Attention
Mecanismo que permite a un modelo enfocarse en partes relevantes de la entrada. Calcula pesos de importancia entre elementos.

### Attention Head
Una instancia del mecanismo de atención con sus propias matrices de proyección (W_Q, W_K, W_V).

### Attention Weights
Pesos que indican cuánta "atención" presta cada posición a las demás. Resultado de softmax sobre los scores.

---

## B

### BERT
**Bidirectional Encoder Representations from Transformers**. Modelo que usa solo el encoder del Transformer con pre-entrenamiento bidireccional.

---

## C

### Causal Mask
Máscara triangular que impide que posiciones futuras sean visibles. Esencial para decoders autoregresivos.

```python
mask = torch.triu(torch.ones(n, n), diagonal=1) == 0
```

### Cross-Attention
Atención donde Q viene de una secuencia y K, V de otra. Usado en decoders para atender al encoder.

---

## D

### Decoder
Parte del Transformer que genera la salida. Usa masked self-attention + cross-attention + feed-forward.

### d_k (d_key)
Dimensión de las Keys (y Queries). Típicamente `d_model / num_heads`.

### d_model
Dimensión del modelo (tamaño de embeddings y representaciones). Valores comunes: 256, 512, 768, 1024.

### d_v (d_value)
Dimensión de los Values. Generalmente igual a d_k.

---

## E

### Encoder
Parte del Transformer que procesa la entrada. Stack de capas con self-attention + feed-forward.

### Encoder Layer
Una capa del encoder: Multi-Head Attention → Add & Norm → Feed-Forward → Add & Norm.

---

## F

### Feed-Forward Network (FFN)
Red neuronal posición-wise: dos capas lineales con activación ReLU/GELU.

$$\text{FFN}(x) = \text{Linear}_2(\text{ReLU}(\text{Linear}_1(x)))$$

Típicamente d_ff = 4 × d_model.

---

## G

### GPT
**Generative Pre-trained Transformer**. Modelo decoder-only para generación de texto.

---

## K

### Key (K)
En atención, representa "contra qué comparamos". Se usa para calcular similitud con Query.

---

## L

### Layer Normalization
Normalización que opera sobre la dimensión de features (no batch). Más estable que Batch Norm para secuencias.

```python
LayerNorm(x) = γ * (x - μ) / σ + β
```

---

## M

### Masked Self-Attention
Self-attention con máscara causal. Cada posición solo puede atender a posiciones anteriores.

### Multi-Head Attention
Múltiples heads de atención en paralelo, concatenados y proyectados. Captura diferentes tipos de relaciones.

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$$

---

## P

### Padding Mask
Máscara que indica qué posiciones son padding y deben ignorarse en atención.

### Positional Encoding
Información de posición añadida a embeddings. El Transformer original usa funciones sinusoidales:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

---

## Q

### Query (Q)
En atención, representa "qué estamos buscando". Define qué información queremos extraer.

---

## R

### Residual Connection
Conexión que suma la entrada a la salida de una capa: `output = x + layer(x)`. Facilita el flujo de gradientes.

### RoPE
**Rotary Position Embedding**. Encoding posicional que codifica posición relativa rotando vectores.

---

## S

### Scaled Dot-Product Attention
Mecanismo de atención base del Transformer:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Self-Attention
Atención donde Q, K, V provienen de la misma secuencia. Cada token atiende a todos los demás.

### Softmax
Función que convierte scores a probabilidades (valores positivos que suman 1).

---

## T

### Transformer
Arquitectura de red neuronal basada enteramente en atención, sin recurrencia ni convoluciones. Introducida en "Attention Is All You Need" (2017).

---

## V

### Value (V)
En atención, representa "qué información extraer". Los pesos de atención ponderan los values.

---

## Fórmulas Clave

| Concepto | Fórmula |
|----------|---------|
| Attention | $\text{softmax}(QK^T/\sqrt{d_k})V$ |
| Multi-Head | $\text{Concat}(\text{head}_1,...,\text{head}_h)W^O$ |
| PE (sin) | $\sin(pos/10000^{2i/d})$ |
| PE (cos) | $\cos(pos/10000^{2i/d})$ |
| FFN | $W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$ |

---

_Semana 25 | Módulo: Deep Learning | Bootcamp IA: Zero to Hero_
