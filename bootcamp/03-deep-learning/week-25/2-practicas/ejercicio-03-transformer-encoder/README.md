# 🏗️ Ejercicio 03: Transformer Encoder

## 🎯 Objetivo

Construir un **Transformer Encoder** completo combinando Multi-Head Attention, Feed-Forward Network, y Layer Normalization.

---

## 📋 Conceptos Clave

### Arquitectura de una Encoder Layer

```
Input
  │
  ├──────────────────┐
  ▼                  │
Multi-Head Attention │
  │                  │
  ▼                  │
  + ◄────────────────┘  (Residual Connection)
  │
  ▼
Layer Norm
  │
  ├──────────────────┐
  ▼                  │
Feed-Forward         │
  │                  │
  ▼                  │
  + ◄────────────────┘  (Residual Connection)
  │
  ▼
Layer Norm
  │
Output
```

### Componentes

1. **Multi-Head Attention**: Captura relaciones entre tokens
2. **Feed-Forward Network**: MLP con expansión (4x típicamente)
3. **Add & Norm**: Conexión residual + Layer Normalization

---

## 📝 Instrucciones

### Paso 1: Feed-Forward Network

Red neuronal simple: Linear → ReLU → Dropout → Linear

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
```

### Paso 2: Encoder Layer

Combina atención + feed-forward con residuales:

```python
# Attention con residual
x = x + self.dropout(self.attention(x, x, x))
x = self.norm1(x)

# FFN con residual
x = x + self.dropout(self.ffn(x))
x = self.norm2(x)
```

### Paso 3: Positional Encoding

Los Transformers no tienen noción de orden. Añadimos posición:

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Paso 4: Transformer Encoder Completo

Apila N encoder layers con embedding + positional encoding.

### Paso 5: Probar con Secuencias

Procesa secuencias de tokens y observa las representaciones.

---

## ✅ Criterios de Éxito

- [ ] FeedForward expande y contrae correctamente
- [ ] EncoderLayer aplica residuales y normalización
- [ ] PositionalEncoding añade información de posición
- [ ] TransformerEncoder procesa secuencias correctamente

---

## 🔗 Recursos

- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Residual Connections](https://arxiv.org/abs/1512.03385)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)

---

## ⏱️ Tiempo Estimado

60 minutos
