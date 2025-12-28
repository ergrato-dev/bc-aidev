# 🔄 Self-Attention

## 🎯 Objetivos

- Comprender qué es Self-Attention
- Diferenciarlo de atención encoder-decoder
- Implementar Self-Attention completo

---

## 1. ¿Qué es Self-Attention?

### Definición

En **Self-Attention**, Query, Key y Value provienen de la **misma secuencia**:

```
Input: "El gato se sentó"
        ↓    ↓    ↓    ↓
       [Q]  [Q]  [Q]  [Q]   ← Queries (de la misma entrada)
       [K]  [K]  [K]  [K]   ← Keys (de la misma entrada)
       [V]  [V]  [V]  [V]   ← Values (de la misma entrada)
```

Cada token puede "atender" a todos los demás tokens de la secuencia.

### ¿Por Qué es Útil?

Captura relaciones **dentro** de una secuencia:
- "El **gato** negro que **vimos** ayer **se** escapó" 
- ¿A qué se refiere "se"? → Self-attention conecta "se" con "gato"

---

## 2. Proyecciones Lineales

### De Input a Q, K, V

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Proyecciones lineales
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)  # (batch, seq_len, d_model)
        V = self.W_v(x)  # (batch, seq_len, d_model)
        return Q, K, V
```

### ¿Por Qué Proyecciones?

- **Diferentes representaciones**: Q, K, V capturan aspectos distintos
- **Parámetros aprendibles**: El modelo aprende qué buscar
- **Flexibilidad**: d_k puede diferir de d_model

---

## 3. Self-Attention Completo

```python
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) opcional
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Proyectar a Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2. Calcular attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.d_model ** 0.5)
        
        # 3. Aplicar máscara (opcional)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 4. Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 5. Combinar con Values
        output = torch.matmul(attention_weights, V)
        
        return output
```

---

## 4. Máscaras en Self-Attention

### Padding Mask

Ignora tokens de padding:

```python
def create_padding_mask(seq, pad_token=0):
    """
    Args:
        seq: (batch, seq_len) - IDs de tokens
    Returns:
        mask: (batch, 1, 1, seq_len)
    """
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)
    return mask  # True donde hay tokens reales
```

### Causal Mask (Look-Ahead)

Para generación autoregresiva, evita ver el futuro:

```python
def create_causal_mask(seq_len):
    """
    Máscara triangular inferior.
    
    Returns:
        mask: (seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

# Ejemplo para seq_len=4:
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

---

## 5. Ejemplo Práctico

```python
# Configuración
d_model = 64
seq_len = 5
batch_size = 2

# Crear módulo
self_attn = SelfAttention(d_model)

# Input aleatorio (simulando embeddings)
x = torch.randn(batch_size, seq_len, d_model)
print(f'Input: {x.shape}')

# Forward sin máscara
output = self_attn(x)
print(f'Output: {output.shape}')

# Forward con máscara causal
causal_mask = create_causal_mask(seq_len)
output_masked = self_attn(x, mask=causal_mask)
print(f'Output (masked): {output_masked.shape}')
```

---

## 6. Visualizando Self-Attention

```python
# Obtener attention weights (modificar la clase para retornarlos)
class SelfAttentionWithWeights(SelfAttention):
    def forward(self, x, mask=None, return_weights=False):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.d_model ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        if return_weights:
            return output, attention_weights
        return output

# Uso
model = SelfAttentionWithWeights(d_model=64)
x = torch.randn(1, 4, 64)
out, weights = model(x, return_weights=True)
print(f'Attention weights shape: {weights.shape}')  # (1, 4, 4)
```

---

## 7. Self-Attention vs Cross-Attention

| Característica | Self-Attention | Cross-Attention |
|----------------|----------------|-----------------|
| Q de | misma secuencia | decoder |
| K, V de | misma secuencia | encoder |
| Uso | encoder, decoder | decoder |
| Propósito | relaciones internas | alinear enc-dec |

---

## 8. Complejidad

### Temporal

$$O(n^2 \cdot d)$$

- n² comparaciones entre tokens
- d dimensión de cada comparación

### Espacial

$$O(n^2 + nd)$$

- n² para matriz de atención
- nd para Q, K, V

### Implicación

Para secuencias muy largas (n > 1000), self-attention es costoso.
→ Variantes: Linear Attention, Sparse Attention, Flash Attention

---

## ✅ Checklist de Comprensión

- [ ] Entiendo que Q, K, V vienen del mismo input
- [ ] Sé implementar proyecciones lineales
- [ ] Comprendo padding mask y causal mask
- [ ] Conozco la complejidad O(n²)

---

## 📚 Recursos

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Visualization](https://github.com/jessevig/bertviz)
