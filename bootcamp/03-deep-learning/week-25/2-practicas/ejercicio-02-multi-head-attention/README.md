# 🔀 Ejercicio 02: Multi-Head Attention

## 🎯 Objetivo

Implementar **Multi-Head Attention** combinando múltiples mecanismos de atención en paralelo para capturar diferentes tipos de relaciones.

---

## 📋 Conceptos Clave

### ¿Por Qué Múltiples Heads?

Un solo head de atención aprende un tipo de relación. Múltiples heads permiten:
- Capturar relaciones **sintácticas** (sujeto-verbo)
- Capturar relaciones **semánticas** (sinónimos)
- Diferentes "puntos de vista" simultáneos

### Fórmula

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

donde cada head:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

---

## 📝 Instrucciones

### Paso 1: Setup e Importaciones

Abre `starter/main.py` y configura el entorno.

### Paso 2: Proyecciones Lineales

Cada head tiene sus propias matrices de proyección:

```python
self.W_q = nn.Linear(d_model, d_k)
self.W_k = nn.Linear(d_model, d_k)
self.W_v = nn.Linear(d_model, d_v)
```

### Paso 3: Reshape para Múltiples Heads

El truco es dividir la dimensión en `num_heads`:

```python
# De (batch, seq, d_model) a (batch, num_heads, seq, d_k)
Q = Q.view(batch, seq_len, num_heads, d_k).transpose(1, 2)
```

### Paso 4: Atención en Paralelo

Aplicamos atención a todos los heads simultáneamente (gracias al batch dimension).

### Paso 5: Concatenar y Proyectar

Unimos los heads y aplicamos W_O:

```python
# De (batch, num_heads, seq, d_k) a (batch, seq, d_model)
output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
output = self.W_o(output)
```

### Paso 6: Clase Completa

Implementa la clase `MultiHeadAttention` completa.

---

## ✅ Criterios de Éxito

- [ ] La clase inicializa correctamente con num_heads y d_model
- [ ] Las proyecciones Q, K, V funcionan
- [ ] El reshape a múltiples heads es correcto
- [ ] La salida tiene shape (batch, seq_len, d_model)

---

## 🔗 Recursos

- [Multi-Head Attention Explained](https://jalammar.github.io/illustrated-transformer/#multi-headed-attention)
- [PyTorch nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)

---

## ⏱️ Tiempo Estimado

45 minutos
