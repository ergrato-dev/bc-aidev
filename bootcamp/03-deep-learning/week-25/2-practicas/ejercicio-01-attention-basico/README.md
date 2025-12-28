# 👁️ Ejercicio 01: Atención Básica

## 🎯 Objetivo

Implementar el mecanismo de **Scaled Dot-Product Attention** desde cero para comprender cómo Query, Key y Value interactúan.

---

## 📋 Conceptos Clave

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Donde:
- **Q** (Query): Lo que estamos buscando
- **K** (Key): Índices contra los que comparamos
- **V** (Value): Información a recuperar
- **d_k**: Dimensión de las keys (para estabilidad numérica)

---

## 📝 Instrucciones

### Paso 1: Importaciones y Setup

Abre `starter/main.py` y descomenta las importaciones básicas:

```python
import torch
import torch.nn.functional as F
```

### Paso 2: Producto Punto Q·K^T

El primer paso es calcular la similitud entre queries y keys:

```python
# scores tiene forma (batch, seq_len_q, seq_len_k)
scores = torch.matmul(Q, K.transpose(-2, -1))
```

Descomenta la sección correspondiente en `starter/main.py`.

### Paso 3: Escalado por √d_k

Sin escalar, los valores pueden ser muy grandes y softmax satura:

```python
d_k = K.size(-1)
scaled_scores = scores / (d_k ** 0.5)
```

### Paso 4: Softmax

Convertimos scores a probabilidades (suman 1 por fila):

```python
attention_weights = F.softmax(scaled_scores, dim=-1)
```

### Paso 5: Multiplicar por Values

Finalmente, usamos los pesos para combinar los values:

```python
output = torch.matmul(attention_weights, V)
```

### Paso 6: Función Completa

Une todo en una función `scaled_dot_product_attention`.

### Paso 7: Visualización

Visualiza los pesos de atención con matplotlib para entender qué tokens "atienden" a cuáles.

---

## ✅ Criterios de Éxito

- [ ] La función `scaled_dot_product_attention` retorna output y weights
- [ ] El output tiene la misma forma que V
- [ ] Los weights suman 1 en la última dimensión
- [ ] Puedes visualizar el patrón de atención

---

## 🔗 Recursos

- [Attention Is All You Need (Paper)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

## ⏱️ Tiempo Estimado

30 minutos
