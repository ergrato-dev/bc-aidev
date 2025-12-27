# ⚠️ Problema de Secuencias Largas

## 🎯 Objetivos

- Entender el vanishing gradient en RNNs
- Comprender el exploding gradient
- Conocer técnicas de mitigación
- Motivar la necesidad de LSTM/GRU

---

## 1. El Problema del Gradiente

### Repaso: Cálculo de Gradientes

En BPTT, los gradientes se propagan hacia atrás en el tiempo:

$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \cdot \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}$$

Cada término $\frac{\partial h_k}{\partial h_{k-1}}$ depende de:
- La función de activación (tanh)
- Los pesos recurrentes $W_{hh}$

---

## 2. Vanishing Gradient

### ¿Qué ocurre?

La derivada de tanh está acotada: $|\tanh'(x)| \leq 1$

Cuando multiplicamos muchos valores < 1:

$$0.5 \times 0.5 \times 0.5 \times 0.5 = 0.0625$$

Después de 100 pasos:

$$0.5^{100} \approx 10^{-30}$$

### Consecuencias

```
Paso 1    Paso 50    Paso 100
  │          │          │
  ▼          ▼          ▼
Gradiente  Gradiente  Gradiente
 grande    pequeño     ≈ 0
```

Las capas tempranas **no aprenden** porque sus gradientes son ~0.

### Ejemplo Visual

```python
import torch
import matplotlib.pyplot as plt

def simulate_vanishing(steps, factor=0.5):
    """Simula el decaimiento del gradiente."""
    gradients = [1.0]
    for _ in range(steps):
        gradients.append(gradients[-1] * factor)
    return gradients

grads = simulate_vanishing(50)
plt.semilogy(grads)
plt.xlabel('Pasos temporales')
plt.ylabel('Magnitud del gradiente (log)')
plt.title('Vanishing Gradient')
```

---

## 3. Exploding Gradient

### ¿Qué ocurre?

Si los valores de $W_{hh}$ son grandes, los gradientes crecen exponencialmente:

$$2.0 \times 2.0 \times 2.0 \times 2.0 = 16$$

Después de 50 pasos:

$$2^{50} \approx 10^{15}$$

### Consecuencias

- Pesos se vuelven NaN o Inf
- Entrenamiento inestable
- Pérdida explota

### Detección

```python
# Señales de exploding gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        if grad_norm > 100:
            print(f'⚠️ Gradient explosion in {name}: {grad_norm}')
        if torch.isnan(grad_norm):
            print(f'❌ NaN gradient in {name}')
```

---

## 4. Técnicas de Mitigación

### 4.1 Gradient Clipping

Limita la magnitud de los gradientes:

```python
# Clip por norma global
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip por valor
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**Funcionamiento**:

```
Si ||g|| > max_norm:
    g = g * (max_norm / ||g||)
```

### 4.2 Inicialización Cuidadosa

```python
# Inicialización ortogonal para W_hh
nn.init.orthogonal_(self.W_hh.weight)

# Inicialización Xavier para otras capas
nn.init.xavier_uniform_(self.W_xh.weight)
```

### 4.3 Skip Connections (Inspiración ResNet)

```python
# Añadir conexión residual
h_new = h_prev + tanh(W_xh(x) + W_hh(h_prev))
```

### 4.4 Truncated BPTT

Limitar la longitud del backprop:

```python
# En vez de propagar por toda la secuencia
for i in range(0, seq_len, truncate_len):
    chunk = sequence[i:i+truncate_len]
    output, hidden = model(chunk, hidden)
    loss.backward()
    hidden = hidden.detach()  # Cortar el grafo
```

---

## 5. Experimento: Dependencias a Largo Plazo

### El Problema

Considera la tarea de predecir el género del verbo basándose en el sujeto:

```
"The cat, which sat on the mat, [was/were] happy."
           ↑                        ↑
        sujeto                    verbo
        (7 palabras de distancia)
```

### Código de Demostración

```python
import torch
import torch.nn as nn

def test_long_dependency(model, seq_lengths):
    """Prueba si el modelo puede aprender dependencias largas."""
    results = {}
    
    for length in seq_lengths:
        # Crear tarea: recordar el primer elemento
        X = torch.zeros(100, length, 10)
        X[:, 0, :] = torch.randn(100, 10)  # Señal al inicio
        y = X[:, 0, 0] > 0  # Clasificación basada en primer elemento
        
        # Entrenar y evaluar
        model.reset_parameters()
        accuracy = train_and_evaluate(model, X, y)
        results[length] = accuracy
    
    return results

# Resultados típicos con RNN vanilla:
# seq_len=10:  95% accuracy
# seq_len=50:  60% accuracy  
# seq_len=100: 50% accuracy (aleatorio)
```

---

## 6. Visualización del Problema

### Gradientes por Paso Temporal

```
Paso temporal:  1    10    20    30    40    50
                │     │     │     │     │     │
Gradiente:    1.0   0.1  0.01  0.001  ~0    ~0
                ████  ██    █     ·     
                
Las primeras capas no reciben señal de error
```

### Flujo de Información

```
     Información del pasado
            │
            ▼
     ┌──────────────┐
t=1  │   entrada    │◄── Información rica
     └──────────────┘
            │
            ▼ (se degrada)
     ┌──────────────┐
t=10 │   oculto     │◄── Algo de información
     └──────────────┘
            │
            ▼ (se pierde)
     ┌──────────────┐
t=50 │   oculto     │◄── Casi nada
     └──────────────┘
```

---

## 7. La Solución: LSTM y GRU

Las arquitecturas LSTM y GRU resuelven estos problemas mediante:

1. **Caminos de gradiente directos** (highways)
2. **Puertas multiplicativas** que controlan el flujo
3. **Cell state** que preserva información a largo plazo

```
RNN Vanilla:  h_t = tanh(W·[h_{t-1}, x_t])
              └── Información se degrada exponencialmente

LSTM:         C_t = f_t * C_{t-1} + i_t * C̃_t
              └── Cell state puede preservar información indefinidamente
```

---

## ✅ Checklist de Comprensión

- [ ] Entiendo por qué los gradientes se desvanecen
- [ ] Sé qué causa el exploding gradient
- [ ] Conozco técnicas de mitigación (clipping, init)
- [ ] Comprendo por qué las RNNs fallan en dependencias largas
- [ ] Entiendo la motivación para LSTM/GRU

---

## 📚 Siguiente Paso

En el siguiente archivo veremos cómo **LSTM** resuelve estos problemas con su arquitectura de puertas.
