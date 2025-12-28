# 📖 Glosario - Semana 27: Optimización en Deep Learning

Términos clave ordenados alfabéticamente.

---

## A

### Adam (Adaptive Moment Estimation)

Optimizador que combina momentum (primer momento) con RMSprop (segundo momento). Adapta el learning rate por parámetro.

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

**Hiperparámetros**: β₁=0.9 (momentum), β₂=0.999 (escala), ε=1e-8

### AdamW

Variante de Adam con **weight decay desacoplado**. Aplica regularización L2 directamente a los pesos, no al gradiente.

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**Uso**: Recomendado sobre Adam cuando se necesita regularización.

---

## B

### Batch Size

Número de muestras procesadas antes de actualizar los pesos. Afecta la estabilidad del gradiente y el uso de memoria.

- **Pequeño** (16-32): Más ruido, mejor generalización
- **Grande** (256+): Más estable, puede requerir ajuste de LR

### Bias Correction

Corrección aplicada en Adam para compensar la inicialización en cero de los momentos m y v.

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

---

## C

### Callback

Función ejecutada en puntos específicos del entrenamiento (inicio/fin de época, batch, etc.).

```python
# Ejemplo: Early Stopping es un callback
if early_stopping(val_loss):
    break
```

### Checkpoint

Archivo que guarda el estado del modelo, optimizador y scheduler para poder resumir entrenamiento.

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint.pth')
```

### Cosine Annealing

Scheduler que reduce el LR siguiendo una curva coseno, permitiendo decay suave.

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t\pi}{T}))$$

---

## E

### Early Stopping

Técnica que detiene el entrenamiento cuando la métrica de validación deja de mejorar por N épocas (patience).

**Propósito**: Prevenir overfitting y ahorrar tiempo de cómputo.

### Exploding Gradients

Problema donde los gradientes crecen exponencialmente durante backpropagation, causando actualizaciones inestables.

**Solución**: Gradient clipping, mejor inicialización.

---

## G

### Glorot Initialization

Ver **Xavier Initialization**.

### Gradient Clipping

Técnica que limita la magnitud de los gradientes para prevenir exploding gradients.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Tipos**: Por norma (recomendado) o por valor.

---

## H

### He Initialization (Kaiming)

Inicialización de pesos diseñada para activaciones ReLU. Escala varianza considerando que ReLU "mata" valores negativos.

$$W \sim N(0, \sqrt{\frac{2}{n_{in}}})$$

```python
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```

---

## L

### Learning Rate (LR)

Hiperparámetro que controla el tamaño del paso en la actualización de pesos. Crucial para convergencia.

- **Muy alto**: No converge, oscila
- **Muy bajo**: Convergencia lenta, mínimos locales

### Learning Rate Schedule

Estrategia para modificar el LR durante el entrenamiento.

**Tipos comunes**: StepLR, CosineAnnealing, OneCycleLR, ReduceOnPlateau

### L2 Regularization

Penalización añadida a la loss proporcional a la norma L2 de los pesos.

$$L_{total} = L_{original} + \lambda \sum w_i^2$$

---

## M

### Momentum

Técnica que acumula gradientes pasados para acelerar convergencia y suavizar oscilaciones.

$$v_t = \beta v_{t-1} + \nabla L$$
$$w = w - \eta v_t$$

**β típico**: 0.9

### Model State Dict

Diccionario de PyTorch que contiene todos los parámetros (pesos y biases) del modelo.

```python
model.state_dict()  # Obtener
model.load_state_dict(state_dict)  # Cargar
```

---

## O

### OneCycleLR

Scheduler que implementa la política 1cycle: warmup hasta max_lr, luego decay hasta min_lr.

```python
scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=10, steps_per_epoch=len(loader))
```

**Importante**: Hacer `scheduler.step()` por batch, no por época.

### Optimizer State Dict

Estado interno del optimizador (momentos, contadores) necesario para resumir entrenamiento.

---

## P

### Patience

Número de épocas sin mejora que Early Stopping espera antes de detener el entrenamiento.

**Valor típico**: 5-10 épocas

---

## R

### ReduceLROnPlateau

Scheduler que reduce el LR cuando una métrica deja de mejorar.

```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
scheduler.step(val_loss)  # Necesita la métrica
```

---

## S

### SGD (Stochastic Gradient Descent)

Optimizador básico que actualiza pesos en dirección opuesta al gradiente.

$$w = w - \eta \nabla L$$

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### StepLR

Scheduler que reduce el LR por un factor cada N épocas.

```python
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # LR × 0.1 cada 30 épocas
```

---

## V

### Vanishing Gradients

Problema donde los gradientes se vuelven muy pequeños durante backpropagation, impidiendo que las capas iniciales aprendan.

**Causas**: Inicialización incorrecta, activaciones saturantes (sigmoid).

**Soluciones**: He initialization, ReLU, BatchNorm, skip connections.

---

## W

### Warmup

Período inicial de entrenamiento donde el LR aumenta gradualmente desde un valor pequeño.

**Propósito**: Estabilizar entrenamiento inicial, especialmente con batch sizes grandes.

### Weight Decay

Técnica de regularización que penaliza pesos grandes. En AdamW, se aplica directamente:

$$w = w - \lambda w$$

**Diferencia con L2**: Weight decay es independiente del gradiente.

---

## X

### Xavier Initialization (Glorot)

Inicialización de pesos para activaciones lineales, tanh o sigmoid. Mantiene varianza considerando entrada y salida.

$$W \sim N(0, \sqrt{\frac{2}{n_{in} + n_{out}}})$$

```python
nn.init.xavier_normal_(layer.weight)
```

---

## Fórmulas Clave

### Update Rules

| Optimizador | Fórmula |
|-------------|---------|
| SGD | $w = w - \eta \nabla L$ |
| Momentum | $v = \beta v + \nabla L$; $w = w - \eta v$ |
| Adam | $m = \beta_1 m + (1-\beta_1)\nabla L$; $v = \beta_2 v + (1-\beta_2)(\nabla L)^2$ |

### Inicialización

| Método | Varianza |
|--------|----------|
| Xavier | $\frac{2}{n_{in} + n_{out}}$ |
| He | $\frac{2}{n_{in}}$ |

---

_Semana 27 - Optimización en Deep Learning_
