# 📖 Glosario - Semana 23: CNNs II

## B

### BasicBlock
Bloque residual básico usado en ResNet-18 y ResNet-34. Consiste en dos convoluciones 3×3 con una skip connection.

```python
# Estructura: Conv3x3 → BN → ReLU → Conv3x3 → BN → (+x) → ReLU
```

### Backbone
Parte principal de una red neuronal que extrae características. En transfer learning, se refiere a las capas convolucionales preentrenadas.

### Bottleneck
Bloque residual eficiente usado en ResNet-50+. Usa convoluciones 1×1 para reducir y expandir canales.

```
1×1 (reducir) → 3×3 (procesar) → 1×1 (expandir)
```

---

## C

### Cosine Annealing
Estrategia de decaimiento del learning rate que sigue una curva coseno, permitiendo una reducción suave.

$$lr_t = lr_{min} + \frac{1}{2}(lr_{max} - lr_{min})(1 + \cos(\frac{t\pi}{T}))$$

---

## D

### Degradation Problem
Fenómeno donde redes más profundas tienen **peor** rendimiento que redes menos profundas, incluso en el conjunto de entrenamiento. ResNet resuelve esto con skip connections.

### Discriminative Learning Rates
Técnica de fine-tuning donde diferentes capas usan diferentes learning rates. Capas profundas (cerca de entrada) usan LR más bajo.

```python
# head: lr=1e-3, layer4: lr=1e-4, layer3: lr=1e-5, ...
```

### Downsample
Capa que ajusta las dimensiones de la skip connection cuando cambia el número de canales o el tamaño espacial.

---

## E

### Early Stopping
Técnica de regularización que detiene el entrenamiento cuando la métrica de validación deja de mejorar.

### Expansion
Factor por el cual el Bottleneck block expande los canales de salida. En ResNet, expansion=4.

### Exploding Gradients
Problema donde los gradientes crecen exponencialmente durante backpropagation, causando inestabilidad.

---

## F

### Feature Extraction
Estrategia de transfer learning donde se congela el backbone y solo se entrena el clasificador.

### Fine-tuning
Estrategia de transfer learning donde se entrenan algunas o todas las capas del modelo preentrenado con un learning rate bajo.

### Freeze/Unfreeze
Congelar (freeze) significa hacer que los pesos no se actualicen (`requires_grad=False`). Descongelar (unfreeze) permite que se entrenen.

---

## G

### Gradual Unfreezing
Técnica donde se descongelan capas progresivamente durante el entrenamiento, comenzando por las más cercanas a la salida.

---

## I

### Identity Mapping
Función identidad `f(x) = x`. Las skip connections permiten que la red aprenda esta función fácilmente.

### ImageNet
Dataset de ~1.2 millones de imágenes en 1000 categorías. Base para la mayoría de modelos preentrenados de visión.

---

## L

### Layer Groups
Agrupación de capas para aplicar diferentes configuraciones (LR, freeze/unfreeze) durante fine-tuning.

### Learning Rate Warmup
Técnica donde el LR comienza bajo y aumenta gradualmente durante las primeras epochs para estabilizar el entrenamiento.

---

## P

### Pretrained Weights
Pesos de una red entrenada en un dataset grande (ej: ImageNet) que se reutilizan como punto de partida.

---

## R

### Residual Learning
Paradigma donde la red aprende la función residual `F(x) = H(x) - x` en lugar de `H(x)` directamente, facilitando el aprendizaje de la identidad.

### ResNet (Residual Network)
Familia de arquitecturas que usan skip connections para permitir el entrenamiento de redes muy profundas (18, 34, 50, 101, 152 capas).

---

## S

### Skip Connection
Conexión que suma la entrada de un bloque a su salida: `y = F(x) + x`. Permite el flujo directo de gradientes.

### Stem
Primeras capas de una red (conv inicial, batch norm, pooling) antes de los bloques residuales.

---

## T

### Test Time Augmentation (TTA)
Técnica donde se aplican múltiples augmentations a una imagen durante inferencia y se promedian las predicciones.

### Transfer Learning
Técnica donde se reutiliza conocimiento de un modelo entrenado en un dataset grande para un nuevo problema relacionado.

---

## V

### Vanishing Gradients
Problema donde los gradientes se hacen exponencialmente pequeños durante backpropagation, impidiendo que las primeras capas aprendan.

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial W_n} \cdot \prod_{i=1}^{n-1} \frac{\partial W_{i+1}}{\partial W_i} \approx 0$$

---

## W

### Warmup
Ver Learning Rate Warmup.

### Weight Decay
Regularización L2 que penaliza pesos grandes. Comúnmente usado junto con fine-tuning.

```python
optimizer = Adam(params, lr=1e-4, weight_decay=1e-4)
```
