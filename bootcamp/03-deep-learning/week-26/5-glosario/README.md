# 📖 Glosario - Semana 26: Regularización

## B

### Batch Normalization
Técnica que normaliza las activaciones de cada capa a media 0 y varianza 1 dentro de cada mini-batch. Acelera entrenamiento y permite learning rates mayores.

### Bias-Variance Trade-off
Balance entre error por suposiciones simplificadas (bias/underfitting) y error por sensibilidad a datos (variance/overfitting).

## C

### Co-adaptación
Fenómeno donde neuronas dependen excesivamente unas de otras, reducido por Dropout.

### ColorJitter
Transformación que modifica aleatoriamente brillo, contraste, saturación y tono de imágenes.

### CutMix
Técnica de augmentation que corta y pega regiones entre imágenes, mezclando también las etiquetas.

### Cutout
Técnica que elimina regiones rectangulares aleatorias de imágenes durante entrenamiento.

## D

### Data Augmentation
Técnicas para crear variaciones artificiales de datos de entrenamiento, aumentando efectivamente el tamaño del dataset.

### Dropout
Técnica de regularización que apaga neuronas aleatoriamente durante entrenamiento con probabilidad p.

### Dropout2d
Variante de Dropout para CNNs que apaga canales completos en lugar de activaciones individuales.

## E

### Early Stopping
Técnica que detiene el entrenamiento cuando la métrica de validación deja de mejorar.

### Ensemble
Combinación de múltiples modelos. Dropout puede verse como entrenamiento implícito de un ensemble.

## G

### Gamma (γ)
Parámetro aprendible de escala en Batch Normalization.

### Gap Train-Test
Diferencia entre accuracy de entrenamiento y test. Indicador de overfitting.

### Generalización
Capacidad del modelo de funcionar bien en datos no vistos durante entrenamiento.

## I

### Internal Covariate Shift
Cambio en la distribución de activaciones durante entrenamiento, problema que BatchNorm mitiga.

### Inverted Dropout
Implementación de Dropout que escala activaciones por 1/(1-p) durante entrenamiento.

## L

### L1 Regularization (Lasso)
Penalización que suma valores absolutos de pesos: λΣ|w|.

### L2 Regularization (Ridge)
Penalización que suma cuadrados de pesos: λΣw². También llamada Weight Decay.

### Layer Normalization
Normalización sobre features en lugar de batch. Usada en Transformers.

## M

### Mixup
Técnica que mezcla pares de imágenes y etiquetas: x' = λx₁ + (1-λ)x₂.

### model.eval()
Modo de evaluación en PyTorch. Desactiva Dropout y usa running statistics en BatchNorm.

### model.train()
Modo de entrenamiento en PyTorch. Activa Dropout y usa batch statistics en BatchNorm.

## O

### Overfitting
Cuando el modelo memoriza datos de entrenamiento sin generalizar. Alta accuracy en train, baja en test.

## R

### RandomCrop
Transformación que recorta una región aleatoria de la imagen.

### RandomHorizontalFlip
Transformación que voltea la imagen horizontalmente con cierta probabilidad.

### RandomRotation
Transformación que rota la imagen un ángulo aleatorio.

### Running Statistics
Media y varianza acumuladas en BatchNorm, usadas durante inferencia.

## U

### Underfitting
Cuando el modelo es muy simple para capturar patrones. Baja accuracy en train y test.

## W

### Weight Decay
Técnica que reduce magnitud de pesos multiplicándolos por factor < 1 en cada paso. Equivalente a L2 con SGD.
