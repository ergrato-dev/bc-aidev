# 📖 Glosario - Semana 28

## Proyecto Final de Deep Learning

Términos clave para el proyecto integrador de Computer Vision o NLP.

---

## A

### Accuracy
Proporción de predicciones correctas sobre el total. Métrica básica de clasificación.

$$\text{Accuracy} = \frac{\text{Predicciones Correctas}}{\text{Total de Predicciones}}$$

### Attention Mechanism
Mecanismo que permite al modelo enfocarse en partes relevantes de la entrada. Base de los Transformers.

### AutoTokenizer
Clase de Hugging Face que carga automáticamente el tokenizer correcto para cualquier modelo.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

---

## B

### Backbone
Red neuronal base (sin la capa clasificadora) usada en transfer learning. Ej: ResNet sin la capa fc.

### BERT (Bidirectional Encoder Representations from Transformers)
Modelo de lenguaje preentrenado de Google que revolucionó el NLP. Usa atención bidireccional.

### Batch Size
Número de muestras procesadas antes de actualizar los pesos. Trade-off entre velocidad y estabilidad.

---

## C

### Checkpoint
Estado guardado del modelo durante el entrenamiento. Permite resumir o seleccionar el mejor modelo.

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch
}, 'checkpoint.pth')
```

### Classification Head
Capa(s) final(es) que transforman las features del backbone en predicciones de clase.

### Confusion Matrix
Tabla que muestra predicciones vs. valores reales. Útil para analizar errores por clase.

---

## D

### Data Augmentation
Técnicas para aumentar artificialmente el dataset de entrenamiento aplicando transformaciones.

```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2)
])
```

### DistilBERT
Versión destilada de BERT: 40% más pequeño, 60% más rápido, retiene 97% del rendimiento.

---

## E

### Early Stopping
Técnica que detiene el entrenamiento cuando la métrica de validación deja de mejorar.

### Embedding
Representación vectorial densa de tokens, palabras o imágenes en un espacio continuo.

### Epoch
Una pasada completa por todo el dataset de entrenamiento.

---

## F

### F1-Score
Media armónica de precision y recall. Útil cuando las clases están desbalanceadas.

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Feature Extraction
Usar un modelo preentrenado solo para extraer features, sin actualizar sus pesos.

### Fine-tuning
Ajustar los pesos de un modelo preentrenado en un nuevo dataset o tarea.

### FP16 (Mixed Precision)
Entrenamiento usando float16 para acelerar cálculos y reducir memoria en GPU.

---

## H

### Hugging Face
Plataforma y librería líder para NLP con miles de modelos preentrenados y datasets.

---

## I

### ImageNet
Dataset masivo de imágenes (14M+) usado para preentrenar modelos de visión.

### Inference
Usar un modelo entrenado para hacer predicciones en nuevos datos.

```python
model.eval()
with torch.no_grad():
    predictions = model(input)
```

---

## L

### Label
Etiqueta o clase correcta de una muestra. Usado para entrenamiento supervisado.

### Learning Rate
Hiperparámetro que controla el tamaño de los pasos de optimización.

### Load Best Model at End
Estrategia del Trainer que carga el mejor checkpoint al finalizar el entrenamiento.

---

## M

### Max Length
Longitud máxima de secuencia para tokenización. Textos más largos se truncan.

### Model Zoo
Colección de modelos preentrenados listos para usar. Ej: TorchVision, Hugging Face Hub.

---

## N

### Normalize
Estandarizar datos para tener media 0 y desviación estándar 1.

```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # ImageNet mean
    std=[0.229, 0.224, 0.225]    # ImageNet std
)
```

### num_labels
Número de clases en clasificación. Determina el tamaño de la capa de salida.

---

## P

### Padding
Rellenar secuencias cortas para igualar longitudes en un batch.

### Pipeline
Abstracción de Hugging Face para inferencia fácil con modelos preentrenados.

```python
from transformers import pipeline
classifier = pipeline("text-classification", model="./model")
```

### Precision
Proporción de predicciones positivas que son correctas.

$$\text{Precision} = \frac{TP}{TP + FP}$$

### Pretrained
Modelo entrenado previamente en un dataset grande (ImageNet, Wikipedia).

---

## R

### Recall
Proporción de positivos reales que fueron detectados.

$$\text{Recall} = \frac{TP}{TP + FN}$$

### ResNet (Residual Network)
Arquitectura CNN con conexiones residuales que permiten entrenar redes muy profundas.

### RoBERTa
Versión optimizada de BERT con mejor entrenamiento y sin Next Sentence Prediction.

---

## S

### Softmax
Función que convierte logits en probabilidades que suman 1.

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

### State Dict
Diccionario con los pesos del modelo en PyTorch.

```python
model.state_dict()  # Obtener pesos
model.load_state_dict(state_dict)  # Cargar pesos
```

---

## T

### Tokenization
Proceso de convertir texto en tokens (subpalabras, palabras o caracteres).

### Trainer
API de alto nivel de Hugging Face para entrenar modelos de forma simplificada.

### TrainingArguments
Configuración del entrenamiento en Hugging Face (epochs, lr, batch_size, etc.).

### Transfer Learning
Usar conocimiento de un modelo preentrenado para una nueva tarea relacionada.

### Truncation
Cortar secuencias que excedan la longitud máxima permitida.

---

## V

### Validation Set
Porción del dataset usada para evaluar durante el entrenamiento y ajustar hiperparámetros.

---

## W

### Warmup
Período inicial de entrenamiento con learning rate creciente gradualmente.

### Weight Decay
Regularización L2 que penaliza pesos grandes. Previene overfitting.

```python
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
```

### Weights
Parámetros del modelo que se actualizan durante el entrenamiento.

---

## Fórmulas Clave

| Métrica | Fórmula |
|---------|---------|
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ |
| Precision | $\frac{TP}{TP + FP}$ |
| Recall | $\frac{TP}{TP + FN}$ |
| F1-Score | $2 \times \frac{P \times R}{P + R}$ |

---

_Glosario Semana 28 - Proyecto Final Deep Learning_
