# 🖼️ Ejercicio 01: Clasificación de Imágenes

## 🎯 Objetivo

Aprender a clasificar imágenes usando modelos pre-entrenados de PyTorch (ResNet, EfficientNet).

## 📋 Conceptos Clave

- **Clasificación de imágenes**: Asignar una etiqueta a toda la imagen
- **Transfer learning**: Usar modelos entrenados en ImageNet
- **Top-k predictions**: Las k predicciones más probables
- **Softmax**: Convertir logits a probabilidades

## ⏱️ Tiempo Estimado

30 minutos

---

## 📝 Instrucciones

### Paso 1: Configuración del Entorno

Instala las dependencias necesarias:

```bash
pip install torch torchvision pillow requests
```

### Paso 2: Cargar un Modelo Pre-entrenado

Los modelos de `torchvision` vienen pre-entrenados en ImageNet (1000 clases):

```python
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# Cargar modelo con pesos pre-entrenados
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()  # Modo evaluación

print(f"Modelo cargado: ResNet50")
print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

### Paso 3: Preparar el Preprocesamiento

Las imágenes deben transformarse al formato esperado por el modelo:

```python
# Transformaciones estándar ImageNet
preprocess = transforms.Compose([
    transforms.Resize(256),              # Redimensionar
    transforms.CenterCrop(224),          # Recortar al centro
    transforms.ToTensor(),               # Convertir a tensor
    transforms.Normalize(                # Normalizar
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Paso 4: Clasificar una Imagen

```python
from PIL import Image
import requests
from io import BytesIO

# Descargar imagen de ejemplo
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert('RGB')

# Preprocesar
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # Añadir dimensión batch

# Inferencia
with torch.no_grad():
    output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
```

### Paso 5: Interpretar Resultados

```python
# Cargar etiquetas de ImageNet
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(IMAGENET_LABELS_URL).text.strip().split('\n')

# Top-5 predicciones
top5_prob, top5_catid = torch.topk(probabilities, 5)

print("\n🎯 Top-5 Predicciones:")
for i in range(5):
    print(f"  {i+1}. {labels[top5_catid[i]]:30s} ({top5_prob[i].item()*100:.2f}%)")
```

### Paso 6: Probar Otros Modelos

```python
# EfficientNet (más eficiente)
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

model_efficient = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model_efficient.eval()

# MobileNet (para móviles)
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

model_mobile = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model_mobile.eval()
```

---

## 🔍 Análisis

### Comparación de Modelos

| Modelo | Parámetros | Top-1 Acc | Velocidad |
|--------|------------|-----------|-----------|
| ResNet50 | 25.6M | 80.4% | Media |
| EfficientNet-B0 | 5.3M | 77.1% | Rápida |
| MobileNetV3-S | 2.5M | 67.7% | Muy rápida |

### ¿Cuándo usar cada modelo?

- **ResNet50**: Balance precisión/velocidad
- **EfficientNet**: Recursos limitados, buena precisión
- **MobileNet**: Aplicaciones móviles/edge

---

## ✅ Checklist de Verificación

- [ ] Modelo ResNet50 cargado correctamente
- [ ] Imagen preprocesada con transformaciones ImageNet
- [ ] Predicción ejecutada sin errores
- [ ] Top-5 predicciones mostradas con probabilidades
- [ ] Probado con al menos 2 modelos diferentes

---

## 📚 Recursos

- [torchvision.models](https://pytorch.org/vision/stable/models.html)
- [ImageNet Classes](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
