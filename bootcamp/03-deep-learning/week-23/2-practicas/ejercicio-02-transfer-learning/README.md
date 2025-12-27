# Ejercicio 02: Transfer Learning con ResNet

## 🎯 Objetivo

Aplicar Transfer Learning usando ResNet preentrenado en ImageNet para clasificar un dataset personalizado.

---

## 📋 Conceptos Clave

- **Feature Extraction**: Usar CNN como extractor de características
- **Pretrained Weights**: Pesos entrenados en ImageNet (1.2M imágenes)
- **Classifier Head**: Nueva capa para nuestras clases
- **Freeze Layers**: Congelar pesos del backbone

---

## 🔧 Paso 1: Cargar Modelo Preentrenado

```python
import torch
import torch.nn as nn
from torchvision import models

# Cargar ResNet18 con pesos de ImageNet
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
print(f'Parámetros originales: {sum(p.numel() for p in model.parameters()):,}')
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## 🔧 Paso 2: Inspeccionar Arquitectura

Exploramos la estructura del modelo:

```python
# Ver estructura
print(model)

# Capas principales
print('\n--- Capas principales ---')
for name, module in model.named_children():
    print(f'{name}: {module.__class__.__name__}')

# La última capa (fc) es el clasificador
print(f'\nClasificador original: {model.fc}')
print(f'Clases ImageNet: {model.fc.out_features}')
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## 🔧 Paso 3: Modificar Clasificador

Reemplazamos la capa final para nuestro número de clases:

```python
num_classes = 10  # Ejemplo: 10 clases

# Obtener dimensión de entrada del clasificador
in_features = model.fc.in_features
print(f'Features de entrada: {in_features}')

# Reemplazar clasificador
model.fc = nn.Linear(in_features, num_classes)
print(f'Nuevo clasificador: {model.fc}')
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## 🔧 Paso 4: Congelar Backbone

Para Feature Extraction, congelamos todas las capas excepto el clasificador:

```python
# Congelar todas las capas
for param in model.parameters():
    param.requires_grad = False

# Descongelar solo el clasificador
for param in model.fc.parameters():
    param.requires_grad = True

# Verificar
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'Parámetros entrenables: {trainable:,} / {total:,}')
print(f'Porcentaje: {100 * trainable / total:.2f}%')
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## 🔧 Paso 5: Preparar Datos

Configuramos transformaciones compatibles con ImageNet:

```python
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Transformaciones de ImageNet
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset de ejemplo (CIFAR-10)
train_dataset = datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print(f'Batches de entrenamiento: {len(train_loader)}')
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## 🔧 Paso 6: Entrenar Solo Clasificador

Entrenamos únicamente la capa final:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Solo optimizamos parámetros entrenables
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)
criterion = nn.CrossEntropyLoss()

# Entrenar 1 epoch de ejemplo
model.train()
for i, (images, labels) in enumerate(train_loader):
    if i >= 10:  # Solo 10 batches de ejemplo
        break
    
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    if i % 5 == 0:
        print(f'Batch {i}, Loss: {loss.item():.4f}')

print('\n¡Feature extraction completado!')
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## 🔧 Paso 7: Función Reutilizable

Creamos una función para facilitar el transfer learning:

```python
def create_transfer_model(base_model, num_classes, freeze_backbone=True):
    """
    Crea un modelo para transfer learning.
    
    Args:
        base_model: Modelo preentrenado
        num_classes: Número de clases del nuevo dataset
        freeze_backbone: Si True, congela el backbone
    
    Returns:
        Modelo modificado
    """
    # Obtener features de entrada
    if hasattr(base_model, 'fc'):
        in_features = base_model.fc.in_features
        classifier_name = 'fc'
    elif hasattr(base_model, 'classifier'):
        in_features = base_model.classifier[-1].in_features
        classifier_name = 'classifier'
    
    # Congelar backbone si se requiere
    if freeze_backbone:
        for param in base_model.parameters():
            param.requires_grad = False
    
    # Nuevo clasificador
    new_classifier = nn.Linear(in_features, num_classes)
    setattr(base_model, classifier_name, new_classifier)
    
    return base_model

# Ejemplo de uso
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = create_transfer_model(model, num_classes=5, freeze_backbone=True)
print(f'Modelo adaptado para 5 clases')
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## ✅ Checklist de Verificación

- [ ] Modelo preentrenado cargado correctamente
- [ ] Arquitectura inspeccionada y comprendida
- [ ] Clasificador reemplazado con nuevas clases
- [ ] Backbone congelado para feature extraction
- [ ] Datos preparados con transformaciones de ImageNet
- [ ] Entrenamiento del clasificador funcionando

---

## 📚 Recursos

- [torchvision.models](https://pytorch.org/vision/stable/models.html)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
