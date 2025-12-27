# ⚙️ Fine-tuning: Estrategias Avanzadas

## 🎯 Objetivos

- Dominar estrategias de congelación de capas
- Implementar learning rates diferenciales
- Conocer técnicas avanzadas de fine-tuning

---

## 1. Estrategias de Congelación

### Congelar Todo Excepto Clasificador

Para datasets pequeños o muy similares a ImageNet:

```python
# Congelar todo el backbone
for param in model.parameters():
    param.requires_grad = False

# Descongelar solo el clasificador
for param in model.fc.parameters():
    param.requires_grad = True
```

### Congelar por Capas (ResNet)

Descongelar progresivamente desde las últimas capas:

```python
# ResNet tiene: conv1, bn1, layer1, layer2, layer3, layer4, fc

# Estrategia 1: Solo últimas 2 capas
for name, param in model.named_parameters():
    if 'layer3' in name or 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Estrategia 2: Todo excepto primeras capas
for name, param in model.named_parameters():
    if 'conv1' in name or 'bn1' in name or 'layer1' in name:
        param.requires_grad = False
    else:
        param.requires_grad = True
```

### Función de Utilidad

```python
def freeze_until(model, layer_name):
    """Congela todas las capas hasta layer_name (exclusive)."""
    found = False
    for name, param in model.named_parameters():
        if layer_name in name:
            found = True
        param.requires_grad = found
    
    # Verificar
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Entrenables: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# Uso
freeze_until(model, 'layer3')  # Entrena layer3, layer4, fc
```

---

## 2. Learning Rates Diferenciales

### ¿Por Qué LR Diferentes?

- **Capas iniciales**: Features genéricas, necesitan poco ajuste → LR bajo
- **Capas finales**: Features específicas, necesitan más ajuste → LR alto
- **Clasificador**: Desde cero → LR más alto

### Implementación con Parameter Groups

```python
# Agrupar parámetros
param_groups = [
    # Capas iniciales: LR muy bajo
    {'params': model.conv1.parameters(), 'lr': 1e-5},
    {'params': model.bn1.parameters(), 'lr': 1e-5},
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    
    # Capas medias: LR bajo
    {'params': model.layer2.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    
    # Capas finales: LR medio
    {'params': model.layer4.parameters(), 'lr': 5e-4},
    
    # Clasificador: LR alto
    {'params': model.fc.parameters(), 'lr': 1e-3},
]

optimizer = torch.optim.Adam(param_groups)
```

### Función Automatizada

```python
def get_param_groups(model, base_lr=1e-3, factor=0.1):
    """Crea grupos de parámetros con LR decreciente."""
    layers = [
        (model.conv1, model.bn1),
        (model.layer1,),
        (model.layer2,),
        (model.layer3,),
        (model.layer4,),
        (model.fc,)
    ]
    
    param_groups = []
    for i, layer_group in enumerate(layers):
        lr = base_lr * (factor ** (len(layers) - 1 - i))
        params = []
        for layer in layer_group:
            params.extend(layer.parameters())
        param_groups.append({'params': params, 'lr': lr})
        print(f"Grupo {i}: LR = {lr:.2e}")
    
    return param_groups

# Uso
param_groups = get_param_groups(model, base_lr=1e-3, factor=0.1)
# Grupo 0: LR = 1.00e-06  (conv1, bn1)
# Grupo 1: LR = 1.00e-05  (layer1)
# Grupo 2: LR = 1.00e-04  (layer2)
# Grupo 3: LR = 1.00e-03  (layer3)
# Grupo 4: LR = 1.00e-02  (layer4)
# Grupo 5: LR = 1.00e-01  (fc)

optimizer = torch.optim.Adam(param_groups)
```

---

## 3. Gradual Unfreezing

### Concepto

Descongelar capas progresivamente durante el entrenamiento:

```
Época 1-2:   [FROZEN] [FROZEN] [FROZEN] [TRAIN]
Época 3-4:   [FROZEN] [FROZEN] [TRAIN]  [TRAIN]
Época 5-6:   [FROZEN] [TRAIN]  [TRAIN]  [TRAIN]
Época 7+:    [TRAIN]  [TRAIN]  [TRAIN]  [TRAIN]
```

### Implementación

```python
def gradual_unfreeze(model, epoch, schedule):
    """
    Descongela capas según el schedule.
    
    Args:
        schedule: dict {epoch: layer_name} - cuándo descongelar cada capa
    """
    for e, layer_name in schedule.items():
        if epoch >= e:
            for name, param in model.named_parameters():
                if layer_name in name:
                    param.requires_grad = True

# Schedule de descongelación
schedule = {
    0: 'fc',      # Desde el inicio
    2: 'layer4',  # Época 2
    4: 'layer3',  # Época 4
    6: 'layer2',  # Época 6
}

# En el loop de entrenamiento
for epoch in range(num_epochs):
    gradual_unfreeze(model, epoch, schedule)
    train_one_epoch(model, train_loader, optimizer)
```

---

## 4. Discriminative Fine-tuning (ULMFiT)

### Concepto

LR decrece exponencialmente desde la última capa hacia la primera:

$$LR_l = LR_{base} \times \eta^{L-l}$$

Donde $\eta < 1$ (típicamente 2.6).

```python
def discriminative_lr(model, base_lr=1e-3, decay=2.6):
    """LR discriminativo estilo ULMFiT."""
    # Obtener capas en orden
    layers = list(model.children())
    n_layers = len(layers)
    
    param_groups = []
    for i, layer in enumerate(layers):
        lr = base_lr / (decay ** (n_layers - 1 - i))
        param_groups.append({
            'params': layer.parameters(),
            'lr': lr
        })
    
    return param_groups
```

---

## 5. Data Augmentation para Fine-tuning

### Augmentation Más Agresivo

Con menos datos, necesitas más augmentation:

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        shear=10
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])
```

### Test Time Augmentation (TTA)

Promediar predicciones de múltiples augmentations:

```python
def predict_with_tta(model, image, n_augments=5):
    """Predicción con TTA."""
    model.eval()
    predictions = []
    
    # Original
    with torch.no_grad():
        pred = model(image.unsqueeze(0))
        predictions.append(pred)
    
    # Augmentations
    for _ in range(n_augments - 1):
        augmented = transforms.RandomHorizontalFlip(p=0.5)(image)
        with torch.no_grad():
            pred = model(augmented.unsqueeze(0))
            predictions.append(pred)
    
    # Promedio
    return torch.stack(predictions).mean(dim=0)
```

---

## 6. Regularización para Fine-tuning

### Dropout en Clasificador

```python
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)
```

### Weight Decay Diferenciado

```python
param_groups = [
    {'params': model.features.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3, 'weight_decay': 1e-2},
]
```

### Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## 7. Receta Completa de Fine-tuning

```python
import torch
import torch.nn as nn
from torchvision import models, transforms

def setup_finetuning(num_classes, freeze_until='layer3', base_lr=1e-3):
    """Configura modelo para fine-tuning."""
    
    # 1. Cargar modelo
    model = models.resnet50(weights='IMAGENET1K_V2')
    
    # 2. Reemplazar clasificador
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # 3. Congelar capas
    freeze_until_layer(model, freeze_until)
    
    # 4. Configurar optimizer con LR diferencial
    param_groups = [
        {'params': model.layer3.parameters(), 'lr': base_lr * 0.1},
        {'params': model.layer4.parameters(), 'lr': base_lr * 0.5},
        {'params': model.fc.parameters(), 'lr': base_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    
    # 5. Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )
    
    # 6. Loss con label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    return model, optimizer, scheduler, criterion


def freeze_until_layer(model, layer_name):
    """Congela capas hasta layer_name."""
    found = False
    for name, param in model.named_parameters():
        if layer_name in name:
            found = True
        param.requires_grad = found
```

---

## ✅ Resumen de Estrategias

| Estrategia | Cuándo Usar | Implementación |
|------------|-------------|----------------|
| Congelar todo | Dataset muy pequeño | `requires_grad = False` |
| LR diferencial | Dataset mediano | Parameter groups |
| Gradual unfreeze | Evitar overfitting inicial | Schedule por época |
| Data augmentation | Siempre en fine-tuning | Transforms agresivos |

---

## 🔗 Navegación

[← Transfer Learning](03-transfer-learning.md) | [Volver a README →](../README.md)
