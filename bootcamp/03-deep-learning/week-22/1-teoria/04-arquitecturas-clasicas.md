# 🏛️ Arquitecturas Clásicas: LeNet y VGG

## 🎯 Objetivos

- Conocer la historia y evolución de las arquitecturas CNN
- Entender LeNet-5 como la primera CNN práctica
- Dominar la filosofía de diseño de VGG
- Implementar ambas arquitecturas en PyTorch

---

## 📋 Contenido

### 1. LeNet-5 (1998)

La primera red convolucional exitosa, diseñada por Yann LeCun para reconocimiento de dígitos escritos a mano.

#### Historia

- **Creador**: Yann LeCun (AT&T Bell Labs)
- **Año**: 1998
- **Aplicación**: Lectura automática de cheques bancarios
- **Dataset**: MNIST (dígitos 0-9)
- **Impacto**: Demostró que CNNs podían superar métodos tradicionales

#### Arquitectura

```
ENTRADA → C1 → S2 → C3 → S4 → C5 → F6 → SALIDA

Capa    | Tipo          | Output    | Parámetros
--------|---------------|-----------|------------
Input   | -             | 32×32×1   | -
C1      | Conv 5×5, 6   | 28×28×6   | 156
S2      | AvgPool 2×2   | 14×14×6   | 0
C3      | Conv 5×5, 16  | 10×10×16  | 2,416
S4      | AvgPool 2×2   | 5×5×16    | 0
C5      | Conv 5×5, 120 | 1×1×120   | 48,120
F6      | FC 84         | 84        | 10,164
Output  | FC 10         | 10        | 850
--------|---------------|-----------|------------
Total   |               |           | ~61,706
```

#### Características Clave

| Característica | Descripción |
|----------------|-------------|
| **Tamaño entrada** | 32×32 (MNIST se rellena de 28×28) |
| **Activación** | Tanh (en el paper original) |
| **Pooling** | Average pooling (subsampling) |
| **Conexiones C3** | No todas las conexiones (optimización) |

#### Implementación en PyTorch

```python
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """
    LeNet-5 modernizado para MNIST.
    
    Cambios del original:
    - ReLU en lugar de Tanh
    - Max pooling en lugar de average
    - Todas las conexiones en C3
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # C1: 1@32×32 → 6@28×28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        
        # S2: 6@28×28 → 6@14×14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # C3: 6@14×14 → 16@10×10
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # S4: 16@10×10 → 16@5×5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # C5: 16@5×5 → 120@1×1
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        
        # F6: 120 → 84
        self.fc1 = nn.Linear(120, 84)
        
        # Output: 84 → 10
        self.fc2 = nn.Linear(84, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = torch.relu(self.conv1(x))  # 32→28
        x = self.pool1(x)               # 28→14
        x = torch.relu(self.conv2(x))  # 14→10
        x = self.pool2(x)               # 10→5
        x = torch.relu(self.conv3(x))  # 5→1
        
        # Classification
        x = x.view(x.size(0), -1)      # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Crear y probar
model = LeNet5()
x = torch.randn(1, 1, 32, 32)
output = model(x)
print(f"Input: {x.shape}, Output: {output.shape}")

# Contar parámetros
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parámetros: {total_params:,}")  # ~61,706
```

---

### 2. VGG (2014)

VGGNet demostró que la **profundidad** es crucial. Ganó el segundo lugar en ImageNet 2014.

#### Historia

- **Creadores**: Visual Geometry Group (Oxford)
- **Año**: 2014
- **Paper**: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- **Innovación**: Usar solo filtros 3×3

#### La Filosofía 3×3

¿Por qué solo filtros 3×3?

```
Dos capas 3×3 equivalen a una capa 5×5:
┌───────────┐
│ 3×3 │ 3×3 │  =  │ 5×5 │
│  →  │  →  │     │     │
└───────────┘

Campo receptivo: 3 + (3-1) = 5

Pero con menos parámetros:
- 2 × (3×3×C×C) = 18C²
- 1 × (5×5×C×C) = 25C²

¡Y más no-linealidades (2 ReLUs vs 1)!
```

#### Arquitecturas VGG

| Modelo | Capas | Descripción |
|--------|-------|-------------|
| VGG-11 | 11 | 8 conv + 3 FC |
| VGG-13 | 13 | 10 conv + 3 FC |
| VGG-16 | 16 | 13 conv + 3 FC |
| VGG-19 | 19 | 16 conv + 3 FC |

#### VGG-16 Detallado

```
Bloque 1: 2× Conv 64  + MaxPool    [224→112]
Bloque 2: 2× Conv 128 + MaxPool    [112→56]
Bloque 3: 3× Conv 256 + MaxPool    [56→28]
Bloque 4: 3× Conv 512 + MaxPool    [28→14]
Bloque 5: 3× Conv 512 + MaxPool    [14→7]
FC: 4096 → 4096 → 1000

Total parámetros: ~138 millones
```

#### Implementación VGG-16

```python
import torch
import torch.nn as nn

def make_vgg_block(in_channels: int, out_channels: int, num_convs: int) -> nn.Sequential:
    """Crea un bloque VGG: N convs + maxpool."""
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(
            in_channels if i == 0 else out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        ))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    """
    VGG-16 para ImageNet.
    
    Arquitectura:
    - 5 bloques convolucionales
    - 3 capas fully connected
    - 138M parámetros
    """
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Bloque 1: 3→64, 224→112
            make_vgg_block(3, 64, 2),
            # Bloque 2: 64→128, 112→56
            make_vgg_block(64, 128, 2),
            # Bloque 3: 128→256, 56→28
            make_vgg_block(128, 256, 3),
            # Bloque 4: 256→512, 28→14
            make_vgg_block(256, 512, 3),
            # Bloque 5: 512→512, 14→7
            make_vgg_block(512, 512, 3),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Crear modelo
model = VGG16(num_classes=10)

# Verificar con imagen pequeña (adaptar para CIFAR)
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"Input: {x.shape}, Output: {output.shape}")
```

#### VGG para CIFAR-10 (adaptado)

```python
class VGG_CIFAR(nn.Module):
    """VGG simplificado para CIFAR-10 (32×32)."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Bloque 1: 32→16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Bloque 2: 16→8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Bloque 3: 8→4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Bloque 4: 4→2
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Para CIFAR-10
model_cifar = VGG_CIFAR(num_classes=10)
x = torch.randn(1, 3, 32, 32)
print(f"VGG-CIFAR output: {model_cifar(x).shape}")
```

---

### 3. Comparación LeNet vs VGG

| Aspecto | LeNet-5 | VGG-16 |
|---------|---------|--------|
| **Año** | 1998 | 2014 |
| **Entrada** | 32×32×1 | 224×224×3 |
| **Profundidad** | 7 capas | 16 capas |
| **Parámetros** | ~60K | ~138M |
| **Filtros** | 5×5 | 3×3 |
| **Activación** | Tanh | ReLU |
| **Pooling** | Average | Max |
| **Dataset** | MNIST | ImageNet |

---

### 4. Principios de Diseño

#### Patrón Común en CNNs

```
INICIO                              FIN
Espacial grande, pocos canales  →   Espacial pequeño, muchos canales

[224×224×3] → [112×112×64] → [56×56×128] → [28×28×256] → [14×14×512] → [7×7×512]

Canales:   3  →   64   →   128   →   256   →   512   →   512
Espacial: 224 →  112   →    56   →    28   →    14   →     7
```

#### Reglas de Diseño

1. **Duplicar canales al reducir espacialidad**
   ```
   Conv 64 → Pool → Conv 128 → Pool → Conv 256
   ```

2. **Usar filtros pequeños y profundidad**
   ```
   3×3 + 3×3 > 5×5 (mismo campo receptivo, menos params)
   ```

3. **Batch Normalization** (post-VGG)
   ```python
   Conv → BatchNorm → ReLU
   ```

4. **Global Average Pooling** (post-VGG)
   ```python
   # En lugar de Flatten + FC grandes
   nn.AdaptiveAvgPool2d(1)  # Reduce a 1×1
   ```

---

### 5. Usando Modelos Pre-entrenados

```python
import torch
from torchvision import models

# VGG-16 pre-entrenado en ImageNet
vgg16 = models.vgg16(weights='IMAGENET1K_V1')

# Ver arquitectura
print(vgg16)

# Usar solo features (transfer learning)
vgg_features = vgg16.features

# Congelar pesos
for param in vgg_features.parameters():
    param.requires_grad = False

# Adaptar para nueva tarea
class CustomVGG(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Para 10 clases
model = CustomVGG(num_classes=10)
```

---

### 6. Evolución Post-VGG

```
VGG (2014)      ResNet (2015)     DenseNet (2016)
    │               │                  │
    │           Skip Connections   Dense Connections
    │               │                  │
    ▼               ▼                  ▼
[Conv-Conv]     [Conv + Input]    [Conv concat all]
[Pool]          [Pool]            [Pool]

Problema:       Solución:         Beneficio:
Degradación     Residual Learning Feature Reuse
con profundidad permite 152+      eficiencia params
```

---

## 📊 Resumen

| Arquitectura | Innovación Principal | Lección |
|--------------|---------------------|---------|
| **LeNet** | Primera CNN funcional | Convoluciones funcionan |
| **AlexNet** | GPU, ReLU, Dropout | Escala con hardware |
| **VGG** | Profundidad con 3×3 | Más profundo = mejor |
| **ResNet** | Skip connections | Entrenar redes muy profundas |

---

## ✅ Checklist de Verificación

- [ ] Conozco la arquitectura de LeNet-5
- [ ] Entiendo por qué VGG usa solo filtros 3×3
- [ ] Puedo implementar ambas arquitecturas en PyTorch
- [ ] Sé usar modelos pre-entrenados de torchvision
- [ ] Entiendo el patrón "más canales, menos espacial"

---

_Siguiente: [Práctica - Convolución Manual](../2-practicas/ejercicio-01-convolucion-manual/)_
