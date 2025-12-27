# 🧠 Ejercicio 02: CNN en PyTorch

## 🎯 Objetivo

Construir una CNN desde cero usando PyTorch, entendiendo cada componente y cómo fluyen los datos.

---

## 📋 Instrucciones

Construirás una CNN paso a paso, agregando capas y verificando las dimensiones en cada paso.

---

## Paso 1: Capa Convolucional Básica

`nn.Conv2d` es la capa fundamental de las CNNs:

```python
import torch
import torch.nn as nn

# Crear una capa convolucional
conv = nn.Conv2d(
    in_channels=1,      # Canales de entrada (grayscale=1, RGB=3)
    out_channels=32,    # Número de filtros
    kernel_size=3,      # Tamaño 3×3
    stride=1,           # Paso 1
    padding=1           # Padding para mantener tamaño
)

# Entrada: (batch, channels, height, width)
x = torch.randn(4, 1, 28, 28)
output = conv(x)

print(f"Input: {x.shape}")
print(f"Output: {output.shape}")
print(f"Parámetros: {conv.weight.shape}")
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

## Paso 2: Pooling y Activaciones

Combina convolución con ReLU y MaxPool:

```python
# Bloque típico: Conv -> ReLU -> Pool
block = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)
)

x = torch.randn(4, 1, 28, 28)
output = block(x)

print(f"Input: {x.shape}")     # [4, 1, 28, 28]
print(f"Output: {output.shape}")  # [4, 32, 14, 14]
```

**Descomenta** la sección del Paso 2.

---

## Paso 3: Calcular Dimensiones de Flatten

Antes de las capas FC, necesitamos saber el tamaño del tensor aplanado:

```python
def calculate_flatten_size(input_shape, conv_layers):
    """
    Calcula el tamaño después de flatten.
    
    Args:
        input_shape: (C, H, W)
        conv_layers: nn.Sequential con capas conv/pool
    """
    x = torch.randn(1, *input_shape)
    with torch.no_grad():
        output = conv_layers(x)
    return output.numel()

# Ejemplo
features = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2)
)

flat_size = calculate_flatten_size((1, 28, 28), features)
print(f"Tamaño flatten: {flat_size}")  # 64 * 7 * 7 = 3136
```

**Descomenta** la sección del Paso 3.

---

## Paso 4: CNN Completa como Clase

Define una CNN completa heredando de `nn.Module`:

```python
class SimpleCNN(nn.Module):
    """CNN simple para clasificación de imágenes 28×28."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Extractor de características
        self.features = nn.Sequential(
            # Bloque 1: 1 -> 32 canales, 28 -> 14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Bloque 2: 32 -> 64 canales, 14 -> 7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

**Descomenta** la sección del Paso 4.

---

## Paso 5: Inspeccionar el Modelo

Verifica la arquitectura y cuenta parámetros:

```python
model = SimpleCNN(num_classes=10)

# Ver arquitectura
print(model)

# Contar parámetros
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parámetros: {count_parameters(model):,}")

# Ver parámetros por capa
for name, param in model.named_parameters():
    print(f"{name}: {param.shape} = {param.numel():,}")
```

**Descomenta** la sección del Paso 5.

---

## Paso 6: Forward Pass y Dimensiones

Verifica el flujo de datos paso a paso:

```python
def trace_dimensions(model, input_tensor):
    """Muestra dimensiones en cada capa."""
    x = input_tensor
    print(f"Input: {x.shape}")
    
    for i, layer in enumerate(model.features):
        x = layer(x)
        print(f"  {layer.__class__.__name__}: {x.shape}")
    
    x = model.classifier[0](x)  # Flatten
    print(f"  Flatten: {x.shape}")
    
    for i, layer in enumerate(model.classifier[1:], 1):
        x = layer(x)
        print(f"  {layer.__class__.__name__}: {x.shape}")
    
    return x

# Trazar
x = torch.randn(1, 1, 28, 28)
output = trace_dimensions(model, x)
```

**Descomenta** la sección del Paso 6.

---

## Paso 7: Batch Normalization

Agrega BatchNorm para mejorar el entrenamiento:

```python
class CNNWithBatchNorm(nn.Module):
    """CNN con Batch Normalization."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Bloque 1 con BatchNorm
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Bloque 2 con BatchNorm
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Bloque 3 con BatchNorm
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

**Descomenta** la sección del Paso 7.

---

## ✅ Verificación

Al completar el ejercicio deberías poder:

- [ ] Crear capas convolucionales con `nn.Conv2d`
- [ ] Combinar Conv, ReLU y Pool en bloques
- [ ] Calcular dimensiones de flatten correctamente
- [ ] Definir CNNs completas como clases
- [ ] Inspeccionar parámetros del modelo
- [ ] Usar BatchNorm en CNNs

---

## 🔗 Navegación

[← Ejercicio Anterior](../ejercicio-01-convolucion-manual/) | [Siguiente Ejercicio →](../ejercicio-03-lenet5-mnist/)
