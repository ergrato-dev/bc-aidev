# Ejercicio 01: Implementación de Bloques Residuales

## 🎯 Objetivo

Implementar desde cero los bloques BasicBlock y Bottleneck de ResNet para comprender cómo funcionan las conexiones residuales.

---

## 📋 Conceptos Clave

- **Skip Connection**: Conexión que suma la entrada directamente a la salida
- **BasicBlock**: 2 convoluciones 3×3 (ResNet-18/34)
- **Bottleneck**: 1×1 → 3×3 → 1×1 (ResNet-50+)
- **Downsample**: Ajusta dimensiones cuando stride > 1

---

## 🔧 Paso 1: Configuración del Entorno

Abre `starter/main.py` y ejecuta la primera sección para verificar las importaciones:

```python
import torch
import torch.nn as nn

# Verificar GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo: {device}')
```

---

## 🔧 Paso 2: BasicBlock sin Skip Connection

Primero implementamos un bloque SIN conexión residual para ver la diferencia:

```python
class PlainBlock(nn.Module):
    """Bloque sin conexión residual."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out)  # Sin skip connection
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## 🔧 Paso 3: BasicBlock con Skip Connection

Ahora añadimos la conexión residual:

```python
class BasicBlock(nn.Module):
    """Bloque residual básico (ResNet-18/34)."""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # ¡Skip connection!
        return self.relu(out)
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## 🔧 Paso 4: Bottleneck Block

Implementamos el bloque más eficiente para redes profundas:

```python
class Bottleneck(nn.Module):
    """Bloque bottleneck (ResNet-50/101/152)."""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 1×1: Reducir canales
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3×3: Procesamiento espacial
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1×1: Expandir canales
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return self.relu(out)
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## 🔧 Paso 5: Comparar Parámetros

Comparamos la eficiencia de cada bloque:

```python
def count_parameters(model):
    """Cuenta parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Crear bloques con 64 canales de entrada
plain = PlainBlock(64, 64)
basic = BasicBlock(64, 64)
bottleneck = Bottleneck(64, 64)

print(f'PlainBlock:  {count_parameters(plain):,} parámetros')
print(f'BasicBlock:  {count_parameters(basic):,} parámetros')
print(f'Bottleneck:  {count_parameters(bottleneck):,} parámetros')
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## 🔧 Paso 6: Verificar Flujo de Gradientes

Probamos que los gradientes fluyen correctamente:

```python
def test_gradient_flow(block, name):
    """Verifica que los gradientes fluyen a través del bloque."""
    x = torch.randn(1, 64, 32, 32, requires_grad=True)
    y = block(x)
    loss = y.sum()
    loss.backward()
    
    grad_norm = x.grad.norm().item()
    print(f'{name}: grad_norm = {grad_norm:.4f}')
    return grad_norm

# Probar cada bloque
test_gradient_flow(PlainBlock(64, 64), 'PlainBlock')
test_gradient_flow(BasicBlock(64, 64), 'BasicBlock')
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## ✅ Checklist de Verificación

- [ ] PlainBlock implementado y funcionando
- [ ] BasicBlock con skip connection implementado
- [ ] Bottleneck block implementado
- [ ] Conteo de parámetros correcto
- [ ] Gradientes fluyen a través de los bloques

---

## 📚 Recursos

- [Deep Residual Learning (Paper)](https://arxiv.org/abs/1512.03385)
- [PyTorch ResNet Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
