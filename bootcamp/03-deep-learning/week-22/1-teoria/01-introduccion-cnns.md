# 🖼️ Introducción a Redes Neuronales Convolucionales

## 🎯 Objetivos

- Entender por qué las CNNs revolucionaron la visión por computadora
- Comprender las limitaciones de redes fully connected para imágenes
- Conocer la estructura general de una CNN
- Apreciar la inspiración biológica de las CNNs

---

## 📋 Contenido

### 1. El Problema de las Imágenes

Las redes fully connected tienen problemas graves con imágenes:

```python
# Una imagen pequeña de 28×28 píxeles (MNIST)
imagen_mnist = 28 * 28  # = 784 neuronas de entrada

# Una imagen de 224×224 RGB (ImageNet)
imagen_imagenet = 224 * 224 * 3  # = 150,528 neuronas de entrada

# Primera capa fully connected con 1000 neuronas
parametros_fc = 150_528 * 1000  # = 150,528,000 parámetros!
```

**Problemas de Fully Connected:**

| Problema | Descripción |
|----------|-------------|
| **Explosión de parámetros** | Millones de parámetros para imágenes pequeñas |
| **Sin estructura espacial** | Trata píxeles como independientes |
| **No invariante a traslación** | Un gato a la izquierda ≠ gato a la derecha |
| **Propenso a overfitting** | Demasiados parámetros, pocos datos |

---

### 2. La Solución: Convoluciones

Las CNNs resuelven estos problemas con tres ideas clave:

#### 2.1 Conexiones Locales

```
Fully Connected:              CNN:
Cada neurona conecta         Cada neurona conecta
con TODOS los píxeles        solo con una región local

[████████████████]           [██░░░░░░░░░░░░░░]
       ↓                            ↓
    [████]                       [██]
```

#### 2.2 Parámetros Compartidos

El mismo filtro se aplica en todas las posiciones:

```python
# Fully Connected: cada conexión tiene su propio peso
parametros_fc = entrada * salida

# CNN: el mismo kernel se usa en toda la imagen
parametros_conv = kernel_size * kernel_size * canales
# 3×3×3 = 27 parámetros vs millones
```

#### 2.3 Invarianza a Traslación

Un gato es un gato sin importar dónde esté en la imagen:

```
Imagen 1:        Imagen 2:        Mismo filtro:
[🐱░░░░]         [░░░░🐱]         [✓ detecta gato]
[░░░░░░]         [░░░░░░]         [en ambas]
```

---

### 3. Inspiración Biológica

Las CNNs están inspiradas en el córtex visual de los mamíferos:

#### Experimento de Hubel & Wiesel (1959)

Descubrieron que el córtex visual tiene:

- **Células simples**: Detectan bordes en orientaciones específicas
- **Células complejas**: Responden a patrones más abstractos
- **Jerarquía**: De características simples a complejas

```
Córtex Visual             CNN
─────────────             ───
V1: Bordes         →     Capa 1: Filtros de bordes
V2: Formas         →     Capa 2: Formas simples  
V4: Objetos        →     Capa 3: Partes de objetos
IT: Categorías     →     Capa final: Clasificación
```

---

### 4. Arquitectura General de una CNN

```
ENTRADA → [CONV → ReLU → POOL]×N → FLATTEN → FC → SALIDA

┌─────────┐   ┌──────────────────────┐   ┌─────────────┐
│ Imagen  │ → │ Extractor de Features│ → │ Clasificador│
│ 224×224 │   │ Conv + Pool layers   │   │ FC layers   │
└─────────┘   └──────────────────────┘   └─────────────┘
```

#### Componentes Principales

| Componente | Función | Ubicación |
|------------|---------|-----------|
| **Conv** | Extrae características locales | Capas iniciales |
| **ReLU** | Introduce no-linealidad | Después de cada conv |
| **Pool** | Reduce dimensionalidad | Después de bloques conv |
| **Flatten** | Convierte 3D → 1D | Transición a FC |
| **FC** | Clasificación final | Capas finales |

---

### 5. Evolución de las CNNs

```
1998        2012         2014         2015         2016
 │           │            │            │            │
 ▼           ▼            ▼            ▼            ▼
LeNet    AlexNet        VGG        ResNet      DenseNet
 │           │            │            │            │
 └── Dígitos │            │            │            │
             └── ImageNet │            │            │
                          └── Profundidad           │
                                       └── Skip Connections
                                                    └── Dense Connections
```

| Año | Modelo | Innovación | Capas |
|-----|--------|------------|-------|
| 1998 | LeNet-5 | Primera CNN práctica | 7 |
| 2012 | AlexNet | GPU, ReLU, Dropout | 8 |
| 2014 | VGG | Filtros 3×3 uniformes | 16-19 |
| 2015 | ResNet | Skip connections | 152 |
| 2016 | DenseNet | Conexiones densas | 201 |

---

### 6. Por qué Funcionan las CNNs

```python
# Las CNNs aprenden jerárquicamente
# Capa 1: Detecta bordes y texturas simples
# Capa 2: Combina bordes en formas (círculos, esquinas)
# Capa 3: Combina formas en partes (ojos, ruedas)
# Capa N: Combina partes en objetos (gatos, coches)

# Ejemplo conceptual
class ConceptualCNN:
    def forward(self, imagen):
        # Capa 1: Bordes
        bordes = self.conv1(imagen)  # "hay un borde vertical aquí"
        
        # Capa 2: Formas
        formas = self.conv2(bordes)  # "hay un círculo aquí"
        
        # Capa 3: Partes
        partes = self.conv3(formas)  # "hay un ojo aquí"
        
        # Clasificador
        clase = self.fc(partes)      # "esto es un gato"
        return clase
```

---

### 7. Ventajas de las CNNs

| Ventaja | Descripción |
|---------|-------------|
| **Eficiencia** | Menos parámetros que fully connected |
| **Invarianza** | Detecta features sin importar posición |
| **Jerarquía** | Aprende de simple a complejo |
| **Transfer Learning** | Features reutilizables entre tareas |

---

## 💻 Ejemplo: CNN Simple en PyTorch

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """CNN básica para clasificación de imágenes."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Extractor de características
        self.features = nn.Sequential(
            # Bloque 1: 1 → 32 canales
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduce a la mitad
            
            # Bloque 2: 32 → 64 canales
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduce a la mitad
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

# Crear modelo
model = SimpleCNN(num_classes=10)

# Entrada de ejemplo: batch de 4 imágenes 28×28 grayscale
x = torch.randn(4, 1, 28, 28)
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")  # [4, 10]
```

---

## 📚 Recursos Adicionales

- 📖 [CS231n: CNNs for Visual Recognition](https://cs231n.github.io/convolutional-networks/)
- 📄 Paper original LeNet: "Gradient-Based Learning Applied to Document Recognition" (LeCun, 1998)
- 🎥 [3Blue1Brown: But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA)

---

## ✅ Checklist de Verificación

- [ ] Entiendo por qué FC no escala para imágenes
- [ ] Conozco las tres ideas clave de las CNNs
- [ ] Sé qué hace cada componente (Conv, ReLU, Pool, FC)
- [ ] Puedo describir la jerarquía de features
- [ ] Conozco la evolución histórica de las CNNs

---

_Siguiente: [Operación de Convolución](02-operacion-convolucion.md)_
