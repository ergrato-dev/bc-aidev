# 📖 Glosario - Semana 21: PyTorch

Términos clave de PyTorch ordenados alfabéticamente.

---

## A

### Autograd
Sistema de diferenciación automática de PyTorch. Calcula gradientes automáticamente registrando operaciones en un grafo computacional dinámico.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()  # Autograd calcula dy/dx = 4
```

---

## B

### Backward
Método que calcula gradientes propagando hacia atrás a través del grafo computacional.

```python
loss.backward()  # Calcula gradientes de todos los tensores con requires_grad=True
```

### Batch
Conjunto de muestras procesadas juntas en una iteración de entrenamiento.

```python
dataloader = DataLoader(dataset, batch_size=32)
```

---

## C

### Computational Graph
Estructura de datos que representa las operaciones matemáticas y sus dependencias. En PyTorch es dinámico (define-by-run).

### CrossEntropyLoss
Función de pérdida para clasificación multiclase. Combina LogSoftmax y NLLLoss.

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, targets)  # targets son índices de clase
```

### CUDA
Plataforma de NVIDIA para computación en GPU. PyTorch usa CUDA para acelerar operaciones.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## D

### DataLoader
Clase que proporciona iteración eficiente sobre datasets con batching, shuffling y carga paralela.

```python
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
```

### Dataset
Clase abstracta que representa un conjunto de datos. Debe implementar `__len__` y `__getitem__`.

### Define-by-Run
Paradigma donde el grafo computacional se construye dinámicamente durante la ejecución, no antes.

### Detach
Método que crea una copia del tensor sin conexión al grafo computacional.

```python
frozen = tensor.detach()  # No propaga gradientes
```

### Dropout
Técnica de regularización que desactiva neuronas aleatoriamente durante entrenamiento.

```python
dropout = nn.Dropout(p=0.5)  # 50% de probabilidad de desactivar
```

### dtype
Tipo de dato de un tensor (float32, int64, etc.).

```python
tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
```

---

## E

### Epoch
Una pasada completa por todo el dataset de entrenamiento.

### eval()
Método que pone el modelo en modo evaluación. Desactiva Dropout y usa estadísticas guardadas en BatchNorm.

```python
model.eval()
with torch.no_grad():
    predictions = model(test_data)
```

---

## F

### Forward
Método que define cómo los datos fluyen a través del modelo.

```python
def forward(self, x):
    return self.fc(x)
```

### Functional API (F)
Módulo `torch.nn.functional` con funciones sin estado (activaciones, pérdidas).

```python
import torch.nn.functional as F
x = F.relu(x)
```

---

## G

### Gradient
Derivada parcial de una función respecto a sus parámetros. Indica la dirección de máximo crecimiento.

### grad
Atributo de un tensor donde se almacenan los gradientes después de `backward()`.

```python
x.grad  # Gradiente de la pérdida respecto a x
```

### GPU
Graphics Processing Unit. Acelera operaciones matriciales del deep learning.

---

## L

### Learning Rate
Hiperparámetro que controla el tamaño del paso en la actualización de parámetros.

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Linear (nn.Linear)
Capa completamente conectada. Aplica transformación $y = xW^T + b$.

```python
fc = nn.Linear(in_features=784, out_features=128)
```

### Loss Function
Función que mide el error entre predicciones y valores reales.

---

## M

### Module (nn.Module)
Clase base para todos los componentes de redes neuronales en PyTorch.

```python
class MiRed(nn.Module):
    def __init__(self):
        super().__init__()
```

---

## N

### no_grad
Context manager que desactiva el cálculo de gradientes. Usado para inferencia.

```python
with torch.no_grad():
    output = model(data)
```

### numel
Método que retorna el número total de elementos en un tensor.

```python
tensor.numel()  # Total de elementos
```

---

## O

### Optimizer
Algoritmo que actualiza los parámetros del modelo usando gradientes.

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer.step()  # Actualiza parámetros
```

---

## P

### Parameters
Tensores entrenables de un modelo. Accesibles via `model.parameters()`.

### PyTorch
Framework de deep learning desarrollado por Meta AI. Conocido por su grafo dinámico y sintaxis pythónica.

---

## R

### ReLU
Rectified Linear Unit. Función de activación $f(x) = \max(0, x)$.

```python
relu = nn.ReLU()
# o
x = F.relu(x)
```

### requires_grad
Flag que indica si un tensor necesita gradientes calculados.

```python
x = torch.tensor([1.0], requires_grad=True)
```

### Reshape
Cambiar la forma de un tensor sin modificar sus datos.

```python
tensor.reshape(3, 4)
tensor.view(3, 4)
```

---

## S

### Squeeze
Elimina dimensiones de tamaño 1 de un tensor.

```python
t = torch.rand(1, 3, 1)
t.squeeze()  # Shape: [3]
```

### state_dict
Diccionario que mapea nombres de parámetros a tensores. Usado para guardar/cargar modelos.

```python
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

### Step
Método del optimizador que actualiza los parámetros usando los gradientes calculados.

```python
optimizer.step()
```

---

## T

### Tensor
Array multidimensional, estructura de datos fundamental en PyTorch.

```python
tensor = torch.tensor([[1, 2], [3, 4]])
```

### to()
Método para mover tensores o modelos a un dispositivo específico.

```python
tensor.to('cuda')
model.to(device)
```

### train()
Método que pone el modelo en modo entrenamiento. Activa Dropout y BatchNorm usa estadísticas del batch.

```python
model.train()
```

---

## U

### Unsqueeze
Añade una dimensión de tamaño 1 en la posición especificada.

```python
t = torch.rand(3)
t.unsqueeze(0)  # Shape: [1, 3]
```

---

## V

### View
Retorna un tensor con diferente shape pero compartiendo memoria con el original.

```python
tensor.view(3, 4)  # Requiere memoria contigua
```

---

## Z

### zero_grad
Método que resetea los gradientes a cero. Necesario antes de cada backward.

```python
optimizer.zero_grad()
```

---

_Semana 21 | Módulo 3: Deep Learning | Bootcamp IA: Zero to Hero_
