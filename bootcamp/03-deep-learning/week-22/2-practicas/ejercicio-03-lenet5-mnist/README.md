# 🧠 Ejercicio 03: LeNet-5 en MNIST

## 🎯 Objetivo

Implementar la arquitectura LeNet-5 original y entrenarla en el dataset MNIST, aplicando todo lo aprendido sobre CNNs.

---

## 📋 Instrucciones

Implementarás LeNet-5 paso a paso, desde la arquitectura hasta el entrenamiento completo.

---

## Paso 1: Cargar Dataset MNIST

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Transformaciones (LeNet-5 original usa 32×32)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # LeNet-5 espera 32×32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST stats
])

# Descargar datos
train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

## Paso 2: Arquitectura LeNet-5

La arquitectura original de 1998:

```python
class LeNet5(nn.Module):
    """
    LeNet-5 (LeCun et al., 1998)
    
    Arquitectura:
    - Input: 32×32×1
    - C1: Conv 5×5, 6 filtros -> 28×28×6
    - S2: AvgPool 2×2 -> 14×14×6
    - C3: Conv 5×5, 16 filtros -> 10×10×16
    - S4: AvgPool 2×2 -> 5×5×16
    - C5: Conv 5×5, 120 filtros -> 1×1×120
    - F6: FC 120 -> 84
    - Output: FC 84 -> 10
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Capas convolucionales
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)      # C1
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)     # C3
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)   # C5
        
        # Pooling (original usa average, aquí también)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Capas fully connected
        self.fc1 = nn.Linear(120, 84)   # F6
        self.fc2 = nn.Linear(84, num_classes)  # Output
        
        # Activación
        self.activation = nn.Tanh()  # Original usa tanh
    
    def forward(self, x):
        # C1 -> S2
        x = self.pool(self.activation(self.conv1(x)))
        # C3 -> S4
        x = self.pool(self.activation(self.conv2(x)))
        # C5
        x = self.activation(self.conv3(x))
        # Flatten
        x = x.view(x.size(0), -1)
        # F6
        x = self.activation(self.fc1(x))
        # Output
        x = self.fc2(x)
        return x
```

**Descomenta** la sección del Paso 2.

---

## Paso 3: LeNet-5 Moderna

Versión moderna con ReLU y mejoras:

```python
class LeNet5Modern(nn.Module):
    """LeNet-5 con mejoras modernas."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # C1: Conv 5×5, 6 filtros
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # C3: Conv 5×5, 16 filtros
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # C5: Conv 5×5, 120 filtros
            nn.Conv2d(16, 120, kernel_size=5),
            nn.BatchNorm2d(120),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

**Descomenta** la sección del Paso 3.

---

## Paso 4: Funciones de Entrenamiento

```python
def train_epoch(model, loader, criterion, optimizer, device):
    """Entrena una época."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    """Evalúa el modelo."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total
```

**Descomenta** la sección del Paso 4.

---

## Paso 5: Loop de Entrenamiento

```python
# Configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando: {device}")

# Modelo
model = LeNet5Modern(num_classes=10).to(device)
print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")

# Optimizador y loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Entrenamiento
num_epochs = 10
best_acc = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )
    scheduler.step()
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'lenet5_best.pth')
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

print(f"\nMejor Accuracy: {best_acc:.2f}%")
```

**Descomenta** la sección del Paso 5.

---

## Paso 6: Visualizar Filtros Aprendidos

```python
import matplotlib.pyplot as plt

def visualize_filters(model, layer_name='conv1'):
    """Visualiza filtros de la primera capa."""
    # Obtener pesos
    conv_layer = getattr(model, layer_name, None)
    if conv_layer is None:
        conv_layer = model.features[0]
    
    weights = conv_layer.weight.data.cpu()
    
    # Plot
    n_filters = weights.shape[0]
    fig, axes = plt.subplots(1, n_filters, figsize=(12, 2))
    
    for i, ax in enumerate(axes):
        w = weights[i, 0]  # Primer canal
        ax.imshow(w, cmap='gray')
        ax.axis('off')
        ax.set_title(f'F{i+1}')
    
    plt.suptitle(f'Filtros de {layer_name}')
    plt.tight_layout()
    plt.savefig('lenet5_filters.png', dpi=150)
    plt.close()
    print("Filtros guardados en 'lenet5_filters.png'")

visualize_filters(model)
```

**Descomenta** la sección del Paso 6.

---

## Paso 7: Visualizar Feature Maps

```python
def visualize_feature_maps(model, image, device):
    """Visualiza activaciones de la primera capa conv."""
    model.eval()
    
    # Forward hasta primera conv
    with torch.no_grad():
        x = image.unsqueeze(0).to(device)
        if hasattr(model, 'conv1'):
            features = torch.relu(model.conv1(x))
        else:
            features = model.features[:3](x)  # Conv+BN+ReLU
    
    features = features.cpu().squeeze(0)
    
    # Plot
    n_maps = features.shape[0]
    fig, axes = plt.subplots(1, n_maps + 1, figsize=(14, 2))
    
    # Imagen original
    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Feature maps
    for i in range(n_maps):
        axes[i+1].imshow(features[i], cmap='viridis')
        axes[i+1].axis('off')
        axes[i+1].set_title(f'FM{i+1}')
    
    plt.suptitle('Feature Maps (capa 1)')
    plt.tight_layout()
    plt.savefig('lenet5_feature_maps.png', dpi=150)
    plt.close()
    print("Feature maps guardados en 'lenet5_feature_maps.png'")

# Obtener una imagen de test
test_image, _ = test_dataset[0]
visualize_feature_maps(model, test_image, device)
```

**Descomenta** la sección del Paso 7.

---

## ✅ Verificación

Al completar el ejercicio deberías:

- [ ] Cargar y preprocesar MNIST para LeNet-5 (32×32)
- [ ] Implementar LeNet-5 original (tanh, avgpool)
- [ ] Implementar LeNet-5 moderna (ReLU, BatchNorm, MaxPool)
- [ ] Entrenar el modelo y alcanzar >98% accuracy
- [ ] Visualizar filtros aprendidos
- [ ] Visualizar feature maps

**Meta**: Accuracy ≥ 98% en MNIST test set.

---

## 🔗 Navegación

[← Ejercicio Anterior](../ejercicio-02-cnn-pytorch/) | [Proyecto →](../../3-proyecto/)
