# 🔄 Ejercicio 03: Callbacks y Checkpoints

## 🎯 Objetivo

Implementar un sistema de callbacks incluyendo Early Stopping, Model Checkpoint y Logger de métricas.

---

## 📋 Instrucciones

### Paso 1: Configuración Inicial

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from pathlib import Path
```

**Abre `starter/main.py`** y descomenta la sección de imports.

---

### Paso 2: Clase EarlyStopping

Implementamos Early Stopping para detener el entrenamiento cuando no hay mejora:

```python
class EarlyStopping:
    """Detiene el entrenamiento si no hay mejora."""
    
    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        """
        Args:
            patience: Épocas sin mejora antes de parar
            min_delta: Cambio mínimo para considerar mejora
            mode: 'min' para loss, 'max' para accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            print(f'  EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def _is_improvement(self, current):
        if self.mode == 'min':
            return current < self.best_value - self.min_delta
        return current > self.best_value + self.min_delta
```

**Descomenta** la clase EarlyStopping.

---

### Paso 3: Clase ModelCheckpoint

Guardamos el mejor modelo durante el entrenamiento:

```python
class ModelCheckpoint:
    """Guarda el mejor modelo."""
    
    def __init__(self, filepath, monitor='val_loss', mode='min'):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
    
    def __call__(self, model, current_value, epoch):
        is_best = (self.mode == 'min' and current_value < self.best_value) or \
                  (self.mode == 'max' and current_value > self.best_value)
        
        if is_best:
            self.best_value = current_value
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_value': self.best_value,
            }, self.filepath)
            print(f'  Checkpoint: Guardado mejor modelo ({self.monitor}={current_value:.4f})')
        
        return is_best
```

**Descomenta** la clase ModelCheckpoint.

---

### Paso 4: Clase MetricsLogger

Registramos métricas para visualización:

```python
class MetricsLogger:
    """Registra métricas durante entrenamiento."""
    
    def __init__(self):
        self.history = {}
    
    def log(self, metrics_dict):
        for name, value in metrics_dict.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)
    
    def plot(self, metrics=None, figsize=(12, 4)):
        metrics = metrics or list(self.history.keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            ax.plot(self.history[metric], marker='o')
            ax.set_title(metric)
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
```

**Descomenta** la clase MetricsLogger.

---

### Paso 5: Modelo y Datos

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)

# Datos con split de validación
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1000)
```

**Descomenta** la sección de modelo y datos.

---

### Paso 6: Training Loop con Callbacks

```python
def train_with_callbacks(model, optimizer, epochs=50):
    """Training loop completo con callbacks."""
    criterion = nn.CrossEntropyLoss()
    
    # Inicializar callbacks
    early_stop = EarlyStopping(patience=5, mode='min')
    checkpoint = ModelCheckpoint('checkpoints/best_model.pth', monitor='val_loss', mode='min')
    logger = MetricsLogger()
    
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (output.argmax(1) == y).sum().item()
            train_total += y.size(0)
        
        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                
                val_loss += loss.item()
                val_correct += (output.argmax(1) == y).sum().item()
                val_total += y.size(0)
        
        # Calcular métricas
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Log métricas
        logger.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        })
        
        print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')
        
        # Callbacks
        checkpoint(model, val_loss, epoch)
        
        if early_stop(val_loss):
            print(f'\n¡Early stopping en época {epoch+1}!')
            break
    
    return logger

# Entrenar
model = SimpleNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

logger = train_with_callbacks(model, optimizer, epochs=50)

# Visualizar
fig = logger.plot(['train_loss', 'val_loss'])
plt.savefig('training_history.png', dpi=150)
plt.show()
```

**Descomenta** la sección de training loop.

---

## 🔍 Preguntas de Reflexión

1. ¿Por qué Early Stopping ayuda a prevenir overfitting?
2. ¿Qué pasa si `patience` es muy bajo o muy alto?
3. ¿Por qué guardamos el `model_state_dict` y no todo el modelo?
4. ¿Cuándo se activó el Early Stopping en tu entrenamiento?

---

## ✅ Checklist

- [ ] EarlyStopping implementado
- [ ] ModelCheckpoint guardando mejor modelo
- [ ] MetricsLogger registrando métricas
- [ ] Gradient clipping aplicado
- [ ] Early stopping deteniendo entrenamiento
- [ ] Gráficas de entrenamiento generadas

---

## 📚 Recursos

- [PyTorch Saving Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Early Stopping](https://en.wikipedia.org/wiki/Early_stopping)
