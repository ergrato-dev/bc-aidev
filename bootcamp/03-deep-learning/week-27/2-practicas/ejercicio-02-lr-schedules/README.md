# 📈 Ejercicio 02: Learning Rate Schedules

## 🎯 Objetivo

Implementar y comparar diferentes estrategias de scheduling del learning rate: StepLR, CosineAnnealingLR, OneCycleLR.

---

## 📋 Instrucciones

### Paso 1: Configuración Inicial

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
```

**Abre `starter/main.py`** y descomenta la sección de imports.

---

### Paso 2: Modelo y Datos

Usamos el mismo modelo y datos que el ejercicio anterior:

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Cargar datos
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

**Descomenta** la sección de modelo y datos.

---

### Paso 3: Función de Entrenamiento con Scheduler

Modificamos la función para incluir el scheduler y registrar el learning rate:

```python
def train_with_scheduler(scheduler_name, model, optimizer, scheduler, epochs=10, step_per_batch=False):
    """
    Entrena con un scheduler específico.
    
    Args:
        step_per_batch: Si True, hace scheduler.step() cada batch (para OneCycleLR)
    """
    criterion = nn.CrossEntropyLoss()
    history = {'loss': [], 'lr': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # OneCycleLR hace step por batch
            if step_per_batch:
                scheduler.step()
        
        # Los demás schedulers hacen step por época
        if not step_per_batch:
            scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['loss'].append(epoch_loss)
        history['lr'].append(current_lr)
        
        print(f'{scheduler_name} - Epoch {epoch+1}: loss={epoch_loss:.4f}, lr={current_lr:.6f}')
    
    return history
```

**Descomenta** la función de entrenamiento.

---

### Paso 4: Configurar Schedulers

```python
EPOCHS = 10
LR_INICIAL = 0.1

schedulers_config = {
    'StepLR': {
        'scheduler': lambda opt: StepLR(opt, step_size=3, gamma=0.5),
        'step_per_batch': False
    },
    'CosineAnnealing': {
        'scheduler': lambda opt: CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=0.001),
        'step_per_batch': False
    },
    'OneCycleLR': {
        'scheduler': lambda opt: OneCycleLR(
            opt, 
            max_lr=0.1, 
            epochs=EPOCHS, 
            steps_per_epoch=len(train_loader)
        ),
        'step_per_batch': True  # ¡Importante!
    },
}
```

**Descomenta** la configuración de schedulers.

---

### Paso 5: Entrenar con Cada Scheduler

```python
results = {}

for name, config in schedulers_config.items():
    print(f'\n{"="*50}')
    print(f'Entrenando con {name}')
    print("="*50)
    
    # Nuevo modelo y optimizador para cada experimento
    model = SimpleNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR_INICIAL)
    scheduler = config['scheduler'](optimizer)
    
    results[name] = train_with_scheduler(
        name, model, optimizer, scheduler, 
        epochs=EPOCHS, 
        step_per_batch=config['step_per_batch']
    )
```

**Descomenta** la sección de entrenamiento.

---

### Paso 6: Visualizar Learning Rate

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfica de Learning Rate
for name, history in results.items():
    axes[0].plot(history['lr'], label=name, marker='o')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Learning Rate')
axes[0].set_title('Evolución del Learning Rate')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# Gráfica de Loss
for name, history in results.items():
    axes[1].plot(history['loss'], label=name, marker='o')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Evolución del Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lr_schedules_comparison.png', dpi=150)
plt.show()
```

**Descomenta** la sección de visualización.

---

## 🔍 Preguntas de Reflexión

1. ¿Por qué OneCycleLR sube el LR al inicio?
2. ¿Cuál scheduler produjo mejor loss final?
3. ¿Por qué OneCycleLR requiere `step_per_batch=True`?
4. ¿Cuándo usarías CosineAnnealing vs StepLR?

---

## ✅ Checklist

- [ ] Imports configurados
- [ ] Modelo y datos cargados
- [ ] Función de entrenamiento con scheduler
- [ ] 3 schedulers configurados correctamente
- [ ] OneCycleLR con step por batch
- [ ] Gráficas de LR y Loss generadas

---

## 📚 Recursos

- [PyTorch LR Schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [OneCycleLR Paper](https://arxiv.org/abs/1708.07120)
