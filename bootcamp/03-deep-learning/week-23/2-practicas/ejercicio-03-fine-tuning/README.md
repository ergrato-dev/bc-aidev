# Ejercicio 03: Estrategias de Fine-tuning

## 🎯 Objetivo

Implementar estrategias avanzadas de fine-tuning: congelamiento gradual, learning rates discriminativos y descongelamiento progresivo.

---

## 📋 Conceptos Clave

- **Gradual Unfreezing**: Descongelar capas progresivamente
- **Discriminative LR**: Diferentes learning rates por capa
- **Layer Groups**: Agrupar capas para diferentes configuraciones
- **Learning Rate Scheduling**: Ajustar LR durante entrenamiento

---

## 🔧 Paso 1: Configuración Inicial

```python
import torch
import torch.nn as nn
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar modelo
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## 🔧 Paso 2: Identificar Grupos de Capas

Agrupamos las capas de ResNet para fine-tuning diferenciado:

```python
def get_layer_groups(model):
    """Agrupa las capas de ResNet para fine-tuning."""
    groups = {
        'stem': [model.conv1, model.bn1],
        'layer1': [model.layer1],
        'layer2': [model.layer2],
        'layer3': [model.layer3],
        'layer4': [model.layer4],
        'head': [model.fc]
    }
    return groups

layer_groups = get_layer_groups(model)
for name, layers in layer_groups.items():
    params = sum(p.numel() for layer in layers for p in layer.parameters())
    print(f'{name}: {params:,} parámetros')
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## 🔧 Paso 3: Congelar por Grupos

Implementamos funciones para congelar/descongelar grupos:

```python
def freeze_group(group_layers):
    """Congela un grupo de capas."""
    for layer in group_layers:
        for param in layer.parameters():
            param.requires_grad = False

def unfreeze_group(group_layers):
    """Descongela un grupo de capas."""
    for layer in group_layers:
        for param in layer.parameters():
            param.requires_grad = True

def count_trainable(model):
    """Cuenta parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Congelar todo excepto head
for name, layers in layer_groups.items():
    if name != 'head':
        freeze_group(layers)

print(f'Parámetros entrenables: {count_trainable(model):,}')
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## 🔧 Paso 4: Discriminative Learning Rates

Asignamos diferentes learning rates a cada grupo:

```python
def get_optimizer_with_discriminative_lr(model, layer_groups, base_lr=1e-3, lr_mult=0.1):
    """
    Crea optimizer con LR discriminativos.
    
    Args:
        model: Modelo
        layer_groups: Diccionario de grupos de capas
        base_lr: LR para la cabeza (clasificador)
        lr_mult: Multiplicador para capas anteriores
    
    Returns:
        Optimizer con parámetros agrupados
    """
    param_groups = []
    
    # De más profundo a más superficial
    group_names = ['head', 'layer4', 'layer3', 'layer2', 'layer1', 'stem']
    
    for i, name in enumerate(group_names):
        if name not in layer_groups:
            continue
        
        layers = layer_groups[name]
        lr = base_lr * (lr_mult ** i)
        
        params = []
        for layer in layers:
            params.extend([p for p in layer.parameters() if p.requires_grad])
        
        if params:
            param_groups.append({
                'params': params,
                'lr': lr,
                'name': name
            })
            print(f'{name}: lr = {lr:.2e}')
    
    return torch.optim.Adam(param_groups)

# Primero descongelar todo
for layers in layer_groups.values():
    unfreeze_group(layers)

optimizer = get_optimizer_with_discriminative_lr(model, layer_groups)
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## 🔧 Paso 5: Gradual Unfreezing

Implementamos descongelamiento progresivo durante el entrenamiento:

```python
class GradualUnfreezer:
    """Gestiona el descongelamiento gradual de capas."""
    
    def __init__(self, model, layer_groups, unfreeze_schedule):
        """
        Args:
            model: Modelo
            layer_groups: Diccionario de grupos
            unfreeze_schedule: Dict {epoch: [grupos a descongelar]}
        """
        self.model = model
        self.layer_groups = layer_groups
        self.schedule = unfreeze_schedule
        
        # Inicialmente congelar todo excepto head
        for name, layers in layer_groups.items():
            if name != 'head':
                freeze_group(layers)
    
    def step(self, epoch):
        """Ejecuta descongelamiento según schedule."""
        if epoch in self.schedule:
            groups_to_unfreeze = self.schedule[epoch]
            for group_name in groups_to_unfreeze:
                if group_name in self.layer_groups:
                    unfreeze_group(self.layer_groups[group_name])
                    print(f'Epoch {epoch}: Descongelado {group_name}')

# Ejemplo de schedule
schedule = {
    0: ['head'],      # Epoch 0: solo head
    2: ['layer4'],    # Epoch 2: + layer4
    4: ['layer3'],    # Epoch 4: + layer3
    6: ['layer2'],    # Epoch 6: + layer2
}

unfreezer = GradualUnfreezer(model, layer_groups, schedule)

# Simular epochs
for epoch in range(8):
    unfreezer.step(epoch)
    print(f'  Trainable: {count_trainable(model):,}')
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## 🔧 Paso 6: Warmup + Cosine Annealing

Combinamos warmup inicial con cosine annealing:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def create_scheduler_with_warmup(optimizer, warmup_epochs, total_epochs):
    """
    Crea scheduler con warmup + cosine annealing.
    
    Args:
        optimizer: Optimizer
        warmup_epochs: Epochs de warmup
        total_epochs: Total de epochs
    
    Returns:
        Scheduler combinado
    """
    # Warmup lineal
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    # Cosine annealing
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-6
    )
    
    # Combinar
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )
    
    return scheduler

# Crear scheduler
scheduler = create_scheduler_with_warmup(optimizer, warmup_epochs=3, total_epochs=20)

# Visualizar LR schedule
print('\nLR Schedule:')
for epoch in range(20):
    lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch:2d}: lr = {lr:.2e}')
    scheduler.step()
```

**Descomenta** la sección correspondiente en `starter/main.py`.

---

## ✅ Checklist de Verificación

- [ ] Grupos de capas identificados correctamente
- [ ] Funciones freeze/unfreeze implementadas
- [ ] Discriminative LR configurado
- [ ] Gradual unfreezing funcionando
- [ ] Scheduler con warmup implementado

---

## 📚 Recursos

- [ULMFiT Paper](https://arxiv.org/abs/1801.06146)
- [PyTorch LR Schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
