# 🔄 Ejercicio 01: Pipeline End-to-End

## 🎯 Objetivo

Construir un pipeline completo de Machine Learning que integre todas las fases: datos, preprocesamiento, modelo, entrenamiento y evaluación.

---

## 📋 Descripción

En este ejercicio crearás un pipeline modular y reutilizable que servirá como base para el proyecto final. Aprenderás a estructurar código de ML de forma profesional.

---

## ⏱️ Duración

90 minutos

---

## 📚 Requisitos Previos

- PyTorch o TensorFlow instalado
- Conocimiento de DataLoaders
- Familiaridad con entrenamiento de redes neuronales

---

## 🗂️ Estructura

```
ejercicio-01-pipeline/
├── README.md          # Este archivo
└── starter/
    └── main.py        # Código para completar
```

---

## 📝 Instrucciones

### Paso 1: Configuración del Proyecto

Entender la estructura de un pipeline profesional:

```python
# Configuración centralizada
config = {
    'data': {
        'batch_size': 32,
        'num_workers': 2,
        'val_split': 0.2
    },
    'model': {
        'num_classes': 10,
        'pretrained': True
    },
    'training': {
        'epochs': 10,
        'lr': 1e-3,
        'weight_decay': 0.01
    }
}
```

**Abre `starter/main.py`** y revisa la estructura del pipeline.

---

### Paso 2: Módulo de Datos

Crear funciones para cargar y preprocesar datos:

```python
def get_transforms(train: bool = True):
    """Retorna transformaciones según el modo."""
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
```

Descomenta la sección correspondiente en `starter/main.py`.

---

### Paso 3: Creación del Modelo

Encapsular la creación del modelo en una función:

```python
def create_model(config: dict) -> nn.Module:
    """Crea y configura el modelo."""
    model = models.resnet18(pretrained=config['pretrained'])
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    return model
```

Descomenta la sección correspondiente.

---

### Paso 4: Loop de Entrenamiento

Implementar el loop de entrenamiento con métricas:

```python
def train_epoch(model, loader, criterion, optimizer, device):
    """Entrena una época."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), 100. * correct / total
```

Descomenta la sección correspondiente.

---

### Paso 5: Evaluación

Implementar función de evaluación:

```python
def evaluate(model, loader, criterion, device):
    """Evalúa el modelo."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(loader), 100. * correct / total
```

Descomenta la sección correspondiente.

---

### Paso 6: Pipeline Completo

Integrar todo en una función principal:

```python
def run_pipeline(config: dict):
    """Ejecuta el pipeline completo."""
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Data
    train_loader, val_loader = get_dataloaders(config['data'])
    
    # 3. Model
    model = create_model(config['model']).to(device)
    
    # 4. Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 5. Train loop
    best_acc = 0
    for epoch in range(config['training']['epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model, best_acc
```

Descomenta y ejecuta el pipeline completo.

---

## ✅ Checklist de Verificación

- [ ] Configuración centralizada funcionando
- [ ] DataLoaders creados correctamente
- [ ] Modelo creado y movido a device
- [ ] Loop de entrenamiento ejecutándose
- [ ] Métricas mostrándose por época
- [ ] Modelo guardado al mejorar

---

## 🎯 Resultado Esperado

```
Epoch 1: Train Loss=1.8234, Acc=45.23% | Val Loss=1.5432, Acc=52.10%
Epoch 2: Train Loss=1.2345, Acc=62.45% | Val Loss=1.1234, Acc=65.80%
Epoch 3: Train Loss=0.9876, Acc=71.20% | Val Loss=0.9543, Acc=72.50%
...
Best model saved with accuracy: 78.50%
```

---

## 💡 Tips

- Usa `tqdm` para barras de progreso
- Implementa logging para guardar métricas
- Considera añadir learning rate scheduling
- Guarda también el optimizer state para resumir

---

## 🚀 Extensiones Opcionales

1. Añadir Early Stopping
2. Implementar learning rate finder
3. Agregar data augmentation configurable
4. Crear función de inferencia para nuevas imágenes

---

_Ejercicio 01 - Pipeline End-to-End | Semana 28_
