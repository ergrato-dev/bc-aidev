# 🚀 Proyecto: Entrenador Optimizado

## 🎯 Objetivo

Construir un pipeline de entrenamiento completo que integre todas las técnicas de optimización aprendidas: optimizadores modernos, learning rate schedules, inicialización de pesos, callbacks y checkpoints.

**Meta**: Alcanzar **>80% accuracy** en test con un entrenamiento eficiente y estable.

---

## 📋 Descripción

Crearás una clase `OptimizedTrainer` que encapsule las mejores prácticas de entrenamiento en Deep Learning:

- ✅ Selección de optimizador (Adam/AdamW)
- ✅ Learning rate scheduling (OneCycleLR o CosineAnnealing)
- ✅ Inicialización de pesos (He/Xavier)
- ✅ Early Stopping para evitar overfitting
- ✅ Model Checkpoint para guardar el mejor modelo
- ✅ Gradient Clipping para estabilidad
- ✅ Logging de métricas y visualización

---

## 🏗️ Estructura del Proyecto

```
entrenador-optimizado/
├── README.md           # Este archivo
├── starter/
│   └── main.py         # Tu implementación (con TODOs)
└── solution/
    └── main.py         # Solución de referencia
```

---

## 📝 Requisitos de Implementación

### 1. Clase `OptimizedTrainer`

Debe incluir:

```python
class OptimizedTrainer:
    def __init__(self, model, config):
        """
        Args:
            model: Red neuronal a entrenar
            config: Diccionario con configuración
        """
        # TODO: Inicializar optimizador, scheduler, callbacks
    
    def init_weights(self):
        """Inicializa pesos con He/Xavier."""
        # TODO: Implementar
    
    def train_epoch(self, train_loader):
        """Entrena una época."""
        # TODO: Forward, backward, gradient clipping, optimizer step
    
    def validate(self, val_loader):
        """Evalúa en validación."""
        # TODO: Implementar
    
    def fit(self, train_loader, val_loader, epochs):
        """Loop principal de entrenamiento."""
        # TODO: Integrar todo con callbacks
    
    def save_checkpoint(self, path):
        """Guarda checkpoint completo."""
        # TODO: Implementar
    
    def load_checkpoint(self, path):
        """Carga checkpoint."""
        # TODO: Implementar
```

### 2. Configuración Recomendada

```python
config = {
    'lr': 0.001,
    'optimizer': 'adamw',
    'weight_decay': 0.01,
    'scheduler': 'onecycle',
    'max_lr': 0.01,
    'patience': 7,
    'grad_clip': 1.0,
}
```

### 3. Modelo CNN

Usar una CNN simple para CIFAR-10:

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

---

## ✅ Criterios de Evaluación

| Criterio | Puntos |
|----------|--------|
| `OptimizedTrainer` implementado correctamente | 25 |
| Inicialización de pesos He/Xavier | 10 |
| Optimizador AdamW configurado | 10 |
| LR Scheduler funcionando | 15 |
| Early Stopping implementado | 15 |
| Checkpoint save/load funcional | 10 |
| Accuracy en test > 80% | 10 |
| Código limpio y documentado | 5 |
| **Total** | **100** |

---

## 🚀 Pasos para Completar

1. **Abre** `starter/main.py`
2. **Implementa** cada método marcado con `TODO`
3. **Ejecuta** el entrenamiento
4. **Verifica** que alcanzas >80% accuracy
5. **Compara** con `solution/main.py`

---

## 💡 Tips

- Usa `AdamW` con `weight_decay=0.01`
- `OneCycleLR` con `max_lr=0.01` funciona bien
- Inicializa convoluciones con He (Kaiming)
- `patience=7` es un buen balance para Early Stopping
- Gradient clipping con `max_norm=1.0`

---

## 📊 Resultado Esperado

```
Epoch 1: train_loss=1.8234, val_loss=1.5432, val_acc=0.4523
Epoch 2: train_loss=1.4123, val_loss=1.2345, val_acc=0.5634
...
Epoch 25: train_loss=0.3456, val_loss=0.5678, val_acc=0.8234

¡Entrenamiento completado!
Best validation accuracy: 82.34%
Test accuracy: 81.56%
```

---

## 📚 Recursos

- [PyTorch Training Loop Best Practices](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
