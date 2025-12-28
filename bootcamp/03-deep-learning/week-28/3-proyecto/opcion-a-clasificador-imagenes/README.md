# 🖼️ Proyecto: Clasificador de Imágenes con Transfer Learning

## 🎯 Objetivo

Construir un clasificador de imágenes de alta precisión usando transfer learning con modelos preentrenados de ImageNet.

---

## 📋 Descripción

Desarrollarás un pipeline completo de Computer Vision que incluye:
- Carga y preprocesamiento de datos
- Data augmentation
- Transfer learning con ResNet/EfficientNet
- Entrenamiento con técnicas de optimización avanzadas
- Evaluación y visualización de resultados

---

## ⏱️ Duración

4 horas

---

## 📊 Dataset

**CIFAR-10** - 60,000 imágenes de 10 clases:
- Airplane, Automobile, Bird, Cat, Deer
- Dog, Frog, Horse, Ship, Truck

El dataset se descarga automáticamente (~170MB).

---

## 🗂️ Estructura

```
opcion-a-clasificador-imagenes/
├── README.md          # Este archivo
├── starter/
│   └── main.py        # Código con TODOs
└── solution/
    └── main.py        # Solución completa
```

---

## 📝 Tareas a Implementar

### 1. Preprocesamiento de Datos (20 min)
- [ ] Implementar transformaciones de entrenamiento
- [ ] Implementar transformaciones de validación
- [ ] Crear DataLoaders con split train/val/test

### 2. Modelo (30 min)
- [ ] Cargar modelo preentrenado (ResNet18/50)
- [ ] Adaptar para CIFAR-10 (imágenes 32x32)
- [ ] Modificar capa clasificadora

### 3. Entrenamiento (60 min)
- [ ] Configurar optimizer (AdamW)
- [ ] Implementar learning rate scheduler
- [ ] Añadir early stopping
- [ ] Guardar checkpoints

### 4. Evaluación (30 min)
- [ ] Calcular métricas en test set
- [ ] Generar matriz de confusión
- [ ] Visualizar predicciones

### 5. Documentación (40 min)
- [ ] Documentar código
- [ ] Crear gráficas de entrenamiento
- [ ] Escribir conclusiones

---

## 🎯 Criterios de Éxito

| Métrica | Mínimo | Objetivo | Excelente |
|---------|--------|----------|-----------|
| Test Accuracy | 80% | 85% | 90%+ |
| Código documentado | ✓ | ✓ | ✓ |
| Matriz confusión | ✓ | ✓ | ✓ |
| Gráficas training | - | ✓ | ✓ |

---

## 🚀 Instrucciones

### 1. Preparar Entorno

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install torch torchvision matplotlib scikit-learn tqdm
```

### 2. Implementar el Proyecto

```bash
# Abrir starter y completar los TODOs
cd starter
python main.py
```

### 3. Verificar Resultados

- Accuracy en test > 85%
- Modelo guardado en `best_model.pth`
- Gráficas generadas

---

## 💡 Tips

### Para Mejor Accuracy

1. **Data Augmentation Agresivo**
   ```python
   transforms.RandomCrop(32, padding=4)
   transforms.RandomHorizontalFlip()
   transforms.ColorJitter(0.2, 0.2, 0.2)
   ```

2. **Learning Rate Schedule**
   ```python
   # OneCycleLR funciona muy bien
   scheduler = torch.optim.lr_scheduler.OneCycleLR(
       optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=len(train_loader)
   )
   ```

3. **Fine-tuning Progresivo**
   - Epoch 1-5: Solo clasificador
   - Epoch 6-15: Últimas capas del backbone
   - Epoch 16-20: Todo el modelo con lr bajo

### Errores Comunes

- ❌ No adaptar el modelo para imágenes 32x32
- ❌ Olvidar normalizar con media/std de ImageNet
- ❌ Learning rate muy alto para fine-tuning

---

## 📚 Recursos

- [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

_Proyecto Opción A - Computer Vision | Semana 28_
