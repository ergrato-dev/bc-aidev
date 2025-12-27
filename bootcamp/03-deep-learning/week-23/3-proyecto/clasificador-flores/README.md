# 🌸 Proyecto: Clasificador de Flores con Transfer Learning

## 🎯 Objetivo

Construir un clasificador de flores utilizando el dataset **Flowers-102** con Transfer Learning y Fine-tuning, alcanzando una precisión **≥ 85%**.

---

## 📋 Descripción

El dataset Flowers-102 contiene 102 categorías de flores comunes en el Reino Unido. Es un dataset desafiante debido a:

- **Gran variabilidad intra-clase**: Flores de la misma especie lucen diferentes
- **Similitud inter-clase**: Diferentes especies pueden parecerse
- **Pocas imágenes por clase**: ~40-250 imágenes por categoría

Este proyecto integra todos los conceptos de la semana:
- Bloques residuales (ResNet)
- Transfer Learning
- Fine-tuning con estrategias avanzadas

---

## 📊 Dataset: Flowers-102

| Característica | Valor |
|---------------|-------|
| Clases | 102 categorías de flores |
| Train | ~1,020 imágenes |
| Val | ~1,020 imágenes |
| Test | ~6,149 imágenes |
| Tamaño imagen | Variable (reescalar a 224×224) |

### Descarga Automática

```python
from torchvision.datasets import Flowers102

train_dataset = Flowers102(root='./data', split='train', download=True)
val_dataset = Flowers102(root='./data', split='val', download=True)
test_dataset = Flowers102(root='./data', split='test', download=True)
```

---

## 🏗️ Arquitectura Requerida

1. **Backbone**: ResNet-18 o ResNet-50 preentrenado en ImageNet
2. **Clasificador**: Nueva capa fully connected para 102 clases
3. **Estrategia**: Feature Extraction → Fine-tuning gradual

---

## 📝 Requisitos del Proyecto

### Funcionalidades Obligatorias

1. **Carga de Datos** (15%)
   - [ ] Descargar Flowers-102
   - [ ] Aplicar transformaciones (resize, normalize, augmentation)
   - [ ] Crear DataLoaders con batch_size apropiado

2. **Modelo** (25%)
   - [ ] Cargar ResNet preentrenado
   - [ ] Modificar clasificador para 102 clases
   - [ ] Implementar función para congelar/descongelar capas

3. **Entrenamiento** (30%)
   - [ ] Fase 1: Feature Extraction (backbone congelado)
   - [ ] Fase 2: Fine-tuning (descongelar gradualmente)
   - [ ] Learning rate scheduling
   - [ ] Early stopping

4. **Evaluación** (20%)
   - [ ] Accuracy en test ≥ 85%
   - [ ] Matriz de confusión
   - [ ] Top-5 accuracy

5. **Extras Opcionales** (10%)
   - [ ] Test Time Augmentation (TTA)
   - [ ] Discriminative learning rates
   - [ ] Visualización de predicciones

---

## 🗂️ Estructura del Proyecto

```
clasificador-flores/
├── README.md           # Este archivo
├── starter/
│   └── main.py         # Plantilla con TODOs
└── solution/
    └── main.py         # Solución completa
```

---

## 🚀 Instrucciones

### 1. Configurar Entorno

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install torch torchvision matplotlib tqdm
```

### 2. Completar el Código

Abre `starter/main.py` y completa las secciones marcadas con `TODO`.

### 3. Entrenar el Modelo

```bash
python starter/main.py
```

### 4. Evaluar Resultados

El script debe mostrar:
- Accuracy en test
- Top-5 accuracy
- Tiempo de entrenamiento

---

## 📈 Criterios de Evaluación

| Criterio | Puntos | Requisito |
|----------|--------|-----------|
| Carga de datos correcta | 15 | DataLoaders funcionando |
| Modelo bien configurado | 25 | ResNet + nuevo clasificador |
| Entrenamiento completo | 30 | 2 fases de entrenamiento |
| Accuracy ≥ 85% | 20 | Test accuracy |
| Código documentado | 10 | Comentarios claros |

**Mínimo para aprobar**: 70 puntos + Accuracy ≥ 80%

---

## 💡 Tips

1. **Data Augmentation** es crucial para evitar overfitting
2. Empieza con **Feature Extraction** por 5-10 epochs
3. Luego **Fine-tuning** con LR bajo (1e-4 o menor)
4. Usa **early stopping** para evitar overfitting
5. **ResNet-50** generalmente da mejores resultados que ResNet-18

---

## 📚 Recursos

- [Flowers-102 Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [torchvision.datasets.Flowers102](https://pytorch.org/vision/stable/generated/torchvision.datasets.Flowers102.html)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

## ✅ Checklist Final

- [ ] Dataset descargado y cargado correctamente
- [ ] Modelo ResNet modificado para 102 clases
- [ ] Fase de Feature Extraction completada
- [ ] Fase de Fine-tuning completada
- [ ] Test accuracy ≥ 85%
- [ ] Código documentado y limpio
