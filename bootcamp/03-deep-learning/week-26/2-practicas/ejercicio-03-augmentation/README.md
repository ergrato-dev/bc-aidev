# 🖼️ Ejercicio 03: Data Augmentation

## 🎯 Objetivo

Implementar pipelines de Data Augmentation y medir su impacto en la generalización del modelo.

---

## 📋 Descripción

En este ejercicio aprenderás:

1. Crear pipelines con `torchvision.transforms`
2. Aplicar transformaciones geométricas y de color
3. Visualizar augmentations
4. Medir mejora en test accuracy

---

## 🔧 Requisitos

```bash
pip install torch torchvision matplotlib
```

---

## 📝 Pasos del Ejercicio

### Paso 1: Transforms Básicos (Sin Augmentation)

Pipeline mínimo solo con normalización.

```python
basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Transforms con Augmentation

Pipeline completo con transformaciones aleatorias.

```python
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Visualizar Augmentations

Ver el efecto de las transformaciones en una imagen.

**Descomenta** la sección del Paso 3.

---

### Paso 4: Crear Datasets y Modelo

```python
# Entrenamiento CON augmentation
train_aug = datasets.CIFAR10(transform=augment_transform)

# Test SIEMPRE sin augmentation
test_set = datasets.CIFAR10(transform=basic_transform)
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Entrenar y Comparar

Entrenamos dos modelos: uno con augmentation y otro sin.

**Descomenta** la sección del Paso 5.

---

### Paso 6: Analizar Resultados

Comparamos overfitting y test accuracy.

**Descomenta** la sección del Paso 6.

---

## ✅ Criterios de Éxito

| Métrica | Sin Augmentation | Con Augmentation |
|---------|------------------|------------------|
| Test Accuracy | ~65% | ~72% |
| Gap Train-Test | > 15% | < 10% |

---

## 📚 Recursos

- [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)
- [Data Augmentation Best Practices](https://pytorch.org/vision/stable/transforms.html)
