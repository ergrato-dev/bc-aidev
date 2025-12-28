# 🎯 Proyecto: Clasificador Regularizado

## 📋 Descripción

Construir un clasificador de imágenes que combine **todas las técnicas de regularización** aprendidas para maximizar la generalización y minimizar overfitting.

---

## 🎯 Objetivos

1. Combinar Dropout, BatchNorm y Data Augmentation
2. Reducir gap train-test en más del 50%
3. Alcanzar >85% de test accuracy en CIFAR-10
4. Implementar Early Stopping

---

## 📊 Dataset

**CIFAR-10**: 60,000 imágenes 32x32 RGB en 10 clases.

---

## 🏗️ Arquitectura Requerida

```
Input (3, 32, 32)
    ↓
Conv2d(3→32) → BatchNorm2d → ReLU → MaxPool
    ↓
Conv2d(32→64) → BatchNorm2d → ReLU → MaxPool
    ↓
Conv2d(64→128) → BatchNorm2d → ReLU → MaxPool
    ↓
Flatten → Dropout(0.5)
    ↓
Linear(128*4*4→256) → BatchNorm1d → ReLU → Dropout(0.3)
    ↓
Linear(256→10)
```

---

## ✅ Criterios de Éxito

| Métrica | Objetivo |
|---------|----------|
| Test Accuracy | > 85% |
| Gap Train-Test | < 5% |
| Reducción de Gap | > 50% vs baseline |

---

## 📁 Estructura

```
clasificador-regularizado/
├── README.md
├── starter/
│   └── main.py      # TODO: Implementar
└── solution/
    └── main.py      # Solución completa
```

---

## 🚀 Ejecución

```bash
cd starter
python main.py
```

---

## 📚 Recursos

- [PyTorch CIFAR-10 Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Regularization Techniques](https://pytorch.org/docs/stable/nn.html)
