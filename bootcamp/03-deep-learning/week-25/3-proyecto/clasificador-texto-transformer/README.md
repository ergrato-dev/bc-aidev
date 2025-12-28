# 📝 Proyecto: Clasificador de Texto con Transformer

## 🎯 Objetivo

Construir un **clasificador de sentimientos** usando un Transformer Encoder desde cero. El modelo debe alcanzar **> 85% de accuracy** en el dataset de prueba.

---

## 📋 Descripción

En este proyecto integrador aplicarás todos los conceptos de la semana:

1. **Scaled Dot-Product Attention**
2. **Multi-Head Attention**
3. **Positional Encoding**
4. **Transformer Encoder**
5. **Classification Head**

### Dataset

Usaremos un dataset de reseñas de películas (sentimiento positivo/negativo).

---

## 🏗️ Arquitectura

```
Input Tokens
     │
     ▼
┌─────────────┐
│  Embedding  │
└─────────────┘
     │
     ▼
┌─────────────┐
│ Positional  │
│  Encoding   │
└─────────────┘
     │
     ▼
┌─────────────┐
│ Transformer │ × N layers
│   Encoder   │
└─────────────┘
     │
     ▼
┌─────────────┐
│   [CLS]     │  ← Tomar primer token
│   Pooling   │
└─────────────┘
     │
     ▼
┌─────────────┐
│   Linear    │
│  Classifier │
└─────────────┘
     │
     ▼
  Prediction
(Positive/Negative)
```

---

## 📝 Requisitos

### Modelo

- [ ] Transformer Encoder con al menos 2 capas
- [ ] Multi-Head Attention con 4+ heads
- [ ] Positional Encoding (sinusoidal o aprendido)
- [ ] Classification head con dropout

### Entrenamiento

- [ ] Función de pérdida: CrossEntropyLoss
- [ ] Optimizador: Adam con learning rate scheduling
- [ ] Early stopping para evitar overfitting

### Métricas

- [ ] **Accuracy > 85%** en test set
- [ ] Reportar precision, recall, F1-score
- [ ] Visualizar curvas de entrenamiento

---

## 📁 Estructura

```
clasificador-texto-transformer/
├── README.md
├── starter/
│   └── main.py          # Código inicial con TODOs
└── solution/
    └── main.py          # Solución completa
```

---

## 🚀 Instrucciones

### 1. Preparar Datos

```python
# Cargar dataset (usaremos datos sintéticos o IMDB)
train_loader, test_loader, vocab = prepare_data()
```

### 2. Implementar Modelo

Completa los TODOs en `starter/main.py`:

1. `PositionalEncoding`: Añadir información de posición
2. `TransformerClassifier`: Encoder + Classification head
3. `train_epoch`: Loop de entrenamiento
4. `evaluate`: Evaluación en test set

### 3. Entrenar

```bash
python starter/main.py
```

### 4. Evaluar

El script mostrará:
- Accuracy por época
- Métricas finales
- Gráficas de pérdida

---

## ✅ Criterios de Evaluación

| Criterio | Puntos |
|----------|--------|
| Modelo compila sin errores | 20% |
| Arquitectura correcta | 25% |
| Entrenamiento funciona | 25% |
| Accuracy > 85% | 20% |
| Código documentado | 10% |

---

## 💡 Tips

1. **Empieza simple**: 2 capas, 4 heads, d_model=128
2. **Normaliza**: Usa Layer Normalization
3. **Regulariza**: Dropout 0.1-0.3
4. **Learning rate**: Empieza con 1e-4
5. **Batch size**: 32-64 funciona bien

---

## 🔗 Recursos

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [BERT for Classification](https://arxiv.org/abs/1810.04805)
- [torchtext datasets](https://pytorch.org/text/stable/datasets.html)

---

## ⏱️ Tiempo Estimado

2 horas
