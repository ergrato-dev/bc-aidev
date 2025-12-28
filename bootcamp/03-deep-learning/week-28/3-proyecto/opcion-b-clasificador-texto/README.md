# 📝 Proyecto: Clasificador de Texto con Transformers

## 🎯 Objetivo

Construir un clasificador de sentimiento usando fine-tuning de modelos preentrenados de Hugging Face (DistilBERT/BERT).

---

## 📋 Descripción

Desarrollarás un pipeline completo de NLP que incluye:
- Carga y exploración del dataset
- Tokenización con AutoTokenizer
- Fine-tuning de DistilBERT
- Entrenamiento con Hugging Face Trainer
- Evaluación e inferencia

---

## ⏱️ Duración

4 horas

---

## 📊 Dataset

**IMDB Movie Reviews** - 50,000 reviews de películas:
- 25,000 para entrenamiento
- 25,000 para test
- Clasificación binaria: Positivo / Negativo

El dataset se descarga automáticamente desde Hugging Face.

---

## 🗂️ Estructura

```
opcion-b-clasificador-texto/
├── README.md          # Este archivo
├── starter/
│   └── main.py        # Código con TODOs
└── solution/
    └── main.py        # Solución completa
```

---

## 📝 Tareas a Implementar

### 1. Carga de Datos (15 min)
- [ ] Cargar dataset IMDB desde Hugging Face
- [ ] Explorar estructura y ejemplos
- [ ] Crear subset para desarrollo rápido (opcional)

### 2. Tokenización (30 min)
- [ ] Cargar tokenizer de DistilBERT
- [ ] Implementar función de tokenización
- [ ] Aplicar a todo el dataset

### 3. Modelo (20 min)
- [ ] Cargar modelo preentrenado
- [ ] Configurar para clasificación binaria

### 4. Entrenamiento (60 min)
- [ ] Configurar TrainingArguments
- [ ] Definir función de métricas
- [ ] Entrenar con Trainer API
- [ ] Implementar early stopping

### 5. Evaluación (30 min)
- [ ] Evaluar en test set
- [ ] Analizar errores
- [ ] Probar inferencia con textos nuevos

### 6. Documentación (25 min)
- [ ] Documentar código
- [ ] Guardar modelo
- [ ] Escribir conclusiones

---

## 🎯 Criterios de Éxito

| Métrica | Mínimo | Objetivo | Excelente |
|---------|--------|----------|-----------|
| Test Accuracy | 80% | 85% | 90%+ |
| Test F1-Score | 80% | 85% | 90%+ |
| Código documentado | ✓ | ✓ | ✓ |
| Inferencia funcionando | ✓ | ✓ | ✓ |

---

## 🚀 Instrucciones

### 1. Preparar Entorno

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install transformers datasets evaluate accelerate scikit-learn
pip install torch  # Si no está instalado
```

### 2. Implementar el Proyecto

```bash
cd starter
python main.py
```

### 3. Verificar Resultados

- Accuracy en test > 85%
- Modelo guardado en `./model`
- Inferencia funcionando

---

## 💡 Tips

### Para Mejor Accuracy

1. **Hiperparámetros óptimos**
   ```python
   learning_rate = 2e-5  # Rango: 1e-5 a 5e-5
   batch_size = 16       # 8-32 según GPU
   epochs = 3            # 2-5 para fine-tuning
   warmup_ratio = 0.1    # Warmup importante
   ```

2. **No entrenar demasiado**
   - 3-5 épocas suelen ser suficientes
   - Más épocas pueden causar overfitting

3. **Usar subset para desarrollo**
   ```python
   # Para pruebas rápidas
   train_dataset = train_dataset.select(range(1000))
   ```

### Errores Comunes

- ❌ Learning rate muy alto (> 1e-4)
- ❌ Demasiadas épocas (> 5)
- ❌ No usar warmup
- ❌ Olvidar establecer `fp16=True` con GPU

---

## 📚 Recursos

- [Hugging Face Course - NLP](https://huggingface.co/course/chapter3)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [IMDB Dataset](https://huggingface.co/datasets/imdb)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)

---

_Proyecto Opción B - NLP | Semana 28_
