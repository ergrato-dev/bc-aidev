# 🧠 Ejercicio 03: Modelos Pre-entrenados

## 🎯 Objetivo

Cargar y usar modelos pre-entrenados para inferencia manual sin pipelines.

---

## 📋 Descripción

En este ejercicio aprenderás a cargar modelos con AutoModel, ejecutar inferencia manualmente, y entender los outputs de diferentes tipos de modelos.

---

## 🔧 Pasos del Ejercicio

### Paso 1: Cargar Modelo y Tokenizer

Usar Auto classes para cargar modelos:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModelForSequenceClassification.from_pretrained("model-name")
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

### Paso 2: Preparar Input

Tokenizar texto para el modelo:

```python
inputs = tokenizer("texto", return_tensors="pt")
```

### Paso 3: Ejecutar Inferencia

Pasar inputs al modelo:

```python
with torch.no_grad():
    outputs = model(**inputs)
```

### Paso 4: Procesar Outputs

Convertir logits a predicciones:

```python
probs = torch.softmax(outputs.logits, dim=-1)
pred = torch.argmax(probs, dim=-1)
```

### Paso 5: Diferentes Cabezas

Usar modelos con diferentes cabezas:
- `AutoModelForSequenceClassification`
- `AutoModelForTokenClassification`
- `AutoModelForQuestionAnswering`
- `AutoModelForCausalLM`

### Paso 6: Modelos en Español

Cargar modelos entrenados en español:

```python
model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
```

---

## 📁 Estructura

```
ejercicio-03-modelos/
├── README.md
└── starter/
    └── main.py
```

---

## ▶️ Ejecución

```bash
cd bootcamp/04-especializacion/week-30/2-practicas/ejercicio-03-modelos
python starter/main.py
```

---

## ✅ Criterios de Éxito

- [ ] Puedo cargar modelos con AutoModel
- [ ] Ejecuto inferencia manualmente
- [ ] Proceso logits a probabilidades
- [ ] Uso diferentes cabezas de clasificación
- [ ] Entiendo la estructura de outputs

---

## 🔗 Recursos

- [AutoModel Documentation](https://huggingface.co/docs/transformers/model_doc/auto)
- [Model Outputs](https://huggingface.co/docs/transformers/main_classes/output)
