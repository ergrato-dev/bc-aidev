# 🔄 Ejercicio 02: Generación de Texto

## 🎯 Objetivo

Controlar la generación de texto con parámetros: temperature, top_p, top_k, y más.

---

## 📋 Descripción

En este ejercicio aprenderás a controlar el comportamiento de modelos generativos, entendiendo cómo cada parámetro afecta el output.

---

## 🔧 Pasos del Ejercicio

### Paso 1: Temperature

Controla la aleatoriedad de la generación:
- **Baja (0.1-0.3)**: Determinista, seguro
- **Media (0.7)**: Balance creatividad/coherencia
- **Alta (1.0+)**: Creativo, puede ser incoherente

```python
output = model.generate(
    input_ids,
    temperature=0.7,
    do_sample=True
)
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

### Paso 2: Top-K Sampling

Limita a los K tokens más probables:

```python
output = model.generate(
    input_ids,
    top_k=50,
    do_sample=True
)
```

### Paso 3: Top-P (Nucleus) Sampling

Selecciona tokens hasta acumular probabilidad P:

```python
output = model.generate(
    input_ids,
    top_p=0.9,
    do_sample=True
)
```

### Paso 4: Repetition Penalty

Penaliza tokens que ya aparecieron:

```python
output = model.generate(
    input_ids,
    repetition_penalty=1.2
)
```

### Paso 5: Max Length y Stopping

Controlar longitud y condiciones de parada:

```python
output = model.generate(
    input_ids,
    max_new_tokens=100,
    min_length=20,
    eos_token_id=tokenizer.eos_token_id
)
```

### Paso 6: Beam Search

Explorar múltiples secuencias:

```python
output = model.generate(
    input_ids,
    num_beams=5,
    early_stopping=True
)
```

---

## 📁 Estructura

```
ejercicio-02-generacion/
├── README.md
└── starter/
    └── main.py
```

---

## ▶️ Ejecución

```bash
cd bootcamp/04-especializacion/week-31/2-practicas/ejercicio-02-generacion
python starter/main.py
```

---

## ✅ Criterios de Éxito

- [ ] Entiendo el efecto de temperature
- [ ] Sé combinar top_k y top_p
- [ ] Puedo evitar repeticiones
- [ ] Controlo longitud de generación
- [ ] Entiendo beam search vs sampling

---

## 🔗 Recursos

- [Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [How to generate](https://huggingface.co/blog/how-to-generate)
