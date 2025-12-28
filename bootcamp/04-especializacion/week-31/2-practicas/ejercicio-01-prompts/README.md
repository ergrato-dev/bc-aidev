# ✍️ Ejercicio 01: Prompt Engineering

## 🎯 Objetivo

Dominar técnicas de prompt engineering: zero-shot, few-shot y chain-of-thought.

---

## 📋 Descripción

En este ejercicio diseñarás prompts efectivos para diferentes tareas, comparando resultados entre técnicas y optimizando iterativamente.

---

## 🔧 Pasos del Ejercicio

### Paso 1: Zero-Shot Prompting

Diseñar prompts sin ejemplos:

```python
prompt = """Clasifica el sentimiento del texto como POSITIVO, NEGATIVO o NEUTRAL.

Texto: "Este producto cambió mi vida, lo recomiendo totalmente"
Sentimiento:"""
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

### Paso 2: Few-Shot Prompting

Incluir ejemplos que guíen al modelo:

```python
prompt = """Clasifica el sentimiento.

Texto: "Me encanta este producto" → POSITIVO
Texto: "Terrible, no lo compres" → NEGATIVO
Texto: "Está bien, nada especial" → NEUTRAL

Texto: "Este producto cambió mi vida"
Sentimiento:"""
```

### Paso 3: Chain-of-Thought

Hacer que el modelo razone paso a paso:

```python
prompt = """Resuelve el problema pensando paso a paso.

Problema: María tiene 3 manzanas. Compra 2 bolsas con 4 manzanas cada una. ¿Cuántas tiene?

Pensemos:
1. María empieza con 3 manzanas
2. Compra 2 bolsas × 4 manzanas = 8 manzanas
3. Total: 3 + 8 = 11 manzanas

Respuesta: 11 manzanas

Problema: [nuevo problema]
Pensemos:"""
```

### Paso 4: Structured Output

Forzar formato específico en la respuesta:

```python
prompt = """Extrae información en formato JSON.

Texto: "Apple fue fundada por Steve Jobs en 1976"

{
    "empresa": "Apple",
    "fundador": "Steve Jobs",
    "año": 1976
}"""
```

### Paso 5: Role Prompting

Asignar una personalidad o expertise:

```python
prompt = """Eres un profesor experto en física con 20 años de experiencia.
Explicas conceptos de manera clara usando analogías cotidianas.

Estudiante: ¿Qué es la energía cinética?
Profesor:"""
```

### Paso 6: Comparación y Optimización

Comparar resultados entre técnicas y medir calidad.

---

## 📁 Estructura

```
ejercicio-01-prompts/
├── README.md
└── starter/
    └── main.py
```

---

## ▶️ Ejecución

```bash
cd bootcamp/04-especializacion/week-31/2-practicas/ejercicio-01-prompts
python starter/main.py
```

---

## ✅ Criterios de Éxito

- [ ] Zero-shot funciona para clasificación básica
- [ ] Few-shot mejora consistencia del formato
- [ ] CoT mejora razonamiento en problemas matemáticos
- [ ] Structured output genera JSON válido
- [ ] Role prompting produce respuestas especializadas

---

## 🔗 Recursos

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)
