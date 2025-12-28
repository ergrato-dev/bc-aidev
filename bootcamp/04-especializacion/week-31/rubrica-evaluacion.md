# 📋 Rúbrica de Evaluación - Semana 31

## 🤖 Large Language Models (LLMs)

---

## 📊 Distribución de Puntuación

| Componente          | Porcentaje | Puntos |
| ------------------- | ---------- | ------ |
| 🧠 Conocimiento     | 30%        | 30 pts |
| 💪 Desempeño        | 40%        | 40 pts |
| 📦 Producto         | 30%        | 30 pts |
| **Total**           | **100%**   | **100 pts** |

**Nota mínima aprobatoria: 70 puntos**

---

## 🧠 Conocimiento (30 puntos)

### Conceptos Teóricos

| Criterio | Excelente (10) | Bueno (7) | Suficiente (5) | Insuficiente (0-3) |
|----------|----------------|-----------|----------------|-------------------|
| **Arquitecturas LLM** | Explica diferencias entre GPT, BERT, T5, incluyendo pre-training objectives | Describe arquitecturas principales y sus usos | Conoce diferencias básicas | No distingue arquitecturas |
| **Prompt Engineering** | Domina técnicas avanzadas: few-shot, CoT, self-consistency | Aplica few-shot y zero-shot correctamente | Entiende prompts básicos | No comprende prompting |
| **Fine-tuning** | Entiende PEFT, LoRA, QLoRA y cuándo usar cada uno | Conoce fine-tuning tradicional y LoRA | Sabe qué es fine-tuning | No entiende fine-tuning |

---

## 💪 Desempeño (40 puntos)

### Ejercicios Prácticos

#### Ejercicio 01: Prompt Engineering (12 pts)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Zero-shot prompts | 4 | Diseña prompts efectivos sin ejemplos |
| Few-shot prompts | 4 | Incluye ejemplos que mejoran resultados |
| Chain-of-Thought | 4 | Implementa razonamiento paso a paso |

#### Ejercicio 02: Generación de Texto (14 pts)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Parámetros de generación | 5 | Configura temperature, top_p, top_k correctamente |
| Control de output | 5 | Maneja max_length, repetition_penalty |
| Comparación de modelos | 4 | Evalúa diferentes modelos para la misma tarea |

#### Ejercicio 03: Fine-tuning con LoRA (14 pts)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Configuración PEFT | 5 | Configura LoRA correctamente |
| Dataset preparation | 5 | Prepara datos en formato adecuado |
| Training loop | 4 | Ejecuta entrenamiento sin errores |

---

## 📦 Producto (30 puntos)

### Proyecto: Asistente Especializado

| Criterio | Excelente (30) | Bueno (22) | Suficiente (15) | Insuficiente (0-10) |
|----------|----------------|------------|-----------------|---------------------|
| **Funcionalidad** | Asistente responde coherentemente en su dominio, mantiene personalidad, maneja edge cases | Respuestas coherentes, personalidad consistente | Funciona básicamente pero inconsistente | No funciona o respuestas incoherentes |
| **Sistema de Prompts** | System prompt bien diseñado, few-shot examples, instrucciones claras | System prompt claro con instrucciones | Prompt básico funcional | Sin sistema de prompts definido |
| **Código** | Limpio, documentado, modular, maneja errores | Organizado y funcional | Funciona pero desorganizado | Código difícil de entender |
| **Documentación** | README completo, ejemplos de uso, limitaciones documentadas | Documentación clara | Documentación mínima | Sin documentación |

---

## ✅ Checklist de Verificación

### Conocimientos Mínimos
- [ ] Diferencio GPT (decoder) de BERT (encoder)
- [ ] Entiendo qué es el pre-training y por qué es importante
- [ ] Sé cuándo usar zero-shot vs few-shot
- [ ] Comprendo el concepto de temperature en generación
- [ ] Conozco la diferencia entre fine-tuning completo y LoRA

### Habilidades Prácticas
- [ ] Puedo diseñar prompts para diferentes tareas
- [ ] Controlo parámetros de generación de texto
- [ ] Puedo configurar y entrenar con PEFT/LoRA
- [ ] Sé evaluar outputs de modelos generativos
- [ ] Manejo alucinaciones y respuestas problemáticas

### Proyecto
- [ ] Asistente responde en español/inglés según contexto
- [ ] Sistema de prompts documentado
- [ ] Maneja preguntas fuera de dominio apropiadamente
- [ ] Código organizado y comentado

---

## 📝 Ejemplos de Evaluación

### Prompt Engineering - Excelente
```python
# System prompt bien estructurado
system_prompt = """Eres un asistente experto en Python.
Tu rol es ayudar a programadores a resolver problemas.

Reglas:
1. Responde SOLO sobre Python
2. Incluye ejemplos de código cuando sea útil
3. Explica el razonamiento paso a paso
4. Si no sabes algo, dilo honestamente

Ejemplo de buena respuesta:
Usuario: ¿Cómo itero un diccionario?
Asistente: Para iterar un diccionario en Python tienes varias opciones...
[código de ejemplo]
"""
```

### Fine-tuning - Excelente
```python
# Configuración LoRA apropiada
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                    # Rank de adaptación
    lora_alpha=32,           # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Módulos a adaptar
    lora_dropout=0.05,       # Regularización
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## 🎯 Niveles de Logro

| Nivel | Puntuación | Descripción |
|-------|------------|-------------|
| 🌟 Sobresaliente | 90-100 | Domina LLMs, prompts avanzados, fine-tuning eficiente |
| ✅ Aprobado | 70-89 | Comprende y aplica conceptos correctamente |
| ⚠️ En desarrollo | 50-69 | Necesita reforzar algunos conceptos |
| ❌ No aprobado | 0-49 | Requiere repetir el contenido |

---

## 📚 Recursos de Apoyo

Si tienes dificultades:

1. **Prompt Engineering**: Revisa ejemplos en [Prompt Engineering Guide](https://www.promptingguide.ai/)
2. **Fine-tuning**: Sigue el tutorial de [Hugging Face PEFT](https://huggingface.co/docs/peft)
3. **Generación**: Experimenta en [Hugging Face Spaces](https://huggingface.co/spaces)

---

## 🔄 Proceso de Entrega

1. Completa todos los ejercicios en `2-practicas/`
2. Desarrolla el proyecto en `3-proyecto/`
3. Verifica el checklist de esta rúbrica
4. Sube tu código al repositorio
5. Completa la autoevaluación

---

_Rúbrica Semana 31 | Módulo: Especialización | Bootcamp IA: Zero to Hero_
