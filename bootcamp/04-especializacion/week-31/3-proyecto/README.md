# 🤖 Proyecto: Asistente Especializado

## 🎯 Objetivo

Crear un chatbot con personalidad usando técnicas de prompt engineering y opcionalmente fine-tuning.

---

## 📋 Descripción

Desarrollarás un asistente virtual especializado en un dominio específico, con personalidad definida, sistema de memoria conversacional, y manejo de contexto.

---

## 🎨 Requisitos del Proyecto

### Funcionalidades Obligatorias

1. **Personalidad Definida**
   - System prompt que defina el rol
   - Tono consistente (formal, casual, técnico)
   - Área de expertise específica

2. **Memoria Conversacional**
   - Mantener historial de conversación
   - Referencia a mensajes anteriores
   - Contexto limitado por ventana

3. **Manejo de Contexto**
   - Prompt template estructurado
   - Inyección de información relevante
   - Formato de respuesta consistente

4. **Guardrails**
   - Límites del área de conocimiento
   - Respuestas para preguntas fuera de alcance
   - Manejo de inputs inválidos

### Funcionalidades Opcionales (Bonus)

- Fine-tuning con LoRA para personalidad
- Integración con base de conocimiento
- Múltiples personalidades seleccionables
- Exportación de conversaciones

---

## 🏗️ Arquitectura Sugerida

```
┌─────────────────────────────────────────────┐
│              USER INPUT                      │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│         INPUT PREPROCESSOR                   │
│  - Validación                                │
│  - Sanitización                              │
│  - Detección de intención                    │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│         PROMPT BUILDER                       │
│  - System prompt                             │
│  - Context injection                         │
│  - History management                        │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│              LLM                             │
│  (GPT-2 / LLaMA / etc)                       │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│         RESPONSE PROCESSOR                   │
│  - Formatting                                │
│  - Guardrails check                          │
│  - Memory update                             │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│              RESPONSE                        │
└─────────────────────────────────────────────┘
```

---

## 📁 Estructura del Proyecto

```
3-proyecto/
├── README.md
├── starter/
│   └── main.py          # Esqueleto con TODOs
└── solution/
    └── main.py          # Implementación completa
```

---

## 📝 Especificaciones Técnicas

### System Prompt Template

```python
SYSTEM_PROMPT = """You are {name}, a {role}.

Personality:
- {trait_1}
- {trait_2}
- {trait_3}

Expertise: {expertise}

Rules:
1. Always stay in character
2. If asked about topics outside your expertise, politely redirect
3. Be helpful but concise
4. Use {tone} language

Current context: {context}
"""
```

### Conversación Template

```python
CONVERSATION_TEMPLATE = """
{system_prompt}

### Conversation History:
{history}

### User: {user_input}

### {assistant_name}:
"""
```

---

## 🎯 Ejemplos de Asistentes

### Opción A: Tutor de Python
- **Nombre**: PyMentor
- **Rol**: Tutor de programación Python
- **Tono**: Amigable y paciente
- **Expertise**: Python, programación básica

### Opción B: Chef Virtual
- **Nombre**: ChefAI
- **Rol**: Chef profesional
- **Tono**: Apasionado y detallista
- **Expertise**: Cocina, recetas, técnicas culinarias

### Opción C: Asistente Fitness
- **Nombre**: FitCoach
- **Rol**: Entrenador personal
- **Tono**: Motivador y energético
- **Expertise**: Ejercicio, nutrición básica

### Opción D: Personalizado
- Crea tu propia personalidad

---

## ✅ Criterios de Evaluación

| Criterio | Puntos |
|----------|--------|
| Personalidad consistente | 25 |
| Memoria conversacional funcional | 25 |
| Manejo de contexto | 20 |
| Guardrails implementados | 15 |
| Código limpio y documentado | 10 |
| Funcionalidades bonus | +5 |
| **Total** | **100** |

---

## 📋 Entregables

1. **Código funcional** (`main.py`)
2. **Documentación** del asistente elegido
3. **Ejemplos de conversación** que demuestren las funcionalidades

---

## ▶️ Ejecución

```bash
cd bootcamp/04-especializacion/week-31/3-proyecto
python starter/main.py   # Para desarrollar
python solution/main.py  # Referencia completa
```

---

## 🔗 Recursos

- [LangChain Chat Memory](https://python.langchain.com/docs/modules/memory/)
- [OpenAI Chat Best Practices](https://platform.openai.com/docs/guides/chat)
- [Character AI Guidelines](https://character.ai/help)

---

## 💡 Tips

- Empieza con un system prompt simple y mejóralo iterativamente
- Prueba diferentes temperaturas para encontrar el balance
- Limita el historial a los últimos N mensajes para evitar exceder contexto
- Documenta las decisiones de diseño de tu asistente
