# 🎯 Selección del Proyecto

## 1. Criterios de Selección

Elige un proyecto que cumpla con:

### 1.1 Factibilidad

| Criterio | Pregunta | ✅ Buena señal | ❌ Mala señal |
|----------|----------|----------------|---------------|
| **Tiempo** | ¿Se puede completar en 12h? | MVP claro | Scope creep |
| **Datos** | ¿Hay datos disponibles? | Dataset público | Hay que crear datos |
| **Complejidad** | ¿Es apropiado para tu nivel? | Desafiante pero alcanzable | Muy fácil o imposible |
| **Stack** | ¿Dominas las tecnologías? | Stack del bootcamp | Tecnología nueva |

### 1.2 Valor de Aprendizaje

```
┌─────────────────────────────────────────────────────────────┐
│                 MATRIZ DE SELECCIÓN                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    Alto Impacto    │   🎯 IDEAL        │  ⚠️ CUIDADO       │
│                    │   Chatbot RAG     │   Sistema         │
│                    │   Clasificador    │   complejo        │
│                    │                   │                   │
│  ──────────────────┼───────────────────┼───────────────────│
│                    │                   │                   │
│    Bajo Impacto    │   ✅ SEGURO       │   ❌ EVITAR       │
│                    │   MNIST++         │   Tutorial        │
│                    │   Iris mejorado   │   copiado         │
│                    │                   │                   │
│                    │   Baja            │   Alta            │
│                    │   Complejidad     │   Complejidad     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Opciones Detalladas

### 🗣️ Opción 1: Chatbot RAG

**Descripción**: Sistema de Q&A que responde preguntas basándose en documentos.

```python
# Arquitectura
"""
Usuario → Pregunta → Embeddings → Vector Search → 
→ Contexto Relevante → LLM → Respuesta
"""

# Stack
stack = {
    "framework": "LangChain",
    "vectordb": "ChromaDB",
    "embeddings": "sentence-transformers",
    "llm": "Ollama/OpenAI",
    "api": "FastAPI",
    "ui": "Gradio/Streamlit"
}
```

**Requisitos**:
- Conocimiento de RAG (semana 32)
- Documentos para indexar
- API key de LLM (opcional con Ollama)

**Alcance MVP**:
- [ ] Indexar 10-50 documentos
- [ ] Endpoint de búsqueda
- [ ] Endpoint de chat
- [ ] UI básica

**Métricas**:
- Relevancia de respuestas
- Latencia de respuesta
- Satisfacción de usuario

---

### 🖼️ Opción 2: Clasificador de Imágenes

**Descripción**: App que clasifica imágenes en categorías.

```python
# Arquitectura
"""
Imagen → Preprocessing → CNN/ViT → Predicción → Categoría
"""

# Stack
stack = {
    "modelo": "ResNet/EfficientNet/ViT",
    "framework": "TensorFlow/PyTorch",
    "transfer": "Hugging Face/timm",
    "api": "FastAPI",
    "ui": "Gradio"
}
```

**Ideas de Dataset**:
| Dataset | Categorías | Dificultad |
|---------|------------|------------|
| Flores | 5 tipos | ⭐ |
| Comida | 10 categorías | ⭐⭐ |
| Ropa | 10 categorías | ⭐⭐ |
| Mascotas | Razas | ⭐⭐⭐ |

**Alcance MVP**:
- [ ] Dataset de 1000+ imágenes
- [ ] Modelo con >85% accuracy
- [ ] API que recibe imagen y retorna clase
- [ ] UI para subir imagen

---

### 📊 Opción 3: Analizador de Sentimiento

**Descripción**: Dashboard que analiza sentimiento de textos.

```python
# Arquitectura
"""
Texto → Tokenización → Transformer → Sentimiento + Score
"""

# Stack
stack = {
    "modelo": "BERT/RoBERTa",
    "framework": "Transformers",
    "api": "FastAPI",
    "ui": "Streamlit",
    "visualización": "Plotly"
}
```

**Fuentes de Datos**:
- Reviews de productos (Amazon, Yelp)
- Tweets (Twitter API)
- Comentarios (Reddit, YouTube)

**Alcance MVP**:
- [ ] Análisis de texto individual
- [ ] Análisis de batch
- [ ] Visualización de distribución
- [ ] API REST

---

### 🎯 Opción 4: Sistema de Recomendación

**Descripción**: Sistema que recomienda items basado en preferencias.

```python
# Arquitectura
"""
Usuario + Historial → Features → Modelo → Top-N Recomendaciones
"""

# Stack
stack = {
    "algoritmo": "Collaborative/Content-based",
    "framework": "Scikit-learn/Surprise",
    "api": "FastAPI",
    "cache": "Redis (opcional)"
}
```

**Datasets Sugeridos**:
| Dataset | Items | URL |
|---------|-------|-----|
| MovieLens | Películas | grouplens.org |
| Amazon Reviews | Productos | amazon.com |
| Spotify | Canciones | spotify.com |
| Books | Libros | goodreads.com |

**Alcance MVP**:
- [ ] Dataset procesado
- [ ] Modelo de recomendación
- [ ] API: /recommend/{user_id}
- [ ] Top-10 recomendaciones

---

### 📈 Opción 5: Predictor de Series Temporales

**Descripción**: Sistema de forecasting para predicción temporal.

```python
# Arquitectura
"""
Serie Temporal → Features → Modelo → Predicción Futura
"""

# Stack
stack = {
    "modelo": "Prophet/ARIMA/LSTM",
    "framework": "statsmodels/TensorFlow",
    "api": "FastAPI",
    "ui": "Streamlit",
    "visualización": "Plotly"
}
```

**Ideas de Aplicación**:
- Predicción de ventas
- Demanda de energía
- Tráfico web
- Precios de acciones (educativo)

**Alcance MVP**:
- [ ] Dataset temporal (>1 año de datos)
- [ ] Modelo entrenado
- [ ] Predicción N días adelante
- [ ] Visualización de forecast

---

### 🎨 Opción 6: Proyecto Libre

**Requisitos para aprobación**:

1. **Propuesta escrita** (1 página):
   - Problema a resolver
   - Stack tecnológico
   - Fuente de datos
   - Alcance MVP
   - Timeline

2. **Criterios de evaluación**:
   - Viabilidad en 12 horas
   - Uso de técnicas del bootcamp
   - Valor demostrativo

---

## 3. Checklist de Selección

Antes de empezar, verifica:

### Datos
- [ ] Dataset identificado
- [ ] Acceso confirmado (descarga/API)
- [ ] Licencia verificada
- [ ] Tamaño manejable (<1GB recomendado)

### Tecnología
- [ ] Stack definido
- [ ] Librerías instalables
- [ ] Hardware suficiente (GPU si es necesario)

### Alcance
- [ ] MVP definido con 4-5 features
- [ ] Features priorizadas
- [ ] "Nice to have" identificados

### Tiempo
- [ ] Plan de 12 horas creado
- [ ] Buffer para imprevistos
- [ ] Deadline de cada fase

---

## 4. Decisión Final

Completa esta tabla para tu proyecto elegido:

| Aspecto | Tu Respuesta |
|---------|--------------|
| **Proyecto elegido** | |
| **Problema que resuelve** | |
| **Dataset** | |
| **Stack principal** | |
| **MVP (3-5 features)** | |
| **Métrica de éxito** | |
| **Riesgo principal** | |
| **Plan de mitigación** | |

---

## 💡 Recomendación Final

> Para tu primer proyecto end-to-end, recomendamos **Opción 2 (Clasificador de Imágenes)** o **Opción 3 (Analizador de Sentimiento)** por su balance entre impacto y complejidad.

Si tienes experiencia previa, **Opción 1 (Chatbot RAG)** demuestra conocimientos más avanzados.
