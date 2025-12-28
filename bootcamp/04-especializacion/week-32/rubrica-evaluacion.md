# 📋 Rúbrica de Evaluación - Semana 32

## RAG - Retrieval Augmented Generation

---

## 📊 Distribución de Puntuación

| Componente | Peso | Descripción |
|------------|------|-------------|
| 🧠 Conocimiento | 30% | Comprensión teórica de RAG |
| 💪 Desempeño | 40% | Ejercicios prácticos completados |
| 📦 Producto | 30% | Proyecto Asistente de Documentos |

---

## 🧠 Conocimiento (30 puntos)

### Conceptos Evaluados

| Concepto | Puntos | Criterio |
|----------|--------|----------|
| Arquitectura RAG | 8 | Explica el flujo completo: retrieval → augmentation → generation |
| Embeddings | 7 | Comprende representación vectorial y similitud semántica |
| Vector Databases | 8 | Conoce operaciones: index, query, filter, update |
| Optimización | 7 | Entiende chunking, reranking, hybrid search |

### Niveles de Logro

| Nivel | Puntos | Descripción |
|-------|--------|-------------|
| Excelente | 27-30 | Explica RAG con profundidad y casos de uso |
| Bueno | 21-26 | Comprende todos los componentes |
| Suficiente | 15-20 | Entiende el flujo básico |
| Insuficiente | <15 | Confusión sobre conceptos clave |

---

## 💪 Desempeño (40 puntos)

### Ejercicio 1: Embeddings (12 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Generar embeddings | 4 | Usa sentence-transformers correctamente |
| Similitud coseno | 4 | Calcula similitud entre vectores |
| Búsqueda semántica | 4 | Encuentra documentos por query |

### Ejercicio 2: ChromaDB (14 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Crear colección | 3 | Configura ChromaDB correctamente |
| Insertar documentos | 4 | Añade docs con embeddings y metadata |
| Query semántico | 4 | Recupera documentos relevantes |
| Filtros metadata | 3 | Usa where clauses efectivamente |

### Ejercicio 3: Q&A Documents (14 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Cargar documentos | 3 | Procesa diferentes formatos |
| Chunking | 4 | Divide texto apropiadamente |
| Pipeline RAG | 4 | Integra retrieval + LLM |
| Respuestas coherentes | 3 | Genera respuestas basadas en contexto |

### Niveles de Logro

| Nivel | Puntos | Descripción |
|-------|--------|-------------|
| Excelente | 36-40 | Todos los ejercicios completos y optimizados |
| Bueno | 28-35 | Funcionalidad correcta en todos |
| Suficiente | 20-27 | Ejercicios básicos funcionando |
| Insuficiente | <20 | Ejercicios incompletos o erróneos |

---

## 📦 Producto (30 puntos)

### Proyecto: Asistente de Documentos

#### Funcionalidad (15 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Carga de documentos | 4 | Soporta PDF, TXT, MD |
| Indexación | 4 | Embeddings + ChromaDB funcional |
| Q&A interactivo | 4 | Chat que responde preguntas |
| Citación de fuentes | 3 | Indica de dónde viene la info |

#### Calidad Técnica (10 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Código limpio | 3 | Organizado, legible, DRY |
| Documentación | 3 | Docstrings, README claro |
| Manejo de errores | 2 | Try/except apropiado |
| Configurabilidad | 2 | Parámetros ajustables |

#### Bonus (5 puntos extra)

| Feature | Puntos | Descripción |
|---------|--------|-------------|
| Múltiples colecciones | +1 | Separar docs por tema |
| Reranking | +2 | Implementar segundo paso de ranking |
| Persistencia | +1 | Guardar/cargar índice |
| UI (Gradio/Streamlit) | +1 | Interfaz gráfica |

### Niveles de Logro

| Nivel | Puntos | Descripción |
|-------|--------|-------------|
| Excelente | 27-30 | Sistema robusto con features adicionales |
| Bueno | 21-26 | Todas las funcionalidades core |
| Suficiente | 15-20 | Q&A básico funcionando |
| Insuficiente | <15 | Sistema no funcional |

---

## ✅ Checklist de Verificación

### Ejercicios

- [ ] Ejercicio 1: Embeddings calculados y comparados
- [ ] Ejercicio 2: ChromaDB operaciones CRUD funcionando
- [ ] Ejercicio 3: Pipeline Q&A genera respuestas

### Proyecto

- [ ] Carga al menos 3 documentos diferentes
- [ ] Responde preguntas sobre el contenido
- [ ] Indica la fuente de la información
- [ ] Código documentado con docstrings
- [ ] README con instrucciones de uso

### Conceptos

- [ ] Puedo explicar qué problema resuelve RAG
- [ ] Entiendo cómo funcionan los embeddings semánticos
- [ ] Sé las diferencias entre vector DBs
- [ ] Comprendo estrategias de chunking

---

## 📝 Criterios de Aprobación

| Requisito | Mínimo |
|-----------|--------|
| Conocimiento | 15/30 (50%) |
| Desempeño | 20/40 (50%) |
| Producto | 15/30 (50%) |
| **Total** | **50/100 (50%)** |

**Nota**: Se requiere mínimo 50% en cada componente para aprobar.

---

## 🎯 Retroalimentación

### Fortalezas Comunes
- Comprensión del flujo RAG
- Uso efectivo de ChromaDB
- Integración con LLMs

### Áreas de Mejora Frecuentes
- Chunking muy grande o muy pequeño
- No filtrar resultados irrelevantes
- Prompts sin contexto estructurado
- Falta de manejo de errores

---

_Rúbrica Semana 32 - Bootcamp IA: Zero to Hero_
