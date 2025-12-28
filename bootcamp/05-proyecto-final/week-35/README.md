# 🛠️ Semana 35: Desarrollo del Proyecto Final

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Definir y planificar un proyecto de IA end-to-end
- ✅ Implementar el pipeline completo: datos → modelo → API
- ✅ Aplicar buenas prácticas de desarrollo de software
- ✅ Integrar conocimientos de ML, DL, NLP y MLOps
- ✅ Preparar el proyecto para deployment

---

## 📚 Requisitos Previos

- Módulos 1-4 completados
- Dominio del stack: Python, Pandas, Scikit-learn, TensorFlow/PyTorch
- Conocimientos de NLP y Hugging Face
- FastAPI y Docker (semana 34)

---

## 🗂️ Estructura de la Semana

```
week-35/
├── README.md                    # Este archivo
├── rubrica-evaluacion.md        # Criterios de evaluación
├── 0-assets/                    # Recursos visuales
├── 1-teoria/                    # Guías y metodología
│   ├── 01-guia-proyecto.md      # Guía completa del proyecto
│   ├── 02-seleccion-proyecto.md # Criterios de selección
│   └── 03-arquitectura.md       # Patrones de arquitectura
├── 2-templates/                 # Plantillas de proyecto
│   ├── proyecto-nlp/            # Template para proyectos NLP
│   ├── proyecto-vision/         # Template para Computer Vision
│   └── proyecto-tabular/        # Template para datos tabulares
├── 3-recursos/                  # Material adicional
│   └── README.md
└── 4-glosario/                  # Términos del proyecto
    └── README.md
```

---

## 📝 Contenidos

### 📖 Guías (2 horas)

| #   | Tema                    | Archivo                                                  | Duración |
| --- | ----------------------- | -------------------------------------------------------- | -------- |
| 1   | Guía del Proyecto Final | [01-guia-proyecto.md](1-teoria/01-guia-proyecto.md)      | 45 min   |
| 2   | Selección de Proyecto   | [02-seleccion-proyecto.md](1-teoria/02-seleccion-proyecto.md) | 30 min   |
| 3   | Arquitectura y Diseño   | [03-arquitectura.md](1-teoria/03-arquitectura.md)        | 45 min   |

### 🛠️ Desarrollo (4 horas)

| Fase | Actividad | Duración |
| ---- | --------- | -------- |
| 1    | Definición y planificación | 30 min |
| 2    | Preparación de datos | 1 hora |
| 3    | Desarrollo del modelo | 1.5 horas |
| 4    | Creación de la API | 1 hora |

---

## ⏱️ Distribución del Tiempo

```
Total: 6 horas

┌─────────────────────────────────────────────────────────┐
│  📖 Guías         │████████░░░░░░░░░░░░░░│  2.0h (33%)  │
│  🛠️ Desarrollo    │████████████████░░░░░░│  4.0h (67%)  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Opciones de Proyecto

### Opción 1: Chatbot RAG 💬
Sistema de preguntas y respuestas sobre documentos usando RAG.
- **Tecnologías**: LangChain, ChromaDB, Hugging Face, FastAPI
- **Dificultad**: ⭐⭐⭐

### Opción 2: Clasificador de Imágenes 🖼️
Aplicación de clasificación de imágenes con CNN.
- **Tecnologías**: TensorFlow/PyTorch, FastAPI, Gradio
- **Dificultad**: ⭐⭐

### Opción 3: Analizador de Sentimiento 📊
Dashboard de análisis de sentimiento en redes sociales.
- **Tecnologías**: Transformers, Streamlit, Plotly
- **Dificultad**: ⭐⭐

### Opción 4: Sistema de Recomendación 🎯
Recomendador de productos/contenido personalizado.
- **Tecnologías**: Scikit-learn, FastAPI, Redis
- **Dificultad**: ⭐⭐⭐

### Opción 5: Predictor de Series Temporales 📈
Forecasting de datos temporales (ventas, demanda, etc.).
- **Tecnologías**: Prophet/ARIMA, TensorFlow, Streamlit
- **Dificultad**: ⭐⭐⭐

### Opción 6: Proyecto Libre 🎨
Tu propia idea aprobada por el instructor.
- **Tecnologías**: A definir
- **Dificultad**: Variable

---

## 📌 Entregables Semana 35

Al finalizar esta semana debes tener:

1. **Proyecto seleccionado** ✅
   - [ ] Opción elegida y justificada
   - [ ] Alcance definido

2. **Repositorio configurado** ✅
   - [ ] Estructura de carpetas creada
   - [ ] README.md inicial
   - [ ] requirements.txt

3. **Pipeline de datos** ✅
   - [ ] Datos obtenidos/preparados
   - [ ] Preprocessing implementado
   - [ ] Dataset listo para entrenamiento

4. **Modelo funcional** ✅
   - [ ] Modelo entrenado
   - [ ] Métricas evaluadas
   - [ ] Modelo guardado

5. **API básica** ✅
   - [ ] FastAPI funcionando
   - [ ] Endpoint /predict
   - [ ] Documentación OpenAPI

---

## 🎯 Criterios de Éxito

| Criterio | Mínimo | Óptimo |
|----------|--------|--------|
| Modelo funciona | ✅ | ✅ con buenas métricas |
| API responde | ✅ | ✅ con validación |
| Código limpio | Funcional | Documentado y testeado |
| Documentación | README básico | README completo |

---

## 🔗 Navegación

| ⬅️ Anterior | 🏠 Inicio | Siguiente ➡️ |
|-------------|-----------|---------------|
| [Week 34](../../04-especializacion/week-34/README.md) | [Bootcamp](../../../README.md) | [Week 36](../week-36/README.md) |

---

## 💡 Tips

> 🎯 **Consejo**: Elige un proyecto que te apasione. La motivación es clave para completar un proyecto de calidad.

- **Empieza simple**: Un MVP funcional es mejor que un proyecto ambicioso incompleto
- **Itera**: Versión básica → mejoras → extras
- **Documenta mientras desarrollas**: No dejes la documentación para el final
- **Testea frecuentemente**: Prueba cada componente antes de integrarlo

---

_Semana 35 de 36 | Módulo: Proyecto Final | Bootcamp IA: Zero to Hero_
