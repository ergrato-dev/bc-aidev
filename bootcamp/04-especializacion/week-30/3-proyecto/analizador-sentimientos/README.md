# 📦 Proyecto: Analizador de Sentimientos Multilingüe

## 🎯 Objetivo

Construir un analizador de sentimientos completo usando Hugging Face Transformers que soporte múltiples idiomas y proporcione análisis detallado.

---

## 📋 Descripción

Desarrollarás un sistema de análisis de sentimientos que:
- Detecta el idioma del texto automáticamente
- Usa el modelo apropiado para cada idioma
- Proporciona scores de confianza
- Analiza textos individuales y en batch
- Genera reportes de análisis

---

## 🔧 Requisitos Técnicos

### Dependencias

```bash
pip install transformers torch langdetect
```

### Funcionalidades a Implementar

1. **Detección de idioma** - Identificar el idioma del texto
2. **Análisis de sentimiento** - Clasificar como positivo/negativo/neutral
3. **Modelo multilingüe** - Soporte para múltiples idiomas
4. **Batch processing** - Analizar múltiples textos eficientemente
5. **Reporte de resultados** - Generar estadísticas y resumen

---

## 📁 Estructura

```
analizador-sentimientos/
├── README.md
├── starter/
│   └── main.py      # Plantilla con TODOs
└── solution/
    └── main.py      # Solución completa
```

---

## 🚀 Implementación

### Paso 1: Configuración Inicial

Cargar el modelo multilingüe de sentimientos:

```python
from transformers import pipeline
sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
```

### Paso 2: Clase SentimentAnalyzer

Crear una clase que encapsule toda la funcionalidad:

```python
class SentimentAnalyzer:
    def __init__(self):
        # Cargar modelo
        pass
    
    def analyze(self, text: str) -> dict:
        # Analizar texto individual
        pass
    
    def analyze_batch(self, texts: list) -> list:
        # Analizar múltiples textos
        pass
    
    def generate_report(self, results: list) -> str:
        # Generar reporte estadístico
        pass
```

### Paso 3: Detección de Idioma

Usar langdetect para identificar el idioma:

```python
from langdetect import detect
lang = detect("Bonjour le monde")  # 'fr'
```

### Paso 4: Mapeo de Estrellas a Sentimiento

El modelo retorna 1-5 estrellas, mapear a categorías:
- 1-2 estrellas → NEGATIVE
- 3 estrellas → NEUTRAL  
- 4-5 estrellas → POSITIVE

---

## ▶️ Ejecución

```bash
cd bootcamp/04-especializacion/week-30/3-proyecto/analizador-sentimientos

# Starter (completar TODOs)
python starter/main.py

# Solución
python solution/main.py
```

---

## 📊 Output Esperado

```
=== Analizador de Sentimientos Multilingüe ===

Texto: "I love this product, it's amazing!"
  Idioma: en
  Sentimiento: POSITIVE (⭐⭐⭐⭐⭐)
  Confianza: 87.32%

Texto: "Este producto es terrible"
  Idioma: es
  Sentimiento: NEGATIVE (⭐)
  Confianza: 72.15%

=== Reporte de Análisis ===
Total textos: 10
Positivos: 6 (60%)
Neutrales: 2 (20%)
Negativos: 2 (20%)
Confianza promedio: 79.5%
```

---

## ✅ Criterios de Evaluación

### Conocimiento (30%)
- [ ] Entiende modelos multilingües
- [ ] Comprende scores y probabilidades
- [ ] Conoce limitaciones del modelo

### Desempeño (40%)
- [ ] Código funciona sin errores
- [ ] Maneja diferentes idiomas
- [ ] Procesa batches eficientemente

### Producto (30%)
- [ ] Implementa todas las funcionalidades
- [ ] Código documentado
- [ ] Genera reportes informativos

---

## 🎯 Retos Adicionales

1. **Añadir más idiomas**: Probar con chino, japonés, árabe
2. **Análisis por aspecto**: Identificar qué aspectos son positivos/negativos
3. **Visualización**: Crear gráficos con matplotlib
4. **API REST**: Exponer como servicio web con FastAPI

---

## 🔗 Recursos

- [Modelo Multilingüe](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
- [langdetect](https://pypi.org/project/langdetect/)
- [Pipelines Guide](https://huggingface.co/docs/transformers/main_classes/pipelines)
