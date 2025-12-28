# 📝 Semana 29: Fundamentos de NLP

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender los fundamentos del Procesamiento de Lenguaje Natural
- ✅ Implementar técnicas de preprocesamiento de texto
- ✅ Entender y aplicar diferentes estrategias de tokenización
- ✅ Trabajar con word embeddings (Word2Vec, GloVe)
- ✅ Crear representaciones vectoriales de texto

---

## 📚 Requisitos Previos

- Módulo 3: Deep Learning completado
- Conocimiento de redes neuronales
- Python y NumPy

---

## 🗂️ Estructura de la Semana

```
week-29/
├── README.md                    # Este archivo
├── rubrica-evaluacion.md        # Criterios de evaluación
├── 0-assets/                    # Diagramas y recursos
├── 1-teoria/                    # Material teórico
│   ├── 01-introduccion-nlp.md
│   ├── 02-preprocesamiento.md
│   ├── 03-tokenizacion.md
│   └── 04-word-embeddings.md
├── 2-practicas/                 # Ejercicios guiados
│   ├── ejercicio-01-preprocesamiento/
│   ├── ejercicio-02-tokenizacion/
│   └── ejercicio-03-embeddings/
├── 3-proyecto/                  # Proyecto semanal
│   └── buscador-semantico/
├── 4-recursos/                  # Material adicional
│   └── README.md
└── 5-glosario/                  # Términos clave
    └── README.md
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| # | Tema | Archivo | Duración |
|---|------|---------|----------|
| 1 | Introducción a NLP | [01-introduccion-nlp.md](1-teoria/01-introduccion-nlp.md) | 20 min |
| 2 | Preprocesamiento de Texto | [02-preprocesamiento.md](1-teoria/02-preprocesamiento.md) | 25 min |
| 3 | Tokenización | [03-tokenizacion.md](1-teoria/03-tokenizacion.md) | 25 min |
| 4 | Word Embeddings | [04-word-embeddings.md](1-teoria/04-word-embeddings.md) | 20 min |

### 💻 Prácticas (2.5 horas)

| # | Ejercicio | Carpeta | Duración |
|---|-----------|---------|----------|
| 1 | Preprocesamiento de Texto | [ejercicio-01-preprocesamiento/](2-practicas/ejercicio-01-preprocesamiento/) | 45 min |
| 2 | Técnicas de Tokenización | [ejercicio-02-tokenizacion/](2-practicas/ejercicio-02-tokenizacion/) | 45 min |
| 3 | Word Embeddings | [ejercicio-03-embeddings/](2-practicas/ejercicio-03-embeddings/) | 60 min |

### 📦 Proyecto (2 horas)

| Proyecto | Descripción | Carpeta |
|----------|-------------|---------|
| Buscador Semántico | Sistema de búsqueda usando similaridad de embeddings | [buscador-semantico/](3-proyecto/buscador-semantico/) |

---

## ⏱️ Distribución del Tiempo

```
Total: 6 horas

┌─────────────────────────────────────────────────────────┐
│  📖 Teoría      │████████░░░░░░░░░░░░░░░░│  1.5h (25%)  │
│  💻 Prácticas   │████████████████░░░░░░░░│  2.5h (42%)  │
│  📦 Proyecto    │████████████░░░░░░░░░░░░│  2.0h (33%)  │
└─────────────────────────────────────────────────────────┘
```

### Sugerencia de Planificación

| Día | Actividad | Tiempo |
|-----|-----------|--------|
| Día 1 | Teoría: Intro + Preprocesamiento | 45 min |
| Día 2 | Teoría: Tokenización + Embeddings | 45 min |
| Día 3 | Práctica 1 y 2 | 1.5h |
| Día 4 | Práctica 3 | 1h |
| Día 5 | Proyecto | 2h |

---

## 📌 Entregables

### Ejercicios Completados
- [ ] Ejercicio 1: Pipeline de preprocesamiento funcionando
- [ ] Ejercicio 2: Diferentes tokenizadores implementados
- [ ] Ejercicio 3: Embeddings cargados y operaciones vectoriales

### Proyecto Semanal
- [ ] Buscador semántico funcionando
- [ ] Búsqueda por similaridad coseno
- [ ] Al menos 3 consultas de ejemplo
- [ ] Código documentado

---

## 🎯 Competencias a Desarrollar

### Técnicas
- Preprocesamiento de texto (limpieza, normalización)
- Tokenización (palabra, subpalabra, carácter)
- Representaciones vectoriales de texto
- Similaridad semántica

### Conceptuales
- Entender el pipeline de NLP
- Diferencias entre representaciones sparse y dense
- Trade-offs de diferentes tokenizadores

---

## 🔗 Navegación

| ⬅️ Anterior | 🏠 Módulo | Siguiente ➡️ |
|-------------|-----------|--------------|
| [Semana 28](../../03-deep-learning/week-28/README.md) | [Especialización](../README.md) | [Semana 30](../week-30/README.md) |

---

## 💡 Tips para esta Semana

> 🎯 **Consejo**: Los embeddings son la base de todo NLP moderno. Dedica tiempo a entender intuitivamente qué representan y cómo se usan.

### Conceptos Clave
- **Tokenización**: Dividir texto en unidades procesables
- **Embeddings**: Vectores densos que capturan significado semántico
- **Similaridad**: Medir qué tan relacionados están dos textos

### Errores Comunes
- ❌ No normalizar el texto antes de tokenizar
- ❌ Ignorar el manejo de palabras fuera de vocabulario (OOV)
- ❌ No entender la diferencia entre embeddings estáticos y contextuales

---

## 📚 Recursos Rápidos

- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/)
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)

---

_Semana 29 de 36 | Módulo: Especialización | Bootcamp IA: Zero to Hero_
