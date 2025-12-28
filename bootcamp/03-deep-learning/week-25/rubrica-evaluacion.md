# 📋 Rúbrica de Evaluación - Semana 25

## 🎯 Transformers: Attention is All You Need

---

## 📊 Distribución de Puntuación

| Componente | Peso | Descripción |
|------------|------|-------------|
| 🧠 Conocimiento | 30% | Comprensión teórica de attention y transformers |
| 💪 Desempeño | 35% | Ejercicios prácticos completados |
| 📦 Producto | 35% | Proyecto clasificador de texto |

---

## 🧠 Conocimiento (30%)

### Conceptos Evaluados

| Concepto | Puntos | Criterio |
|----------|--------|----------|
| Mecanismo de Atención | 8 | Explica Query, Key, Value |
| Self-Attention | 8 | Comprende cálculo de attention scores |
| Multi-Head Attention | 7 | Entiende propósito de múltiples heads |
| Positional Encoding | 7 | Explica necesidad y funcionamiento |

### Niveles de Logro

| Nivel | Rango | Descripción |
|-------|-------|-------------|
| Excelente | 90-100% | Explica transformer completo con fórmulas |
| Bueno | 75-89% | Comprende attention y arquitectura |
| Suficiente | 60-74% | Entiende conceptos básicos |
| Insuficiente | <60% | No comprende el mecanismo de atención |

---

## 💪 Desempeño (35%)

### Ejercicios Prácticos

| Ejercicio | Puntos | Criterios |
|-----------|--------|-----------|
| Attention Básico | 12 | Implementa scaled dot-product attention |
| Multi-Head Attention | 12 | Múltiples heads con concatenación |
| Transformer Encoder | 11 | Encoder layer completo funcionando |

### Criterios por Ejercicio

#### Ejercicio 1: Attention Básico
- [ ] Calcula attention scores correctamente (4 pts)
- [ ] Aplica softmax (4 pts)
- [ ] Multiplica por Values (4 pts)

#### Ejercicio 2: Multi-Head Attention
- [ ] Proyecta Q, K, V por cada head (4 pts)
- [ ] Ejecuta attention en paralelo (4 pts)
- [ ] Concatena y proyecta salida (4 pts)

#### Ejercicio 3: Transformer Encoder
- [ ] Self-attention + Add & Norm (4 pts)
- [ ] Feed-forward network (4 pts)
- [ ] Residual connections (3 pts)

---

## 📦 Producto (35%)

### Proyecto: Clasificador de Texto con Transformer

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Funcionalidad | 12 | Modelo entrena y predice |
| Accuracy | 10 | > 85% en test set |
| Arquitectura | 8 | Transformer encoder bien estructurado |
| Código | 5 | Limpio y documentado |

### Niveles de Logro del Proyecto

| Nivel | Accuracy | Puntos |
|-------|----------|--------|
| Excelente | > 90% | 35/35 |
| Bueno | 85-90% | 30/35 |
| Suficiente | 75-85% | 25/35 |
| Insuficiente | < 75% | 15/35 |

---

## 📈 Escala de Calificación

| Calificación | Rango | Significado |
|--------------|-------|-------------|
| A | 90-100% | Sobresaliente |
| B | 80-89% | Notable |
| C | 70-79% | Aprobado |
| D | 60-69% | Suficiente |
| F | <60% | No aprobado |

---

## ✅ Checklist de Autoevaluación

### Conocimiento
- [ ] Puedo explicar Q, K, V y su propósito
- [ ] Entiendo por qué se escala por √d_k
- [ ] Sé qué es multi-head attention y por qué es útil
- [ ] Comprendo el rol del positional encoding

### Desempeño
- [ ] Implementé attention desde cero
- [ ] Creé multi-head attention funcional
- [ ] Construí un transformer encoder layer

### Producto
- [ ] Mi clasificador alcanza accuracy > 85%
- [ ] El código está documentado
- [ ] Puedo explicar cada componente

---

## 🎯 Fórmulas Clave

**Scaled Dot-Product Attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Multi-Head Attention:**
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

---

_Rúbrica Semana 25 | Módulo Deep Learning_
