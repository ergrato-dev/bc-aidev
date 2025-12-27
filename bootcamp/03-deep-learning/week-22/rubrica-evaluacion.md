# 📋 Rúbrica de Evaluación - Semana 22

## 🖼️ CNNs I: Convoluciones, Pooling y Arquitecturas

---

## 📊 Distribución de Puntuación

| Tipo de Evidencia | Peso | Descripción |
|-------------------|------|-------------|
| 🧠 Conocimiento | 30% | Comprensión teórica de CNNs |
| 💪 Desempeño | 35% | Ejercicios prácticos completados |
| 📦 Producto | 35% | Proyecto clasificador CIFAR-10 |

**Nota mínima aprobatoria: 70%**

---

## 🧠 Conocimiento (30%)

### Conceptos Evaluados

| Concepto | Puntos | Criterio |
|----------|--------|----------|
| Operación de convolución | 8 | Explicar matemáticamente la convolución |
| Kernel y feature maps | 6 | Describir rol de filtros y mapas de características |
| Tipos de pooling | 6 | Diferenciar max, average y global pooling |
| Padding y stride | 5 | Calcular dimensiones de salida |
| Arquitecturas clásicas | 5 | Comparar LeNet-5 y VGG |

### Niveles de Desempeño

| Nivel | Rango | Descripción |
|-------|-------|-------------|
| Excelente | 90-100% | Domina todos los conceptos con profundidad |
| Bueno | 80-89% | Comprende bien con errores menores |
| Suficiente | 70-79% | Entiende conceptos básicos |
| Insuficiente | <70% | Conceptos fundamentales poco claros |

---

## 💪 Desempeño (35%)

### Ejercicio 1: Convolución Manual (10 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Implementación correcta | 4 | Convolución 2D funcional |
| Manejo de bordes | 3 | Padding implementado correctamente |
| Visualización | 3 | Mostrar kernel aplicado a imagen |

### Ejercicio 2: CNN en PyTorch (12 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Capas convolucionales | 4 | Conv2d configurado correctamente |
| Pooling y flatten | 4 | MaxPool2d y transición a lineal |
| Forward pass | 4 | Flujo correcto de datos |

### Ejercicio 3: LeNet-5 MNIST (13 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Arquitectura LeNet | 4 | Estructura fiel al paper |
| Entrenamiento | 4 | Loop de entrenamiento funcional |
| Accuracy ≥ 98% | 3 | Rendimiento esperado en MNIST |
| Visualización filtros | 2 | Mostrar qué aprenden los filtros |

---

## 📦 Producto (35%)

### Proyecto: Clasificador CIFAR-10

#### Requisitos Funcionales (20 puntos)

| Requisito | Puntos | Criterio |
|-----------|--------|----------|
| Carga de datos | 3 | CIFAR-10 cargado con DataLoader |
| Arquitectura CNN | 5 | Mínimo 3 capas convolucionales |
| Entrenamiento completo | 4 | Epochs, loss y accuracy tracked |
| Accuracy ≥ 70% | 5 | Rendimiento mínimo en test |
| Guardar modelo | 3 | state_dict guardado correctamente |

#### Calidad de Código (10 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Modularidad | 3 | Código organizado en funciones/clases |
| Documentación | 3 | Docstrings y comentarios claros |
| Estilo Python | 2 | PEP 8, type hints |
| Reproducibilidad | 2 | Seeds fijados, código ejecutable |

#### Extras (5 puntos bonus)

| Extra | Puntos | Descripción |
|-------|--------|-------------|
| Accuracy ≥ 75% | +2 | Superar objetivo base |
| Visualización feature maps | +2 | Mostrar qué detecta cada capa |
| Data augmentation | +1 | Técnicas de aumentación |

---

## 📝 Rúbrica Detallada del Proyecto

### Arquitectura CNN

| Nivel | Puntos | Descripción |
|-------|--------|-------------|
| Excelente | 5 | Arquitectura bien diseñada con justificación |
| Bueno | 4 | Arquitectura funcional con buenas decisiones |
| Suficiente | 3 | Arquitectura básica que funciona |
| Insuficiente | 0-2 | Arquitectura incorrecta o no funciona |

### Entrenamiento

| Nivel | Puntos | Descripción |
|-------|--------|-------------|
| Excelente | 4 | Training loop optimizado con early stopping |
| Bueno | 3 | Training loop completo y funcional |
| Suficiente | 2 | Training básico funcionando |
| Insuficiente | 0-1 | Errores en el entrenamiento |

### Rendimiento

| Accuracy | Puntos |
|----------|--------|
| ≥ 75% | 5 + 2 bonus |
| 70-74% | 5 |
| 65-69% | 3 |
| 60-64% | 2 |
| < 60% | 0 |

---

## 📅 Entregables y Fechas

| Entregable | Peso | Fecha |
|------------|------|-------|
| Ejercicios 1-3 | 35% | Día 4 |
| Proyecto CIFAR-10 | 35% | Día 7 |
| Cuestionario teórico | 30% | Día 7 |

---

## ✅ Checklist de Autoevaluación

### Conocimiento
- [ ] Puedo explicar la operación de convolución matemáticamente
- [ ] Entiendo la diferencia entre padding 'same' y 'valid'
- [ ] Sé calcular el tamaño de salida de una capa convolucional
- [ ] Puedo describir la arquitectura de LeNet-5
- [ ] Entiendo por qué VGG usa filtros 3×3

### Desempeño
- [ ] Implementé convolución 2D manualmente
- [ ] Creé una CNN desde cero en PyTorch
- [ ] Entrené LeNet-5 en MNIST con ≥98% accuracy
- [ ] Visualicé los filtros aprendidos

### Producto
- [ ] Mi CNN clasifica CIFAR-10 con ≥70% accuracy
- [ ] El código está bien documentado
- [ ] Puedo explicar cada decisión de arquitectura
- [ ] El modelo se guarda y carga correctamente

---

## 🎯 Objetivos de Aprendizaje Verificables

| Objetivo | Evidencia | ✓ |
|----------|-----------|---|
| Comprender convolución | Ejercicio 1 + Quiz | ☐ |
| Dominar pooling | Ejercicio 2 + Quiz | ☐ |
| Conocer arquitecturas | Quiz + Proyecto | ☐ |
| Implementar CNNs | Ejercicios 2-3 | ☐ |
| Entrenar CNNs | Proyecto CIFAR-10 | ☐ |

---

_Rúbrica Semana 22 | Módulo: Deep Learning | Bootcamp IA: Zero to Hero_
