# 📋 Rúbrica de Evaluación - Semana 26

## 🛡️ Regularización en Deep Learning

---

## 📊 Distribución de Puntos

| Tipo de Evidencia | Porcentaje | Puntos |
|-------------------|------------|--------|
| 🧠 Conocimiento | 30% | 30 |
| 💪 Desempeño | 35% | 35 |
| 📦 Producto | 35% | 35 |
| **Total** | **100%** | **100** |

---

## 🧠 Conocimiento (30 puntos)

### Conceptos Teóricos

| Criterio | Excelente (10) | Bueno (7) | Suficiente (5) | Insuficiente (0-4) |
|----------|----------------|-----------|----------------|-------------------|
| Overfitting | Explica causas, síntomas y cómo detectarlo con métricas | Identifica overfitting correctamente | Comprensión básica | No distingue overfitting |
| Dropout | Entiende funcionamiento, inverted dropout, cuándo aplicar | Implementa correctamente | Sabe qué es | Confunde conceptos |
| Batch Norm | Comprende normalización, parámetros γ/β, train vs eval | Aplica en arquitecturas | Conoce el concepto | No entiende su propósito |

---

## 💪 Desempeño (35 puntos)

### Ejercicio 01: Dropout (12 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Implementación | 4 | nn.Dropout aplicado correctamente |
| Posición | 3 | Ubicado en capas apropiadas |
| Comparación | 3 | Métricas con/sin dropout |
| Visualización | 2 | Gráficas de training/validation |

### Ejercicio 02: Batch Normalization (12 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Implementación | 4 | nn.BatchNorm aplicado correctamente |
| Train/Eval mode | 3 | Diferencia entre model.train() y model.eval() |
| Convergencia | 3 | Demuestra convergencia más rápida |
| Análisis | 2 | Explica efecto en gradientes |

### Ejercicio 03: Data Augmentation (11 puntos)

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| Transforms | 4 | Múltiples transformaciones aplicadas |
| Pipeline | 3 | Compose configurado correctamente |
| Visualización | 2 | Muestra imágenes aumentadas |
| Impacto | 2 | Mide mejora en generalización |

---

## 📦 Producto (35 puntos)

### Proyecto: Clasificador Regularizado

| Criterio | Excelente (35) | Bueno (28) | Suficiente (21) | Insuficiente (0-20) |
|----------|----------------|------------|-----------------|---------------------|
| **Baseline** | Modelo sin regularización documentado | Baseline funcional | Baseline básico | Sin baseline |
| **Regularización** | Todas las técnicas (Dropout, BN, Aug, WD, ES) | 4 técnicas aplicadas | 3 técnicas | < 3 técnicas |
| **Comparación** | Tablas y gráficas comparativas detalladas | Comparación clara | Métricas básicas | Sin comparación |
| **Mejora** | Gap train-test reducido significativamente | Mejora visible | Alguna mejora | Sin mejora |
| **Código** | Limpio, documentado, modular | Bien organizado | Funcional | Difícil de seguir |

### Métricas Objetivo

| Métrica | Objetivo |
|---------|----------|
| Reducción de gap (train-test) | > 50% |
| Test accuracy | > 85% |
| Tiempo de convergencia | Reducido con BN |

---

## 📝 Criterios Generales

### Código

- [ ] Sigue convenciones de Python (PEP 8)
- [ ] Type hints en funciones principales
- [ ] Docstrings explicativos
- [ ] Imports organizados
- [ ] Sin código duplicado

### Documentación

- [ ] README del proyecto completo
- [ ] Explicación de decisiones técnicas
- [ ] Resultados reproducibles
- [ ] Referencias a recursos utilizados

---

## 🎯 Niveles de Desempeño

| Nivel | Puntos | Descripción |
|-------|--------|-------------|
| 🌟 Excelente | 90-100 | Domina todas las técnicas y las aplica estratégicamente |
| ✅ Competente | 70-89 | Implementa correctamente y analiza resultados |
| 📈 En desarrollo | 50-69 | Comprende conceptos pero implementación incompleta |
| ❌ Insuficiente | 0-49 | No demuestra comprensión de regularización |

---

## ✅ Checklist de Entrega

### Ejercicios
- [ ] ejercicio-01-dropout completado
- [ ] ejercicio-02-batch-norm completado
- [ ] ejercicio-03-augmentation completado

### Proyecto
- [ ] Modelo baseline entrenado
- [ ] Modelo regularizado entrenado
- [ ] Tabla comparativa de métricas
- [ ] Gráficas de loss y accuracy
- [ ] Código documentado
- [ ] README con análisis

---

## 📚 Recursos de Apoyo

- [Dropout Paper](https://jmlr.org/papers/v15/srivastava14a.html)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
- [PyTorch Data Augmentation](https://pytorch.org/vision/stable/transforms.html)

---

_Rúbrica Semana 26 | Módulo: Deep Learning | Bootcamp IA: Zero to Hero_
