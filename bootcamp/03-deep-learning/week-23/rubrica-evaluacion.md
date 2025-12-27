# 📋 Rúbrica de Evaluación - Semana 23

## CNNs II: ResNet, Transfer Learning, Fine-tuning

---

## 📊 Distribución de Puntuación

| Tipo de Evidencia | Porcentaje | Descripción |
|-------------------|------------|-------------|
| 🧠 Conocimiento   | 30%        | Comprensión teórica de conceptos |
| 💪 Desempeño      | 35%        | Ejercicios prácticos completados |
| 📦 Producto       | 35%        | Proyecto semanal funcional |

---

## 🧠 Conocimiento (30%)

### Conceptos Evaluados

| Concepto | Puntos | Criterio |
|----------|--------|----------|
| Problema de profundidad | 5 | Explica vanishing gradient y degradación |
| Conexiones residuales | 8 | Comprende skip connections y por qué funcionan |
| Transfer Learning | 8 | Entiende reutilización de features y cuándo aplicar |
| Fine-tuning | 5 | Conoce estrategias de congelación |
| Arquitecturas modernas | 4 | Diferencia ResNet, EfficientNet, etc. |
| **Total** | **30** | |

### Niveles de Desempeño

| Nivel | Rango | Descripción |
|-------|-------|-------------|
| Excelente | 27-30 | Dominio completo, puede explicar a otros |
| Bueno | 21-26 | Comprende bien, pequeñas confusiones |
| Suficiente | 15-20 | Entiende lo básico |
| Insuficiente | < 15 | No comprende conceptos clave |

---

## 💪 Desempeño (35%)

### Ejercicios Prácticos

| Ejercicio | Puntos | Criterios de Evaluación |
|-----------|--------|-------------------------|
| **01: Bloques Residuales** | 12 | |
| - BasicBlock implementado | 4 | Forward pass correcto |
| - Bottleneck implementado | 4 | Dimensiones correctas |
| - ResNet ensamblada | 4 | Modelo funcional |
| **02: Transfer Learning** | 12 | |
| - Carga modelo preentrenado | 3 | Sin errores |
| - Modifica clasificador | 4 | Adapta a N clases |
| - Feature extraction | 5 | Entrena solo clasificador |
| **03: Fine-tuning** | 11 | |
| - Congela capas selectivamente | 4 | Params correctos |
| - LR diferencial | 4 | Diferentes LR por grupo |
| - Entrena y evalúa | 3 | Resultados coherentes |
| **Total** | **35** | |

### Criterios de Calidad

- ✅ Código ejecuta sin errores
- ✅ Dimensiones de tensores correctas
- ✅ Parámetros congelados/entrenables correctos
- ✅ Comentarios explicativos

---

## 📦 Producto (35%)

### Proyecto: Clasificador de Flores

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| **Arquitectura** | 8 | |
| - Modelo base apropiado | 4 | ResNet18/34 o similar |
| - Clasificador adaptado | 4 | Correcto para 102 clases |
| **Entrenamiento** | 12 | |
| - Data augmentation | 3 | Transforms apropiados |
| - Scheduler LR | 3 | Reduce LR apropiadamente |
| - Early stopping | 3 | Evita overfitting |
| - Accuracy ≥85% | 3 | Métrica objetivo |
| **Análisis** | 8 | |
| - Comparativa scratch vs TL | 4 | Documenta diferencias |
| - Curvas de entrenamiento | 2 | Loss y accuracy |
| - Matriz de confusión | 2 | Visualización |
| **Código y Documentación** | 7 | |
| - Código limpio y organizado | 3 | Sigue convenciones |
| - Comentarios útiles | 2 | Explica decisiones |
| - Reproducibilidad | 2 | Seeds fijadas |
| **Total** | **35** | |

### Niveles de Logro - Proyecto

| Nivel | Accuracy | Descripción |
|-------|----------|-------------|
| Excelente | ≥ 90% | Supera expectativas |
| Bueno | 85-89% | Cumple objetivo |
| Suficiente | 75-84% | Cercano al objetivo |
| Insuficiente | < 75% | No alcanza mínimo |

---

## 📈 Bonificaciones

| Bonus | Puntos Extra | Criterio |
|-------|--------------|----------|
| Accuracy ≥ 92% | +3 | Excelente optimización |
| Comparativa múltiples modelos | +2 | ResNet vs EfficientNet vs otros |
| Grad-CAM visualización | +3 | Muestra qué aprende la red |
| Documentación excepcional | +2 | README detallado con análisis |

**Máximo bonus**: +5 puntos (no excede 100%)

---

## ⚠️ Penalizaciones

| Penalización | Puntos | Motivo |
|--------------|--------|--------|
| Código no ejecuta | -10 | Errores críticos |
| Sin data augmentation | -5 | Obligatorio para TL |
| Overfitting severo | -5 | Train >> Test accuracy |
| Plagio | -100% | Código copiado sin atribución |
| Entrega tardía | -10/día | Máximo 3 días |

---

## 📝 Checklist de Entrega

### Ejercicios
- [ ] `ejercicio-01`: BasicBlock y Bottleneck funcionando
- [ ] `ejercicio-02`: Transfer learning con modelo preentrenado
- [ ] `ejercicio-03`: Fine-tuning con LR diferencial

### Proyecto
- [ ] `main.py` ejecutable
- [ ] Modelo guardado (`.pth`)
- [ ] Visualizaciones generadas
- [ ] Accuracy ≥ 85% documentado

### Documentación
- [ ] Código comentado
- [ ] Preguntas de análisis respondidas

---

## 🎯 Criterio de Aprobación

| Requisito | Mínimo |
|-----------|--------|
| Conocimiento | ≥ 15/30 (50%) |
| Desempeño | ≥ 21/35 (60%) |
| Producto | ≥ 21/35 (60%) |
| **Total** | ≥ 70/100 |

**Nota**: Debe alcanzarse el mínimo en CADA categoría para aprobar.

---

## 🔗 Navegación

[← Volver a la Semana](README.md)
