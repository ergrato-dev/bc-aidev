# 📋 Rúbrica de Evaluación - Semana 18

## 🏆 Proyecto Final de Machine Learning

### Distribución de Puntos

| Categoría                  | Porcentaje | Puntos |
| -------------------------- | ---------- | ------ |
| 🧠 Conocimiento (Teoría)   | 20%        | 20     |
| 💪 Desempeño (Proceso)     | 40%        | 40     |
| 📦 Producto (Entregables)  | 40%        | 40     |
| **Total**                  | **100%**   | **100** |

---

## 🧠 Conocimiento (20%)

### Metodología y Fundamentos

| Criterio                           | Excelente (5) | Bueno (4) | Suficiente (3) | Insuficiente (0-2) |
| ---------------------------------- | ------------- | --------- | -------------- | ------------------ |
| Comprensión de CRISP-DM            | Aplica todas las fases correctamente | Aplica la mayoría | Aplica parcialmente | No sigue metodología |
| Selección de modelos justificada   | Justificación técnica completa | Justificación parcial | Selección sin justificar | Selección aleatoria |
| Métricas apropiadas                | Métricas perfectamente elegidas | Métricas adecuadas | Métricas básicas | Métricas incorrectas |
| Interpretación de resultados       | Análisis profundo y correcto | Análisis correcto | Análisis superficial | Sin análisis |

**Puntos máximos: 20**

---

## 💪 Desempeño (40%)

### Proceso de Desarrollo

| Criterio                           | Excelente (10) | Bueno (7-9) | Suficiente (4-6) | Insuficiente (0-3) |
| ---------------------------------- | -------------- | ----------- | ---------------- | ------------------ |
| **EDA (Análisis Exploratorio)**    | Completo con insights valiosos | Completo y correcto | Básico pero funcional | Incompleto o incorrecto |
| **Feature Engineering**            | Creativo y efectivo | Múltiples features útiles | Features básicas | Sin feature engineering |
| **Validación**                     | CV estratificado, múltiples métricas | Cross-validation correcto | Train/test split | Sin validación |
| **Optimización**                   | GridSearch/RandomSearch + análisis | Optimización sistemática | Optimización básica | Sin optimización |

**Puntos máximos: 40**

---

## 📦 Producto (40%)

### Entregables Finales

| Criterio                           | Excelente (10) | Bueno (7-9) | Suficiente (4-6) | Insuficiente (0-3) |
| ---------------------------------- | -------------- | ----------- | ---------------- | ------------------ |
| **Notebook Principal**             | Profesional, bien organizado | Claro y completo | Funcional | Desorganizado |
| **Código Limpio**                  | Modular, documentado, PEP8 | Bien estructurado | Funcional | Spaghetti code |
| **Visualizaciones**                | Informativas y publicables | Claras y correctas | Básicas | Confusas o ausentes |
| **Documentación**                  | README completo, conclusiones claras | Documentación adecuada | Documentación mínima | Sin documentación |

**Puntos máximos: 40**

---

## 🎯 Criterios Específicos del Proyecto

### Score del Modelo (Bonus)

| Accuracy Titanic | Bonus |
| ---------------- | ----- |
| > 0.82           | +10   |
| > 0.80           | +5    |
| > 0.78           | +2    |
| < 0.77           | 0     |

### Penalizaciones

| Infracción                              | Penalización |
| --------------------------------------- | ------------ |
| Código no reproducible                  | -10          |
| Fuga de datos (data leakage)            | -15          |
| Sin cross-validation                    | -10          |
| Plagio de soluciones                    | -100         |

---

## 📊 Escala de Calificación Final

| Puntos    | Calificación | Descripción                    |
| --------- | ------------ | ------------------------------ |
| 90-100+   | ⭐⭐⭐⭐⭐  | Excelente - Portfolio ready    |
| 80-89     | ⭐⭐⭐⭐    | Muy Bueno - Profesional        |
| 70-79     | ⭐⭐⭐      | Bueno - Competente             |
| 60-69     | ⭐⭐        | Suficiente - Aprobado          |
| < 60      | ⭐          | Insuficiente - Revisar         |

---

## ✅ Checklist de Entrega

### Obligatorios

- [ ] Notebook ejecutable sin errores
- [ ] EDA con al menos 5 visualizaciones
- [ ] Mínimo 3 modelos comparados
- [ ] Cross-validation implementado
- [ ] Archivo submission.csv generado
- [ ] README.md del proyecto

### Opcionales (Bonus)

- [ ] Pipeline con sklearn Pipeline
- [ ] Análisis de importancia de features
- [ ] Ensemble de modelos
- [ ] Stacking/Blending
- [ ] Análisis de errores del modelo

---

## 📝 Formato de Entrega

```
week-18/
├── 3-proyecto/
│   └── titanic-competition/
│       ├── README.md              # Documentación del proyecto
│       ├── notebooks/
│       │   └── titanic-solution.ipynb
│       ├── src/
│       │   └── pipeline.py        # Código modular
│       ├── submissions/
│       │   └── submission.csv
│       └── requirements.txt
```

---

## 🔗 Navegación

| ⬅️ Regresar              | 🏠 Semana               |
| ------------------------ | ----------------------- |
| [README](README.md)      | Semana 18               |
