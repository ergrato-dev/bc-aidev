# 📋 Rúbrica de Evaluación - Semana 33

## 🎯 Computer Vision Avanzado

### Distribución de Puntos

| Tipo de Evidencia | Porcentaje | Puntos |
|-------------------|------------|--------|
| 🧠 Conocimiento   | 30%        | 30 pts |
| 💪 Desempeño      | 40%        | 40 pts |
| 📦 Producto       | 30%        | 30 pts |
| **Total**         | **100%**   | **100 pts** |

---

## 🧠 Conocimiento (30 pts)

### Conceptos Teóricos

| Criterio | Excelente (10) | Bueno (7) | Suficiente (5) | Insuficiente (0-4) |
|----------|----------------|-----------|----------------|-------------------|
| Tareas de CV | Explica clasificación, detección y segmentación con claridad | Explica 2 de 3 tareas correctamente | Conoce las tareas pero confunde detalles | No distingue entre tareas |
| YOLO | Entiende arquitectura y funcionamiento de YOLO | Conoce uso básico de YOLO | Sabe que YOLO detecta objetos | No entiende YOLO |
| Métricas | Calcula e interpreta mAP, IoU, precisión, recall | Conoce las métricas principales | Sabe que existen métricas | No conoce métricas |

---

## 💪 Desempeño (40 pts)

### Ejercicios Prácticos

| Ejercicio | Excelente (13-14) | Bueno (10-12) | Suficiente (7-9) | Insuficiente (0-6) |
|-----------|-------------------|---------------|------------------|-------------------|
| Clasificación | Clasifica imágenes con modelo pre-entrenado, entiende output | Clasifica correctamente | Clasifica con ayuda | No logra clasificar |
| Detección YOLO | Detecta objetos, visualiza bboxes, filtra por confianza | Detecta y visualiza básico | Ejecuta detección | No completa ejercicio |
| Segmentación | Implementa segmentación semántica e instancias | Implementa un tipo | Ejecuta modelo | No logra segmentar |

---

## 📦 Producto (30 pts)

### Proyecto: Detector de Objetos

| Criterio | Excelente (10) | Bueno (7) | Suficiente (5) | Insuficiente (0-4) |
|----------|----------------|-----------|----------------|-------------------|
| Funcionalidad | Detecta en imágenes y video, múltiples clases | Detecta en imágenes correctamente | Detección básica funciona | No funciona |
| Código | Limpio, modular, documentado, type hints | Organizado y comentado | Funcional pero desordenado | Difícil de entender |
| Visualización | Bboxes con labels, confianza, colores por clase | Bboxes básicos con labels | Muestra detecciones | Sin visualización |

---

## 📊 Escala de Calificación

| Puntuación | Calificación | Descripción |
|------------|--------------|-------------|
| 90-100     | A            | Excelente dominio de CV |
| 80-89      | B            | Buen manejo de detección y segmentación |
| 70-79      | C            | Competencia básica alcanzada |
| 60-69      | D            | Necesita reforzar conceptos |
| < 60       | F            | No alcanza competencias mínimas |

---

## ✅ Checklist de Entrega

### Ejercicios
- [ ] Clasificación con ResNet/EfficientNet completada
- [ ] Detección con YOLOv8 funcionando
- [ ] Segmentación implementada
- [ ] Código ejecutable sin errores

### Proyecto
- [ ] `detector-objetos/` con estructura completa
- [ ] Procesa imágenes individuales
- [ ] Procesa video (opcional: tiempo real)
- [ ] README con instrucciones de uso
- [ ] Visualización clara de resultados

### Documentación
- [ ] Comentarios explicativos en código
- [ ] Type hints en funciones
- [ ] README del proyecto completo

---

## 🎯 Criterios de Aprobación

- **Mínimo 70%** en cada categoría
- **Todos los ejercicios** completados
- **Proyecto funcional** con detección básica
- **Código ejecutable** sin errores críticos

---

## 📝 Notas Adicionales

- Se valora el uso de GPU para entrenamiento
- Bonus por procesamiento en tiempo real
- Bonus por fine-tuning con dataset custom
- Se permite usar modelos pre-entrenados de Ultralytics

---

## 🔗 Navegación

| ⬅️ Teoría | 🏠 Semana | Prácticas ➡️ |
|-----------|-----------|--------------|
| [1-teoria](1-teoria/) | [README](README.md) | [2-practicas](2-practicas/) |
