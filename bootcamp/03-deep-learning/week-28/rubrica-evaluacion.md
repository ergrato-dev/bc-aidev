# 📋 Rúbrica de Evaluación - Semana 28

## Proyecto Final de Deep Learning

### 📊 Distribución de Puntuación

| Componente | Porcentaje | Puntos |
|------------|------------|--------|
| Funcionalidad del Modelo | 35% | 35 |
| Calidad del Código | 20% | 20 |
| Documentación | 20% | 20 |
| Métricas Alcanzadas | 15% | 15 |
| Presentación/README | 10% | 10 |
| **Total** | **100%** | **100** |

---

## 1. Funcionalidad del Modelo (35 puntos)

### Excelente (32-35 puntos)
- Modelo funciona correctamente end-to-end
- Pipeline completo: carga datos → preprocesamiento → entrenamiento → evaluación → inferencia
- Manejo robusto de errores
- Código de inferencia para nuevas muestras
- Modelo guardado y cargable sin problemas

### Bueno (25-31 puntos)
- Modelo funciona correctamente
- Pipeline completo pero con algunos pasos manuales
- Manejo básico de errores
- Modelo guardado correctamente

### Suficiente (18-24 puntos)
- Modelo funciona con supervisión
- Pipeline incompleto o con bugs menores
- Errores no manejados pueden causar fallos

### Insuficiente (0-17 puntos)
- Modelo no funciona o tiene errores críticos
- Pipeline incompleto
- No se puede reproducir el entrenamiento

---

## 2. Calidad del Código (20 puntos)

### Excelente (18-20 puntos)
- Código limpio y bien organizado
- Funciones modulares y reutilizables
- Type hints en funciones principales
- Docstrings completos
- Nombres descriptivos de variables/funciones
- Sin código duplicado
- Configuración separada (no hardcoded)

### Bueno (14-17 puntos)
- Código organizado y legible
- Algunas funciones modulares
- Documentación parcial
- Nombres razonables

### Suficiente (10-13 puntos)
- Código funcional pero desordenado
- Poca modularización
- Documentación mínima

### Insuficiente (0-9 puntos)
- Código difícil de leer
- Sin organización
- Sin documentación

---

## 3. Documentación (20 puntos)

### Excelente (18-20 puntos)
- README completo con:
  - Descripción clara del problema
  - Instrucciones de instalación y ejecución
  - Descripción del dataset
  - Arquitectura del modelo (con diagrama)
  - Resultados y métricas
  - Ejemplos de uso
  - Conclusiones y trabajo futuro
- Notebook bien comentado
- Código autodocumentado

### Bueno (14-17 puntos)
- README con secciones principales
- Notebook con comentarios
- Instrucciones básicas de uso

### Suficiente (10-13 puntos)
- README básico
- Comentarios escasos
- Falta información importante

### Insuficiente (0-9 puntos)
- Sin README o muy incompleto
- Sin comentarios en código
- No se puede entender el proyecto

---

## 4. Métricas Alcanzadas (15 puntos)

### Computer Vision (Opción A)

| Accuracy Test | Puntos |
|---------------|--------|
| > 90% | 15 |
| 85-90% | 12 |
| 80-85% | 9 |
| 75-80% | 6 |
| < 75% | 3 |

### NLP (Opción B)

| F1-Score / Accuracy | Puntos |
|---------------------|--------|
| > 85% | 15 |
| 80-85% | 12 |
| 75-80% | 9 |
| 70-75% | 6 |
| < 70% | 3 |

**Nota**: Se evaluará la métrica principal según el problema elegido.

---

## 5. Presentación / README (10 puntos)

### Excelente (9-10 puntos)
- README profesional y atractivo
- Uso correcto de Markdown
- Imágenes/diagramas de arquitectura
- Gráficas de resultados (loss, accuracy)
- Badges si aplica
- Ejemplos visuales de predicciones

### Bueno (7-8 puntos)
- README bien estructurado
- Algunas visualizaciones
- Formato correcto

### Suficiente (5-6 puntos)
- README básico pero funcional
- Pocas o ninguna visualización

### Insuficiente (0-4 puntos)
- README pobre o inexistente
- Sin estructura clara

---

## 📝 Checklist de Entrega

### Archivos Requeridos

- [ ] `README.md` - Documentación del proyecto
- [ ] `requirements.txt` - Dependencias
- [ ] Notebook o script principal
- [ ] Modelo guardado (`.pth`, `.h5`, o similar)
- [ ] Carpeta `data/` o instrucciones de descarga

### Contenido del README

- [ ] Título y descripción
- [ ] Problema a resolver
- [ ] Dataset utilizado
- [ ] Arquitectura del modelo
- [ ] Instrucciones de instalación
- [ ] Instrucciones de entrenamiento
- [ ] Instrucciones de inferencia
- [ ] Resultados y métricas
- [ ] Visualizaciones
- [ ] Conclusiones

### Código

- [ ] Se ejecuta sin errores
- [ ] Reproducible (seeds fijos)
- [ ] Documentado
- [ ] Modular

---

## 🎯 Criterios de Aprobación

| Requisito | Mínimo |
|-----------|--------|
| Puntuación total | ≥ 70/100 |
| Funcionalidad del modelo | ≥ 20/35 |
| Métricas alcanzadas | ≥ 6/15 |
| Entregables completos | 100% |

---

## 📌 Notas Adicionales

### Bonificaciones (+5 puntos máximo)

- **+2**: Deployment básico (Gradio, Streamlit)
- **+2**: Análisis de errores detallado
- **+1**: Comparación de múltiples modelos

### Penalizaciones

- **-10**: Entrega tardía (por día)
- **-5**: Código no reproducible
- **-5**: Plagio o copia

---

_Rúbrica Semana 28 - Proyecto Final Deep Learning_
