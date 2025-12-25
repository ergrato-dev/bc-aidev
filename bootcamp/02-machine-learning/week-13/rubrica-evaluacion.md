# 📋 Rúbrica de Evaluación - Semana 13

## 🎯 Clustering: K-Means, DBSCAN y Jerárquico

---

## 📊 Distribución de Puntos

| Componente      | Peso     | Puntos  |
| --------------- | -------- | ------- |
| 🧠 Conocimiento | 30%      | 30      |
| 💪 Desempeño    | 40%      | 40      |
| 📦 Producto     | 30%      | 30      |
| **Total**       | **100%** | **100** |

---

## 🧠 Conocimiento (30 puntos)

### Conceptos Teóricos (15 puntos)

| Criterio                       | Excelente (5)                                                          | Bueno (4)                                      | Suficiente (3)                                             | Insuficiente (0-2)                         |
| ------------------------------ | ---------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------ |
| **Aprendizaje no supervisado** | Explica diferencias con supervisado, tipos de problemas y aplicaciones | Comprende el paradigma y sus usos principales  | Identifica que no hay etiquetas pero confunde aplicaciones | No distingue supervisado de no supervisado |
| **Algoritmos de clustering**   | Domina K-Means, DBSCAN y Jerárquico con sus supuestos y limitaciones   | Comprende los tres algoritmos y cuándo usarlos | Conoce la mecánica pero no criterios de selección          | Confunde o no diferencia los algoritmos    |
| **Métricas de evaluación**     | Aplica Silhouette, Inercia, Davies-Bouldin correctamente               | Usa métricas apropiadas y las interpreta       | Calcula métricas pero interpretación limitada              | No sabe evaluar calidad de clusters        |

### Fundamentos Matemáticos (15 puntos)

| Criterio              | Excelente (5)                                                 | Bueno (4)                                         | Suficiente (3)                                         | Insuficiente (0-2)                   |
| --------------------- | ------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------ | ------------------------------------ |
| **Distancias**        | Calcula y aplica Euclidiana, Manhattan, Coseno según contexto | Maneja múltiples distancias correctamente         | Usa distancia Euclidiana pero no otras                 | No comprende concepto de distancia   |
| **Algoritmo K-Means** | Explica convergencia, inicialización, y variantes (K-Means++) | Describe pasos del algoritmo y criterio de parada | Entiende asignación/actualización pero no convergencia | No puede explicar el algoritmo       |
| **DBSCAN y densidad** | Define epsilon, minPts, core/border/noise points formalmente  | Comprende conceptos de densidad y conectividad    | Aplica DBSCAN pero no entiende parámetros              | No comprende clustering por densidad |

---

## 💪 Desempeño (40 puntos)

### Ejercicio 1: K-Means (10 puntos)

| Criterio           | Excelente (10)                                                       | Bueno (7-9)                                      | Suficiente (5-6)                                     | Insuficiente (0-4)                         |
| ------------------ | -------------------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------ |
| **Implementación** | K-Means desde cero funcionando, método del codo, visualización 2D/3D | Implementación correcta con visualización básica | Usa sklearn correctamente pero sin análisis profundo | Errores en implementación o uso incorrecto |

### Ejercicio 2: DBSCAN (10 puntos)

| Criterio           | Excelente (10)                                                         | Bueno (7-9)                                           | Suficiente (5-6)                           | Insuficiente (0-4)                    |
| ------------------ | ---------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------ | ------------------------------------- |
| **Implementación** | DBSCAN con selección óptima de eps/minPts, detecta outliers, visualiza | Aplica DBSCAN correctamente con parámetros razonables | Ejecuta DBSCAN pero no optimiza parámetros | No logra aplicar DBSCAN correctamente |

### Ejercicio 3: Clustering Jerárquico (10 puntos)

| Criterio           | Excelente (10)                                         | Bueno (7-9)                             | Suficiente (5-6)                              | Insuficiente (0-4)                       |
| ------------------ | ------------------------------------------------------ | --------------------------------------- | --------------------------------------------- | ---------------------------------------- |
| **Implementación** | Dendrograma completo, múltiples linkages, corte óptimo | Dendrograma correcto con interpretación | Genera dendrograma pero no lo interpreta bien | No logra crear o interpretar dendrograma |

### Ejercicio 4: Evaluación (10 puntos)

| Criterio     | Excelente (10)                                                  | Bueno (7-9)                                      | Suficiente (5-6)                              | Insuficiente (0-4)                       |
| ------------ | --------------------------------------------------------------- | ------------------------------------------------ | --------------------------------------------- | ---------------------------------------- |
| **Métricas** | Compara algoritmos con múltiples métricas, análisis estadístico | Aplica Silhouette y otras métricas correctamente | Calcula métricas pero comparación superficial | No logra evaluar clusters apropiadamente |

---

## 📦 Producto (30 puntos)

### Proyecto: Segmentación de Clientes

| Criterio             | Excelente (10)                                                              | Bueno (7-9)                                  | Suficiente (5-6)                                     | Insuficiente (0-4)                         |
| -------------------- | --------------------------------------------------------------------------- | -------------------------------------------- | ---------------------------------------------------- | ------------------------------------------ |
| **Preprocesamiento** | Pipeline completo: limpieza, normalización, selección de features, PCA      | Normalización y limpieza correctas           | Preprocesamiento básico                              | Datos sin preparar o errores graves        |
| **Modelado**         | Compara 3+ algoritmos, optimiza hiperparámetros, justifica selección        | Aplica múltiples algoritmos con comparación  | Un algoritmo bien implementado                       | Implementación incorrecta o incompleta     |
| **Análisis**         | Interpreta segmentos con insights de negocio, visualizaciones profesionales | Describe clusters con características claras | Identifica clusters pero sin interpretación profunda | No logra describir o interpretar segmentos |

### Calidad del Código (Incluido en cada sección)

| Aspecto              | Esperado                                        |
| -------------------- | ----------------------------------------------- |
| **Estilo**           | PEP 8, type hints, docstrings                   |
| **Modularidad**      | Funciones reutilizables, código DRY             |
| **Documentación**    | Comentarios explicativos, markdown en notebooks |
| **Reproducibilidad** | Seeds fijos, requirements.txt                   |

---

## 📈 Niveles de Desempeño Global

| Nivel               | Puntos | Descripción                                                      |
| ------------------- | ------ | ---------------------------------------------------------------- |
| 🏆 **Excelente**    | 90-100 | Dominio completo de clustering con implementaciones sofisticadas |
| 🥈 **Bueno**        | 80-89  | Comprensión sólida y aplicación correcta de algoritmos           |
| 🥉 **Suficiente**   | 70-79  | Conocimientos básicos funcionales                                |
| ❌ **Insuficiente** | <70    | Requiere refuerzo en conceptos fundamentales                     |

---

## ✅ Checklist de Entrega

### Ejercicios

- [ ] Ejercicio 1: K-Means implementado y documentado
- [ ] Ejercicio 2: DBSCAN con análisis de parámetros
- [ ] Ejercicio 3: Dendrograma y clustering jerárquico
- [ ] Ejercicio 4: Comparación con métricas múltiples

### Proyecto

- [ ] Notebook/script principal ejecutable
- [ ] Preprocesamiento documentado
- [ ] Mínimo 2 algoritmos comparados
- [ ] Visualizaciones de clusters
- [ ] Interpretación de segmentos
- [ ] Conclusiones y recomendaciones

### Documentación

- [ ] README con instrucciones de ejecución
- [ ] Código comentado y limpio
- [ ] requirements.txt actualizado

---

## 🎯 Criterios de Aprobación

Para aprobar esta semana necesitas:

1. **Mínimo 70% en cada componente**

   - Conocimiento: ≥ 21 puntos
   - Desempeño: ≥ 28 puntos
   - Producto: ≥ 21 puntos

2. **Todos los ejercicios completados**

3. **Proyecto funcional con**:
   - Al menos 2 algoritmos de clustering
   - Métricas de evaluación calculadas
   - Visualizaciones de clusters

---

## 📚 Recursos de Apoyo

Si no alcanzas el nivel esperado:

1. Revisa la teoría de distancias y similitud
2. Practica con datasets sintéticos (make_blobs, make_moons)
3. Visualiza paso a paso el algoritmo K-Means
4. Experimenta con parámetros de DBSCAN
5. Consulta recursos adicionales en 4-recursos/

---

_Rúbrica v1.0 | Semana 13 | Bootcamp IA: Zero to Hero_
