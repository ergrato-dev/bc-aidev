# 🔮 Semana 13: Clustering - Aprendizaje No Supervisado

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender el paradigma del aprendizaje no supervisado
- ✅ Implementar K-Means y entender el algoritmo del centroide
- ✅ Aplicar DBSCAN para detectar clusters de forma arbitraria
- ✅ Construir dendrogramas con clustering jerárquico
- ✅ Evaluar la calidad de clusters con métricas apropiadas
- ✅ Seleccionar el algoritmo adecuado según el problema

---

## 📚 Requisitos Previos

- ✅ Semana 12: SVM, KNN y Naive Bayes
- ✅ Dominio de NumPy y Pandas
- ✅ Comprensión de distancias (Euclidiana, Manhattan)
- ✅ Conocimientos de visualización con Matplotlib

---

## 🗂️ Estructura de la Semana

```
week-13/
├── README.md                    # Este archivo
├── rubrica-evaluacion.md        # Criterios de evaluación
├── 0-assets/                    # Diagramas y recursos visuales
│   ├── 01-kmeans-algoritmo.svg
│   ├── 02-dbscan-conceptos.svg
│   ├── 03-clustering-jerarquico.svg
│   ├── 04-metricas-evaluacion.svg
│   └── 05-comparacion-algoritmos.svg
├── 1-teoria/                    # Material teórico
│   ├── 01-introduccion-clustering.md
│   ├── 02-kmeans.md
│   ├── 03-dbscan.md
│   └── 04-clustering-jerarquico.md
├── 2-practicas/                 # Ejercicios guiados
│   ├── ejercicio-01-kmeans/
│   ├── ejercicio-02-dbscan/
│   ├── ejercicio-03-jerarquico/
│   └── ejercicio-04-evaluacion/
├── 3-proyecto/                  # Proyecto semanal
│   └── segmentacion-clientes/
├── 4-recursos/                  # Material adicional
│   ├── ebooks-free/
│   ├── videografia/
│   └── webgrafia/
└── 5-glosario/                  # Términos clave
    └── README.md
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| #   | Tema                       | Archivo                                                                 | Duración |
| --- | -------------------------- | ----------------------------------------------------------------------- | -------- |
| 1   | Introducción al Clustering | [01-introduccion-clustering.md](1-teoria/01-introduccion-clustering.md) | 20 min   |
| 2   | K-Means                    | [02-kmeans.md](1-teoria/02-kmeans.md)                                   | 25 min   |
| 3   | DBSCAN                     | [03-dbscan.md](1-teoria/03-dbscan.md)                                   | 25 min   |
| 4   | Clustering Jerárquico      | [04-clustering-jerarquico.md](1-teoria/04-clustering-jerarquico.md)     | 20 min   |

### 💻 Prácticas (2.5 horas)

| #   | Ejercicio              | Carpeta                                                          | Duración |
| --- | ---------------------- | ---------------------------------------------------------------- | -------- |
| 1   | K-Means desde Cero     | [ejercicio-01-kmeans/](2-practicas/ejercicio-01-kmeans/)         | 40 min   |
| 2   | DBSCAN y Detección     | [ejercicio-02-dbscan/](2-practicas/ejercicio-02-dbscan/)         | 40 min   |
| 3   | Clustering Jerárquico  | [ejercicio-03-jerarquico/](2-practicas/ejercicio-03-jerarquico/) | 35 min   |
| 4   | Evaluación de Clusters | [ejercicio-04-evaluacion/](2-practicas/ejercicio-04-evaluacion/) | 35 min   |

### 📦 Proyecto (2 horas)

| Proyecto                 | Descripción                                      | Carpeta                                                     |
| ------------------------ | ------------------------------------------------ | ----------------------------------------------------------- |
| Segmentación de Clientes | Sistema de segmentación con múltiples algoritmos | [segmentacion-clientes/](3-proyecto/segmentacion-clientes/) |

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

---

## 🔑 Conceptos Clave

### Aprendizaje No Supervisado

- **Sin etiquetas**: No hay variable objetivo
- **Descubrimiento**: Encontrar patrones ocultos
- **Exploración**: Entender estructura de datos

### Algoritmos de Clustering

| Algoritmo  | Tipo         | Forma Clusters | Outliers | Escalabilidad |
| ---------- | ------------ | -------------- | -------- | ------------- |
| K-Means    | Partición    | Esféricos      | Sensible | Alta          |
| DBSCAN     | Densidad     | Arbitrarios    | Detecta  | Media         |
| Jerárquico | Aglomerativo | Flexibles      | Sensible | Baja          |

### Métricas de Evaluación

- **Silhouette Score**: Cohesión vs separación
- **Inercia (WCSS)**: Varianza intra-cluster
- **Davies-Bouldin**: Ratio de dispersión
- **Método del Codo**: Selección de K óptimo

---

## 📌 Entregables

Al finalizar la semana debes entregar:

1. **Ejercicios completados** (2-practicas/)

   - [ ] ejercicio-01: K-Means implementado
   - [ ] ejercicio-02: DBSCAN con detección de anomalías
   - [ ] ejercicio-03: Dendrograma y clustering jerárquico
   - [ ] ejercicio-04: Evaluación y comparación de métricas

2. **Proyecto semanal** (3-proyecto/)

   - [ ] Sistema de segmentación de clientes
   - [ ] Comparación de algoritmos
   - [ ] Visualizaciones de clusters
   - [ ] Informe de análisis

3. **Autoevaluación**
   - [ ] Completar checklist de verificación
   - [ ] Responder cuestionario de conocimientos

---

## 🔗 Navegación

| ⬅️ Anterior                       | 🏠 Módulo                        | Siguiente ➡️                      |
| --------------------------------- | -------------------------------- | --------------------------------- |
| [Semana 12](../week-12/README.md) | [Machine Learning](../README.md) | [Semana 14](../week-14/README.md) |

---

## 💡 Tips para esta Semana

> 🎯 **Consejo**: El clustering es tanto arte como ciencia. No hay una "respuesta correcta" - la validación requiere conocimiento del dominio además de métricas.

- **Normaliza siempre**: K-Means es sensible a la escala
- **Visualiza primero**: Entiende tus datos antes de clusterizar
- **Prueba varios K**: El método del codo no siempre es claro
- **DBSCAN para anomalías**: Los puntos de ruido son información valiosa

---

_Semana 13 de 36 | Módulo: Machine Learning | Bootcamp IA: Zero to Hero_
