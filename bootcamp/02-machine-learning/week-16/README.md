# 🎯 Semana 16: Clustering - Aprendizaje No Supervisado

## 📋 Descripción

Esta semana exploramos el **aprendizaje no supervisado**, específicamente técnicas de **clustering** para agrupar datos sin etiquetas. Aprenderás a identificar patrones y estructuras ocultas en los datos.

**Duración**: 6 horas  
**Nivel**: Intermedio  
**Prerrequisitos**: Semanas 9-15 completadas

---

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender la diferencia entre aprendizaje supervisado y no supervisado
- ✅ Implementar K-Means y entender sus limitaciones
- ✅ Aplicar DBSCAN para clusters de forma irregular
- ✅ Usar clustering jerárquico y dendrogramas
- ✅ Evaluar la calidad de clustering (Silhouette, Elbow method)
- ✅ Elegir el algoritmo adecuado según el problema

---

## 📚 Contenido

### 1️⃣ Teoría (1.5 horas)

| Archivo                                                             | Tema                                             | Duración |
| ------------------------------------------------------------------- | ------------------------------------------------ | -------- |
| [01-intro-clustering.md](1-teoria/01-intro-clustering.md)           | Introducción al aprendizaje no supervisado       | 20 min   |
| [02-kmeans.md](1-teoria/02-kmeans.md)                               | K-Means: algoritmo, inicialización, limitaciones | 25 min   |
| [03-dbscan.md](1-teoria/03-dbscan.md)                               | DBSCAN: densidad, parámetros eps y min_samples   | 20 min   |
| [04-clustering-jerarquico.md](1-teoria/04-clustering-jerarquico.md) | Clustering jerárquico y dendrogramas             | 20 min   |
| [05-evaluacion-clustering.md](1-teoria/05-evaluacion-clustering.md) | Métricas: Silhouette, Elbow, Davies-Bouldin      | 15 min   |

### 2️⃣ Prácticas (2.5 horas)

| Ejercicio                                                       | Tema                                  | Duración |
| --------------------------------------------------------------- | ------------------------------------- | -------- |
| [ejercicio-01-kmeans](2-practicas/ejercicio-01-kmeans/)         | K-Means desde cero y con scikit-learn | 40 min   |
| [ejercicio-02-dbscan](2-practicas/ejercicio-02-dbscan/)         | DBSCAN y comparación con K-Means      | 35 min   |
| [ejercicio-03-jerarquico](2-practicas/ejercicio-03-jerarquico/) | Clustering jerárquico y dendrogramas  | 35 min   |
| [ejercicio-04-evaluacion](2-practicas/ejercicio-04-evaluacion/) | Evaluación y selección de K óptimo    | 40 min   |

### 3️⃣ Proyecto (2 horas)

| Proyecto                                                   | Descripción                             |
| ---------------------------------------------------------- | --------------------------------------- |
| [segmentacion-clientes](3-proyecto/segmentacion-clientes/) | Segmentación de clientes para marketing |

---

## 🗂️ Estructura

```
week-16/
├── README.md
├── rubrica-evaluacion.md
├── 0-assets/
│   ├── 01-clustering-overview.svg
│   ├── 02-kmeans-algorithm.svg
│   ├── 03-dbscan-concept.svg
│   ├── 04-dendrograma.svg
│   └── 05-evaluacion-clustering.svg
├── 1-teoria/
│   ├── 01-intro-clustering.md
│   ├── 02-kmeans.md
│   ├── 03-dbscan.md
│   ├── 04-clustering-jerarquico.md
│   └── 05-evaluacion-clustering.md
├── 2-practicas/
│   ├── ejercicio-01-kmeans/
│   ├── ejercicio-02-dbscan/
│   ├── ejercicio-03-jerarquico/
│   └── ejercicio-04-evaluacion/
├── 3-proyecto/
│   └── segmentacion-clientes/
├── 4-recursos/
│   └── README.md
└── 5-glosario/
    └── README.md
```

---

## ⏱️ Distribución del Tiempo

| Actividad    | Tiempo  | Porcentaje |
| ------------ | ------- | ---------- |
| 📖 Teoría    | 1.5 h   | 25%        |
| 💻 Prácticas | 2.5 h   | 42%        |
| 📦 Proyecto  | 2.0 h   | 33%        |
| **Total**    | **6 h** | **100%**   |

---

## 📊 Algoritmos de la Semana

| Algoritmo      | Tipo         | Fortalezas                           | Debilidades                     |
| -------------- | ------------ | ------------------------------------ | ------------------------------- |
| **K-Means**    | Partición    | Rápido, escalable                    | Requiere K, sensible a outliers |
| **DBSCAN**     | Densidad     | Formas arbitrarias, detecta outliers | Sensible a eps/min_samples      |
| **Jerárquico** | Aglomerativo | No requiere K, dendrograma           | Costoso computacionalmente      |

---

## 🔗 Navegación

| ⬅️ Anterior                                              | 🏠 Módulo                    | ➡️ Siguiente                                             |
| -------------------------------------------------------- | ---------------------------- | -------------------------------------------------------- |
| [Semana 15: Validación y Métricas](../week-15/README.md) | [Módulo 2: ML](../README.md) | [Semana 17: Reducción Dimensional](../week-17/README.md) |

---

## 💡 Tips de la Semana

> 🎯 **El clustering es exploratorio**: No hay "respuesta correcta". Evalúa múltiples algoritmos y valores de K para encontrar la mejor estructura.

- Siempre **escala tus datos** antes de aplicar clustering
- **Visualiza** los clusters para validar que tienen sentido
- Combina **métricas cuantitativas** con **interpretación del negocio**
- K-Means para clusters esféricos, DBSCAN para formas irregulares

---

_Semana 16 de 36 | Módulo: Machine Learning | Bootcamp IA: Zero to Hero_
