# Proyecto Semana 09: Predicción de Supervivencia en el Titanic

## 🎯 Objetivo

Aplicar el flujo completo de Machine Learning para predecir la supervivencia de pasajeros del Titanic, consolidando los conceptos aprendidos en la semana.

## 📋 Descripción

El hundimiento del Titanic es uno de los naufragios más famosos de la historia. En este proyecto usarás datos reales de los pasajeros para construir un modelo predictivo que determine si un pasajero sobrevivió o no.

Este es un problema clásico de **clasificación binaria**:

- **Clase 0**: No sobrevivió
- **Clase 1**: Sobrevivió

## 📊 Dataset

Usaremos una versión simplificada del dataset Titanic disponible en seaborn:

| Feature  | Descripción                  |
| -------- | ---------------------------- |
| survived | Variable target (0=No, 1=Sí) |
| pclass   | Clase del pasajero (1, 2, 3) |
| sex      | Género (male, female)        |
| age      | Edad en años                 |
| sibsp    | # hermanos/esposos a bordo   |
| parch    | # padres/hijos a bordo       |
| fare     | Tarifa pagada                |
| embarked | Puerto de embarque (C, Q, S) |
| class    | Clase en texto               |
| who      | man, woman, child            |
| alone    | Si viajaba solo              |

## 🛠️ Requisitos

### Dependencias

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

### Estructura del Proyecto

```
3-proyecto/
├── README.md
├── starter/
│   └── main.py      # Código inicial con TODOs
└── .solution/
    └── main.py      # Solución (no incluida en git)
```

## 📝 Tareas a Completar

### Parte 1: Exploración de Datos (EDA)

- [ ] Cargar el dataset Titanic
- [ ] Explorar estructura y tipos de datos
- [ ] Analizar valores nulos
- [ ] Visualizar distribución del target
- [ ] Explorar correlaciones relevantes

### Parte 2: Preparación de Datos

- [ ] Manejar valores nulos (imputación o eliminación)
- [ ] Codificar variables categóricas (sex, embarked)
- [ ] Seleccionar features relevantes
- [ ] Dividir en train/test con stratify

### Parte 3: Modelado

- [ ] Entrenar un modelo de clasificación
- [ ] Hacer predicciones en el conjunto de test

### Parte 4: Evaluación

- [ ] Calcular accuracy
- [ ] Generar matriz de confusión
- [ ] Calcular precision, recall y F1-score
- [ ] Interpretar resultados

## ✅ Criterios de Aceptación

| Criterio         | Requisito Mínimo                |
| ---------------- | ------------------------------- |
| EDA              | Dataset explorado y documentado |
| Preprocesamiento | Datos limpios y codificados     |
| Modelo           | Al menos un modelo entrenado    |
| Accuracy         | ≥ 75% en test set               |
| Métricas         | Classification report completo  |
| Código           | Limpio, comentado y funcional   |

## 🚀 Instrucciones

1. Abre `starter/main.py`
2. Completa cada función siguiendo los TODOs
3. Ejecuta el script para verificar resultados
4. Experimenta mejorando el modelo

## 💡 Hints

1. **Valores nulos en Age**: Puedes imputar con la mediana
2. **Codificación de Sex**: 0=male, 1=female (o usar LabelEncoder)
3. **Features importantes**: pclass, sex, age, fare suelen ser muy predictivos
4. **Modelo sugerido**: Empieza con KNN o LogisticRegression

## 📚 Recursos

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Seaborn Titanic Dataset](https://seaborn.pydata.org/generated/seaborn.load_dataset.html)
- [Sklearn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

## 🎯 Entregables

1. Script `main.py` completado y funcional
2. Resultados de evaluación del modelo
3. (Opcional) Visualizaciones guardadas como PNG

---

**Tiempo estimado**: 2 horas

**Dificultad**: ⭐⭐ Intermedia
