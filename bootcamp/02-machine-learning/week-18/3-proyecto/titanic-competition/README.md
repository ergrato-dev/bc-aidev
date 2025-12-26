# 🏆 Proyecto: Titanic Competition

## 🎯 Objetivo

Desarrollar un pipeline completo de Machine Learning para predecir la supervivencia de pasajeros del Titanic, aplicando todas las técnicas aprendidas en el módulo de ML.

---

## 📋 Descripción

El RMS Titanic se hundió el 15 de abril de 1912 durante su viaje inaugural. De los 2224 pasajeros y tripulantes, más de 1500 murieron. Tu tarea es predecir qué pasajeros sobrevivieron al hundimiento.

### Dataset

| Variable    | Descripción                                 | Tipo       |
| ----------- | ------------------------------------------- | ---------- |
| PassengerId | ID único del pasajero                       | Numérico   |
| Survived    | 0 = No, 1 = Sí (variable target)            | Categórico |
| Pclass      | Clase del ticket (1 = 1ra, 2 = 2da, 3 = 3ra) | Categórico |
| Name        | Nombre del pasajero                         | Texto      |
| Sex         | Sexo                                        | Categórico |
| Age         | Edad en años                                | Numérico   |
| SibSp       | # de hermanos/esposos a bordo               | Numérico   |
| Parch       | # de padres/hijos a bordo                   | Numérico   |
| Ticket      | Número de ticket                            | Texto      |
| Fare        | Tarifa del pasajero                         | Numérico   |
| Cabin       | Número de cabina                            | Texto      |
| Embarked    | Puerto de embarque (C/Q/S)                  | Categórico |

---

## 📁 Estructura

```
titanic-competition/
├── README.md               # Este archivo
├── starter/
│   ├── main.py             # Código base para completar
│   └── requirements.txt    # Dependencias
├── solution/
│   └── main.py             # Solución completa
└── submissions/            # Carpeta para guardar submissions
    └── .gitkeep
```

---

## ✅ Requisitos del Proyecto

### 1. EDA Completo (20%)

- [ ] Análisis de distribuciones
- [ ] Missing values
- [ ] Correlaciones
- [ ] Visualizaciones informativas

### 2. Feature Engineering (25%)

- [ ] Manejo de missing values
- [ ] Creación de nuevas features (FamilySize, Title, etc.)
- [ ] Encoding de categóricas
- [ ] Escalado si es necesario

### 3. Modelado (25%)

- [ ] Baseline con DummyClassifier
- [ ] Mínimo 3 modelos diferentes
- [ ] Cross-validation adecuado
- [ ] Optimización de hiperparámetros

### 4. Pipeline de Producción (15%)

- [ ] sklearn Pipeline
- [ ] ColumnTransformer
- [ ] Sin data leakage

### 5. Submission y Documentación (15%)

- [ ] Generar submission.csv válida
- [ ] Documentar proceso y decisiones
- [ ] Código limpio y comentado

---

## 🎯 Métricas

| Nivel        | Accuracy CV | Descripción                       |
| ------------ | ----------- | --------------------------------- |
| 🔴 Baseline  | ~0.62       | DummyClassifier                   |
| 🟡 Aceptable | ≥ 0.75      | Modelo básico, features básicas   |
| 🟢 Bueno     | ≥ 0.80      | Feature engineering, tuning       |
| 🏆 Excelente | ≥ 0.82      | Pipeline optimizado, ensemble     |

---

## 📝 Instrucciones

1. **Configura el entorno**:
   ```bash
   cd starter
   pip install -r requirements.txt
   ```

2. **Abre `starter/main.py`** y completa los TODOs

3. **Ejecuta el código** para verificar cada sección

4. **Genera la submission** y guárdala en `submissions/`

5. **Documenta tus decisiones** en comentarios

---

## 🏆 Entregables

1. `main.py` completado con tu solución
2. `submission.csv` con predicciones
3. Documentación de decisiones técnicas

---

## 💡 Tips

- Empieza simple: baseline → modelo básico → feature engineering → tuning
- No hagas feature engineering sin validar que mejora el score
- Cuidado con el data leakage
- Guarda checkpoints de tu código funcional antes de experimentar

---

## ⏱️ Tiempo Estimado

- **3 horas** para completar todo el proyecto
- Distribuye: 30min EDA, 1h Feature Engineering, 1h Modelado, 30min Documentación

---

## 🔗 Recursos

- [Kaggle Titanic](https://www.kaggle.com/c/titanic)
- [Feature Engineering for ML](https://www.kaggle.com/learn/feature-engineering)
- [Sklearn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
