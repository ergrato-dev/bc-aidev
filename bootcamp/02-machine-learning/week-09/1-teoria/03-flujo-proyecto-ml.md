# 🔄 Flujo de un Proyecto de Machine Learning

## 🎯 Objetivos

- Comprender el pipeline completo de un proyecto ML
- Conocer cada fase del proceso de desarrollo
- Identificar herramientas para cada etapa
- Aplicar buenas prácticas en cada fase

---

## 1. El Pipeline de Machine Learning

![Pipeline de Machine Learning](../0-assets/03-pipeline-ml.svg)

---

## 2. Fase 1: Definición del Problema

### Preguntas Clave

- ¿Qué problema de negocio queremos resolver?
- ¿ML es la solución adecuada?
- ¿Qué queremos predecir?
- ¿Cómo mediremos el éxito?

### Ejemplo

```python
# Definición clara del problema
problema = {
    'objetivo': 'Predecir si un cliente cancelará su suscripción',
    'tipo': 'Clasificación binaria',
    'metrica_exito': 'F1-Score > 0.80',
    'impacto_negocio': 'Retener clientes en riesgo'
}
```

### Checklist

- [ ] Problema definido claramente
- [ ] Tipo de ML identificado (clasificación/regresión/clustering)
- [ ] Métrica de éxito establecida
- [ ] Factibilidad evaluada

---

## 3. Fase 2: Recolección de Datos

### Fuentes Comunes

- Bases de datos internas
- APIs externas
- Web scraping
- Datasets públicos (Kaggle, UCI)
- Sensores / IoT

### Ejemplo

```python
import pandas as pd

# Cargar datos de diferentes fuentes
df_csv = pd.read_csv('data/clientes.csv')
df_sql = pd.read_sql('SELECT * FROM clientes', conexion)
df_api = pd.DataFrame(requests.get(url).json())

# Combinar fuentes
df = pd.merge(df_csv, df_sql, on='cliente_id')

print(f'Registros totales: {len(df):,}')
print(f'Features: {df.columns.tolist()}')
```

### Consideraciones

- **Cantidad**: ¿Suficientes datos para entrenar?
- **Calidad**: ¿Datos limpios y confiables?
- **Representatividad**: ¿Reflejan el problema real?
- **Legalidad**: ¿Tenemos permiso para usarlos?

---

## 4. Fase 3: Exploración de Datos (EDA)

### Objetivo

Entender los datos antes de modelar.

### Análisis Básico

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/dataset.csv')

# Información general
print('Shape:', df.shape)
print('\nTipos de datos:')
print(df.dtypes)

# Estadísticas descriptivas
print('\nEstadísticas:')
print(df.describe())

# Valores faltantes
print('\nValores nulos:')
print(df.isnull().sum())

# Distribución de la variable objetivo
print('\nDistribución target:')
print(df['target'].value_counts(normalize=True))
```

### Visualizaciones Clave

```python
# Distribución de features numéricos
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

df['edad'].hist(ax=axes[0, 0], bins=30)
axes[0, 0].set_title('Distribución de Edad')

df['ingresos'].hist(ax=axes[0, 1], bins=30)
axes[0, 1].set_title('Distribución de Ingresos')

# Correlaciones
sns.heatmap(df.corr(), annot=True, ax=axes[1, 0])
axes[1, 0].set_title('Matriz de Correlación')

# Target balance
df['target'].value_counts().plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Balance de Clases')

plt.tight_layout()
plt.savefig('eda_plots.png')
```

### Preguntas de EDA

- ¿Hay valores faltantes? ¿Cuántos?
- ¿Las clases están balanceadas?
- ¿Hay outliers?
- ¿Qué features correlacionan con el target?

---

## 5. Fase 4: Preparación de Datos

### Tareas Comunes

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Manejo de valores faltantes
imputer = SimpleImputer(strategy='median')
df['edad'] = imputer.fit_transform(df[['edad']])

# 2. Encoding de variables categóricas
le = LabelEncoder()
df['genero_encoded'] = le.fit_transform(df['genero'])

# 3. Escalado de features
scaler = StandardScaler()
df[['edad', 'ingresos']] = scaler.fit_transform(df[['edad', 'ingresos']])

# 4. Separación features / target
X = df.drop('target', axis=1)
y = df['target']

# 5. División train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Mantener proporción de clases
)

print(f'Train: {len(X_train)}, Test: {len(X_test)}')
```

### Pipeline de Preprocesamiento

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Definir transformaciones por tipo de columna
numeric_features = ['edad', 'ingresos']
categorical_features = ['genero', 'ciudad']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Pipeline completo
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

---

## 6. Fase 5: Entrenamiento del Modelo

### Selección de Algoritmo

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Probar múltiples algoritmos
modelos = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

resultados = {}
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    score = modelo.score(X_test, y_test)
    resultados[nombre] = score
    print(f'{nombre}: {score:.4f}')
```

### Ajuste de Hiperparámetros

```python
from sklearn.model_selection import GridSearchCV

# Definir espacio de búsqueda
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Grid Search con validación cruzada
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print('Mejores parámetros:', grid_search.best_params_)
print('Mejor score:', grid_search.best_score_)

# Usar mejor modelo
mejor_modelo = grid_search.best_estimator_
```

---

## 7. Fase 6: Evaluación del Modelo

### Métricas de Clasificación

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Predicciones
y_pred = modelo.predict(X_test)

# Métricas
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1-Score:', f1_score(y_test, y_pred))

# Reporte completo
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:')
print(cm)
```

### Visualización de Resultados

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Matriz de confusión visual
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.savefig('confusion_matrix.png')
```

### Métricas de Regresión

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = modelo.predict(X_test)

print('MSE:', mean_squared_error(y_test, y_pred))
print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('R²:', r2_score(y_test, y_pred))
```

---

## 8. Fase 7: Despliegue (Deploy)

### Guardar el Modelo

```python
import joblib
import pickle

# Opción 1: joblib (recomendado para sklearn)
joblib.dump(modelo, 'modelo_final.joblib')

# Cargar
modelo_cargado = joblib.load('modelo_final.joblib')

# Opción 2: pickle
with open('modelo_final.pkl', 'wb') as f:
    pickle.dump(modelo, f)
```

### API Simple con Flask

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
modelo = joblib.load('modelo_final.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    datos = request.json
    features = [[datos['edad'], datos['ingresos']]]
    prediccion = modelo.predict(features)
    return jsonify({'prediccion': int(prediccion[0])})

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 9. Fase 8: Monitoreo y Mejora

### Aspectos a Monitorear

- **Data Drift**: ¿Los datos nuevos son diferentes?
- **Model Performance**: ¿El modelo sigue siendo preciso?
- **Latencia**: ¿Las predicciones son rápidas?
- **Errores**: ¿En qué casos falla el modelo?

### Re-entrenamiento

```python
# Pseudocódigo de pipeline de re-entrenamiento
def reentrenar_modelo():
    # 1. Obtener nuevos datos
    nuevos_datos = obtener_datos_recientes()

    # 2. Evaluar modelo actual
    score_actual = evaluar_modelo(modelo_actual, nuevos_datos)

    # 3. Si el rendimiento baja, re-entrenar
    if score_actual < threshold:
        nuevo_modelo = entrenar_modelo(todos_los_datos)
        if evaluar_modelo(nuevo_modelo, test_set) > score_actual:
            desplegar(nuevo_modelo)
```

---

## 10. Resumen del Pipeline

| Fase          | Objetivo               | Herramientas          |
| ------------- | ---------------------- | --------------------- |
| 1. Definir    | Clarificar el problema | Documentación         |
| 2. Recolectar | Obtener datos          | Pandas, SQL, APIs     |
| 3. Explorar   | Entender datos         | Pandas, Matplotlib    |
| 4. Preparar   | Limpiar y transformar  | Sklearn preprocessing |
| 5. Entrenar   | Crear modelo           | Sklearn, GridSearchCV |
| 6. Evaluar    | Medir rendimiento      | Sklearn metrics       |
| 7. Deploy     | Poner en producción    | Flask, Docker, Cloud  |
| 8. Monitorear | Mantener rendimiento   | Logging, dashboards   |

---

## 11. Código Completo: Mini Pipeline

```python
"""Pipeline completo de ML en miniatura"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1-2. Definir problema y cargar datos
df = pd.read_csv('data/customers.csv')

# 3. Explorar (simplificado)
print('Shape:', df.shape)
print('Target distribution:', df['churn'].value_counts())

# 4. Preparar datos
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Entrenar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train_scaled, y_train)

# 6. Evaluar
y_pred = modelo.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# 7. Guardar para deploy
joblib.dump(modelo, 'modelo_churn.joblib')
joblib.dump(scaler, 'scaler_churn.joblib')

print('✅ Pipeline completado')
```

---

## ✅ Checklist de Proyecto ML

- [ ] Problema definido claramente
- [ ] Datos recolectados y explorados
- [ ] EDA completo documentado
- [ ] Datos preprocesados correctamente
- [ ] Train/test split realizado
- [ ] Múltiples modelos probados
- [ ] Hiperparámetros optimizados
- [ ] Métricas de evaluación calculadas
- [ ] Modelo guardado
- [ ] Documentación creada

---

## 🔗 Navegación

| Anterior                                          | Siguiente                                                  |
| ------------------------------------------------- | ---------------------------------------------------------- |
| [← Tipos de Aprendizaje](02-tipos-aprendizaje.md) | [Conceptos Fundamentales →](04-conceptos-fundamentales.md) |
