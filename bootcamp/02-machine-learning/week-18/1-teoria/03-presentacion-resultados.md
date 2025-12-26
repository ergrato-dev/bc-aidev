# 📈 Presentación de Resultados

## 🎯 Objetivo

Aprender a comunicar efectivamente los resultados de un proyecto de Machine Learning.

---

## 📚 Contenido

### 1. Estructura de un Reporte ML

Un buen reporte de proyecto ML debe incluir:

1. **Resumen Ejecutivo**
2. **Definición del Problema**
3. **Análisis Exploratorio**
4. **Metodología**
5. **Resultados**
6. **Conclusiones**
7. **Próximos Pasos**

---

### 2. Resumen Ejecutivo

```markdown
## Resumen Ejecutivo

**Problema**: Predecir supervivencia de pasajeros del Titanic
**Solución**: Modelo de clasificación con Gradient Boosting
**Resultado**: 84.3% accuracy en validación cruzada
**Impacto**: Top 15% en leaderboard de Kaggle

### Métricas Clave

| Métrica   | Valor |
| --------- | ----- |
| Accuracy  | 0.843 |
| Precision | 0.825 |
| Recall    | 0.798 |
| F1-Score  | 0.811 |
```

---

### 3. Visualizaciones Efectivas

#### 3.1 Distribución del Target

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Conteo absoluto
train['Survived'].value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'])
axes[0].set_title('Distribución de Supervivencia')
axes[0].set_xticklabels(['No Sobrevivió', 'Sobrevivió'], rotation=0)

# Porcentaje
train['Survived'].value_counts(normalize=True).plot(kind='pie', ax=axes[1],
    autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'])
axes[1].set_title('Porcentaje de Supervivencia')

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=150)
```

#### 3.2 Correlaciones

```python
# Mapa de correlaciones
plt.figure(figsize=(10, 8))
numeric_cols = train.select_dtypes(include=[np.number]).columns
corr_matrix = train[numeric_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, linewidths=0.5)
plt.title('Matriz de Correlaciones')
plt.tight_layout()
plt.savefig('correlations.png', dpi=150)
```

#### 3.3 Feature Importance

```python
# Importancia de features
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
plt.xlabel('Importancia')
plt.title('Importancia de Features - Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
```

#### 3.4 Comparación de Modelos

![Model Comparison](../0-assets/03-model-comparison.svg)

```python
# Tabla comparativa
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM'],
    'CV Accuracy': [0.782, 0.821, 0.843, 0.815],
    'CV Std': [0.032, 0.025, 0.021, 0.028],
    'Train Time': ['0.02s', '2.5s', '5.1s', '1.2s']
})

# Visualizar
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
bars = ax.bar(results['Model'], results['CV Accuracy'], yerr=results['CV Std'],
              color=colors, capsize=5, alpha=0.8)
ax.set_ylabel('CV Accuracy')
ax.set_title('Comparación de Modelos')
ax.set_ylim(0.7, 0.9)

# Añadir valores
for bar, acc in zip(bars, results['CV Accuracy']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{acc:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
```

#### 3.5 Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Matriz de confusión
cm = confusion_matrix(y_val, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(cm, display_labels=['No Sobrevivió', 'Sobrevivió'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Matriz de Confusión')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
```

#### 3.6 Curva ROC

```python
from sklearn.metrics import roc_curve, auc

# Calcular curva ROC
y_proba = model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
```

---

### 4. Análisis de Errores

```python
# Identificar errores
errors = X_val.copy()
errors['y_true'] = y_val
errors['y_pred'] = y_pred
errors['correct'] = errors['y_true'] == errors['y_pred']

# Analizar errores
print("=== Análisis de Errores ===")
print(f"\nTotal errores: {(~errors['correct']).sum()} / {len(errors)}")

# False Negatives (predijo 0, era 1)
fn = errors[(errors['y_true'] == 1) & (errors['y_pred'] == 0)]
print(f"\nFalse Negatives (no predijo supervivencia): {len(fn)}")
print(fn[['Age', 'Sex', 'Pclass', 'Fare']].describe())

# False Positives (predijo 1, era 0)
fp = errors[(errors['y_true'] == 0) & (errors['y_pred'] == 1)]
print(f"\nFalse Positives (predijo supervivencia incorrecta): {len(fp)}")
print(fp[['Age', 'Sex', 'Pclass', 'Fare']].describe())
```

---

### 5. Estructura del README del Proyecto

```markdown
# 🚢 Titanic Survival Prediction

## 📋 Descripción

Modelo de Machine Learning para predecir la supervivencia de pasajeros del Titanic.

## 🎯 Resultados

- **Accuracy**: 84.3%
- **F1-Score**: 81.1%
- **Kaggle Score**: 0.79425

## 🗂️ Estructura
```

titanic-competition/
├── notebooks/
│ ├── 01_EDA.ipynb
│ ├── 02_Feature_Engineering.ipynb
│ └── 03_Modeling.ipynb
├── src/
│ ├── preprocessing.py
│ └── pipeline.py
├── data/
│ ├── train.csv
│ └── test.csv
├── submissions/
│ └── submission.csv
├── models/
│ └── best_model.pkl
└── README.md

````

## 🔧 Instalación
```bash
pip install -r requirements.txt
````

## 🚀 Uso

```python
from src.pipeline import TitanicPipeline

pipeline = TitanicPipeline()
pipeline.train('data/train.csv')
predictions = pipeline.predict('data/test.csv')
```

## 📊 Features Importantes

1. Sex (0.25)
2. Fare (0.18)
3. Age (0.15)
4. Pclass (0.12)

## 📈 Mejoras Futuras

- [ ] Ensemble de modelos
- [ ] Feature engineering adicional
- [ ] Análisis de errores más profundo

````

---

### 6. Presentación en Slides

**Estructura sugerida (10 slides)**:

1. **Título**: Proyecto, autor, fecha
2. **Problema**: Qué estamos resolviendo
3. **Datos**: Descripción del dataset
4. **EDA**: 2-3 insights principales
5. **Feature Engineering**: Features creadas
6. **Modelos**: Comparación de algoritmos
7. **Resultados**: Métricas finales
8. **Análisis de Errores**: Qué falla
9. **Conclusiones**: Aprendizajes
10. **Próximos Pasos**: Mejoras futuras

---

### 7. Código Limpio para Presentar

```python
"""
Titanic Survival Prediction - Clean Pipeline
=============================================
Autor: Tu Nombre
Fecha: 2024-01
Score: 0.843 CV Accuracy
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

def create_pipeline():
    """Crea el pipeline de preprocesamiento y modelo."""

    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
    categorical_features = ['Sex', 'Embarked', 'Pclass']

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ))
    ])

def main():
    """Función principal."""
    # Cargar datos
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Preparar
    features = ['Age', 'Fare', 'SibSp', 'Parch', 'Sex', 'Embarked', 'Pclass']
    X = train[features]
    y = train['Survived']

    # Entrenar
    pipeline = create_pipeline()
    pipeline.fit(X, y)

    # Predecir
    predictions = pipeline.predict(test[features])

    # Submission
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv('submission.csv', index=False)

    print("✅ Submission generada: submission.csv")

if __name__ == '__main__':
    main()
````

---

## ✅ Checklist de Presentación

- [ ] Resumen ejecutivo claro
- [ ] Visualizaciones informativas
- [ ] Comparación de modelos documentada
- [ ] Métricas apropiadas reportadas
- [ ] Análisis de errores incluido
- [ ] Código limpio y documentado
- [ ] README completo
- [ ] Próximos pasos definidos

---

## 🔗 Navegación

| ⬅️ Anterior                             | 🏠 Semana                 | Siguiente ➡️                                                    |
| --------------------------------------- | ------------------------- | --------------------------------------------------------------- |
| [Pipelines](02-pipelines-produccion.md) | [Semana 18](../README.md) | [Prácticas](../2-practicas/ejercicio-01-eda-completo/README.md) |
