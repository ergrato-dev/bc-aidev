# ============================================
# PROYECTO: PIPELINE DE PREPROCESAMIENTO COMPLETO
# SOLUCIÓN DE REFERENCIA
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================
# PASO 1: Cargar Dataset
# ============================================
print('=== PASO 1: Cargar Dataset ===')

# URL del dataset Adult Income
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Nombres de columnas
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

# Cargar datos
df = pd.read_csv(url, names=column_names, sep=',\\s*', engine='python', na_values='?')

print(f"Dataset shape: {df.shape}")
print(f"\nPrimeras filas:")
print(df.head())

print()

# ============================================
# PASO 2: Exploración y Limpieza Básica
# ============================================
print('=== PASO 2: Exploración ===')

# Info del dataset
print("Tipos de datos:")
print(df.dtypes)

# Valores faltantes
print("\nValores faltantes:")
missing = df.isnull().sum()
print(missing[missing > 0])

# Distribución del target
print("\nDistribución del target:")
print(df['income'].value_counts(normalize=True))

# Separar features y target
X = df.drop(columns=['income'])
y = (df['income'] == '>50K').astype(int)

print(f"\nX shape: {X.shape}")
print(f"y distribution: {y.value_counts().to_dict()}")

print()

# ============================================
# PASO 3: Identificar Tipos de Columnas
# ============================================
print('=== PASO 3: Identificar Columnas ===')

# Columnas numéricas
numeric_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 
                    'capital_loss', 'hours_per_week']

# Columnas categóricas
categorical_features = ['workclass', 'education', 'marital_status', 'occupation',
                        'relationship', 'race', 'sex', 'native_country']

print(f"Numéricas ({len(numeric_features)}): {numeric_features}")
print(f"Categóricas ({len(categorical_features)}): {categorical_features}")

# Verificar
print(f"\nTotal features: {len(numeric_features) + len(categorical_features)}")

print()

# ============================================
# PASO 4: Crear Pipelines Individuales
# ============================================
print('=== PASO 4: Pipelines Individuales ===')

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Pipeline numérico: imputar con mediana + escalar
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline categórico: imputar con moda + one-hot encoding
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

print("Pipeline numérico:")
print(f"  1. SimpleImputer(strategy='median')")
print(f"  2. StandardScaler()")

print("\nPipeline categórico:")
print(f"  1. SimpleImputer(strategy='most_frequent')")
print(f"  2. OneHotEncoder(handle_unknown='ignore')")

print()

# ============================================
# PASO 5: Combinar con ColumnTransformer
# ============================================
print('=== PASO 5: ColumnTransformer ===')

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

print("ColumnTransformer creado:")
print(f"  - 'num': Pipeline numérico → {numeric_features}")
print(f"  - 'cat': Pipeline categórico → {categorical_features}")

print()

# ============================================
# PASO 6: Añadir Selector de Features
# ============================================
print('=== PASO 6: Feature Selection ===')

from sklearn.feature_selection import SelectKBest, f_classif

# Usamos SelectKBest para seleccionar las 30 mejores features
selector = SelectKBest(f_classif, k=30)

print("Selector de features: SelectKBest")
print(f"  - score_func: f_classif (ANOVA F-value)")
print(f"  - k: 30 features")

print()

# ============================================
# PASO 7: Pipeline Completo con Modelo
# ============================================
print('=== PASO 7: Pipeline Completo ===')

from sklearn.linear_model import LogisticRegression

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', selector),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

print("Pipeline completo:")
print("  1. preprocessor (ColumnTransformer)")
print("  2. selector (SelectKBest)")
print("  3. classifier (LogisticRegression)")

print()

# ============================================
# PASO 8: Train/Test Split
# ============================================
print('=== PASO 8: Train/Test Split ===')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"\nDistribución en train: {y_train.value_counts().to_dict()}")
print(f"Distribución en test: {y_test.value_counts().to_dict()}")

print()

# ============================================
# PASO 9: Entrenar y Evaluar
# ============================================
print('=== PASO 9: Entrenar y Evaluar ===')

# Entrenar pipeline completo
full_pipeline.fit(X_train, y_train)

# Predecir
y_pred = full_pipeline.predict(X_test)
y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]

# Métricas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix,
    roc_auc_score
)

print("Métricas en Test Set:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"  F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

# Matriz de confusión
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print()

# ============================================
# PASO 10: Cross-Validation
# ============================================
print('=== PASO 10: Cross-Validation ===')

from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    full_pipeline, X, y, cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    return_train_score=True,
    n_jobs=-1
)

print("Resultados Cross-Validation (5-fold):")
print("-" * 50)
for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    train_scores = cv_results[f'train_{metric}']
    test_scores = cv_results[f'test_{metric}']
    print(f"{metric:12s} | Train: {train_scores.mean():.4f} ± {train_scores.std():.4f} | "
          f"Test: {test_scores.mean():.4f} ± {test_scores.std():.4f}")

print()

# ============================================
# PASO 11: Análisis de Features Seleccionadas
# ============================================
print('=== PASO 11: Análisis de Features ===')

# Obtener nombres de features después del preprocesamiento
preprocessor_fitted = full_pipeline.named_steps['preprocessor']
feature_names_out = preprocessor_fitted.get_feature_names_out()

print(f"Features después de preprocesamiento: {len(feature_names_out)}")

# Ver cuáles seleccionó SelectKBest
selector_fitted = full_pipeline.named_steps['selector']
selected_mask = selector_fitted.get_support()
scores = selector_fitted.scores_

# Crear DataFrame con scores
features_df = pd.DataFrame({
    'feature': feature_names_out,
    'score': scores,
    'selected': selected_mask
}).sort_values('score', ascending=False)

print(f"\nTop 10 features por F-score:")
print(features_df.head(10).to_string(index=False))

# Features seleccionadas
selected_features = features_df[features_df['selected']]['feature'].tolist()
print(f"\nFeatures seleccionadas ({len(selected_features)}):")
for i, feat in enumerate(selected_features[:10], 1):
    print(f"  {i}. {feat}")
if len(selected_features) > 10:
    print(f"  ... y {len(selected_features) - 10} más")

# Visualizar importancia de features
plt.figure(figsize=(12, 8))
top_features = features_df.head(20)
colors = ['steelblue' if sel else 'lightgray' for sel in top_features['selected']]
plt.barh(top_features['feature'], top_features['score'], color=colors)
plt.xlabel('F-Score')
plt.title('Top 20 Features por F-Score (azul = seleccionadas)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()
print("\nGráfico guardado como 'feature_importance.png'")

print()

# ============================================
# PASO 12: Guardar Pipeline
# ============================================
print('=== PASO 12: Guardar Pipeline ===')

import joblib

# Guardar pipeline entrenado
joblib.dump(full_pipeline, 'pipeline_preprocesamiento.joblib')
print("Pipeline guardado como 'pipeline_preprocesamiento.joblib'")

# Demostrar cómo cargar y usar
print("\nPara usar el pipeline guardado:")
print("  loaded_pipeline = joblib.load('pipeline_preprocesamiento.joblib')")
print("  predictions = loaded_pipeline.predict(new_data)")

print()

# ============================================
# BONUS: GridSearchCV para Optimización
# ============================================
print('=== BONUS: GridSearchCV ===')

from sklearn.model_selection import GridSearchCV

# Definir parámetros a buscar
param_grid = {
    'selector__k': [20, 30, 40],
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__solver': ['lbfgs', 'liblinear']
}

# GridSearchCV (con muestra pequeña para rapidez)
grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Entrenar con muestra
sample_size = 5000
X_sample = X.sample(n=sample_size, random_state=42)
y_sample = y.loc[X_sample.index]

print(f"Ejecutando GridSearchCV con {sample_size} muestras...")
grid_search.fit(X_sample, y_sample)

print(f"\nMejores parámetros: {grid_search.best_params_}")
print(f"Mejor F1-Score (CV): {grid_search.best_score_:.4f}")

print()

# ============================================
# RESUMEN FINAL
# ============================================
print('=' * 60)
print('RESUMEN DEL PROYECTO')
print('=' * 60)
print(f"""
Dataset: Adult Income (UCI)
- Samples: {len(df):,}
- Features originales: {X.shape[1]}
- Features después de encoding: {len(feature_names_out)}
- Features seleccionadas: {sum(selected_mask)}

Pipeline:
1. ColumnTransformer
   - Numéricas: Imputer(median) → StandardScaler
   - Categóricas: Imputer(mode) → OneHotEncoder
2. SelectKBest(k=30)
3. LogisticRegression

Resultados (Test Set):
- Accuracy:  {accuracy_score(y_test, y_pred):.4f}
- F1-Score:  {f1_score(y_test, y_pred):.4f}
- ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}

Archivos generados:
- pipeline_preprocesamiento.joblib
- feature_importance.png

✓ Pipeline listo para producción
""")
