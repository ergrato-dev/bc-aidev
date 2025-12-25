# ============================================
# PROYECTO: PIPELINE DE PREPROCESAMIENTO COMPLETO
# ============================================
# Objetivo: Construir un pipeline end-to-end con
# sklearn Pipeline y ColumnTransformer
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================
# PASO 1: Cargar Dataset
# ============================================
print('=== PASO 1: Cargar Dataset ===')

# TODO: Cargar el dataset Adult Income
# Opción 1: Desde sklearn (si disponible)
# Opción 2: Desde URL

# URL del dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Nombres de columnas
# column_names = [
#     'age', 'workclass', 'fnlwgt', 'education', 'education_num',
#     'marital_status', 'occupation', 'relationship', 'race', 'sex',
#     'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
# ]

# TODO: Cargar datos
# df = pd.read_csv(url, names=column_names, sep=', ', engine='python')

# TODO: Explorar datos
# print(df.head())
# print(df.info())
# print(df.describe())

print()

# ============================================
# PASO 2: Exploración y Limpieza Básica
# ============================================
print('=== PASO 2: Exploración ===')

# TODO: Identificar valores faltantes
# El dataset usa '?' para missing values

# TODO: Reemplazar '?' con NaN
# df = df.replace('?', np.nan)

# TODO: Ver distribución del target
# print(df['income'].value_counts(normalize=True))

# TODO: Separar features y target
# X = df.drop(columns=['income'])
# y = (df['income'] == '>50K').astype(int)  # Convertir a binario

print()

# ============================================
# PASO 3: Identificar Tipos de Columnas
# ============================================
print('=== PASO 3: Identificar Columnas ===')

# TODO: Identificar columnas numéricas y categóricas

# numeric_features = [...]
# categorical_features = [...]

# Sugerencia de columnas:
# Numéricas: 'age', 'fnlwgt', 'education_num', 'capital_gain', 
#            'capital_loss', 'hours_per_week'
# Categóricas: 'workclass', 'education', 'marital_status', 'occupation',
#              'relationship', 'race', 'sex', 'native_country'

print()

# ============================================
# PASO 4: Crear Pipelines Individuales
# ============================================
print('=== PASO 4: Pipelines Individuales ===')

# TODO: Importar clases necesarias
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder

# TODO: Pipeline numérico
# numeric_transformer = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# TODO: Pipeline categórico
# categorical_transformer = Pipeline([
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
# ])

print()

# ============================================
# PASO 5: Combinar con ColumnTransformer
# ============================================
print('=== PASO 5: ColumnTransformer ===')

# TODO: Crear ColumnTransformer
# from sklearn.compose import ColumnTransformer

# preprocessor = ColumnTransformer([
#     ('num', numeric_transformer, numeric_features),
#     ('cat', categorical_transformer, categorical_features)
# ])

print()

# ============================================
# PASO 6: Añadir Selector de Features
# ============================================
print('=== PASO 6: Feature Selection ===')

# TODO: Añadir selector de features al pipeline
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import RandomForestClassifier

# Opción 1: SelectKBest
# selector = SelectKBest(f_classif, k=20)

# Opción 2: SelectFromModel
# selector = SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42))

print()

# ============================================
# PASO 7: Pipeline Completo con Modelo
# ============================================
print('=== PASO 7: Pipeline Completo ===')

# TODO: Crear pipeline completo
# from sklearn.linear_model import LogisticRegression

# full_pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('selector', SelectKBest(f_classif, k=20)),
#     ('classifier', LogisticRegression(max_iter=1000, random_state=42))
# ])

print()

# ============================================
# PASO 8: Train/Test Split
# ============================================
print('=== PASO 8: Train/Test Split ===')

# TODO: Dividir datos
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# print(f"Train size: {len(X_train)}")
# print(f"Test size: {len(X_test)}")

print()

# ============================================
# PASO 9: Entrenar y Evaluar
# ============================================
print('=== PASO 9: Entrenar y Evaluar ===')

# TODO: Entrenar pipeline
# full_pipeline.fit(X_train, y_train)

# TODO: Evaluar en test
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, 
#     f1_score, classification_report, confusion_matrix
# )

# y_pred = full_pipeline.predict(X_test)

# print("Métricas en Test Set:")
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print(f"Precision: {precision_score(y_test, y_pred):.4f}")
# print(f"Recall: {recall_score(y_test, y_pred):.4f}")
# print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

print()

# ============================================
# PASO 10: Cross-Validation
# ============================================
print('=== PASO 10: Cross-Validation ===')

# TODO: Evaluar con cross-validation
# from sklearn.model_selection import cross_val_score, cross_validate

# cv_results = cross_validate(
#     full_pipeline, X, y, cv=5,
#     scoring=['accuracy', 'precision', 'recall', 'f1'],
#     return_train_score=True
# )

# print("Resultados Cross-Validation (5-fold):")
# for metric in ['accuracy', 'precision', 'recall', 'f1']:
#     scores = cv_results[f'test_{metric}']
#     print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")

print()

# ============================================
# PASO 11: Análisis de Features Seleccionadas
# ============================================
print('=== PASO 11: Análisis de Features ===')

# TODO: Ver qué features fueron seleccionadas
# Primero, obtener nombres de features después del preprocesamiento

# preprocessor.fit(X_train, y_train)
# feature_names_out = preprocessor.get_feature_names_out()

# Luego, ver cuáles seleccionó SelectKBest
# selector_fitted = full_pipeline.named_steps['selector']
# selected_mask = selector_fitted.get_support()

# selected_features = [name for name, sel in zip(feature_names_out, selected_mask) if sel]
# print(f"Features seleccionadas ({len(selected_features)}):")
# for feat in selected_features:
#     print(f"  - {feat}")

print()

# ============================================
# PASO 12: Guardar Pipeline
# ============================================
print('=== PASO 12: Guardar Pipeline ===')

# TODO: Guardar pipeline entrenado
# import joblib

# joblib.dump(full_pipeline, 'pipeline_preprocesamiento.joblib')
# print("Pipeline guardado como 'pipeline_preprocesamiento.joblib'")

# Para cargar:
# loaded_pipeline = joblib.load('pipeline_preprocesamiento.joblib')

print()

# ============================================
# RESUMEN DEL PROYECTO
# ============================================
print('=== RESUMEN ===')
print("""
Pipeline de Preprocesamiento Completo:

1. Carga de datos con manejo de missing ('?')
2. Identificación de columnas numéricas/categóricas
3. Pipeline numérico: Imputer(median) → StandardScaler
4. Pipeline categórico: Imputer(mode) → OneHotEncoder
5. ColumnTransformer para combinar pipelines
6. SelectKBest para selección de features
7. LogisticRegression como clasificador
8. Evaluación con métricas y cross-validation
9. Pipeline guardado para producción

Ventajas de este enfoque:
✓ Reproducible
✓ Sin data leakage
✓ Fácil de guardar y desplegar
✓ Compatible con GridSearchCV
✓ Maneja datos nuevos automáticamente
""")
