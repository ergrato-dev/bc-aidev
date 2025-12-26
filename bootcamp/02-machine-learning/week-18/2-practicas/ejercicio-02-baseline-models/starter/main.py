"""
Ejercicio 02: Baseline Models
=============================
Crear modelos baseline para el dataset Titanic.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import time

# ============================================
# PASO 1: Cargar y Preparar Datos
# ============================================
print('--- Paso 1: Cargar Datos ---')

# Descomenta las siguientes líneas:
# url_train = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
# train = pd.read_csv(url_train)
# 
# print(f'Shape: {train.shape}')
# 
# # Preparación básica
# # Features simples para baseline
# features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
# 
# # Preparar X e y
# X = train[features].copy()
# y = train['Survived']
# 
# # Encoding de Sex
# X['Sex'] = LabelEncoder().fit_transform(X['Sex'])
# 
# # Imputar Age y Fare
# X['Age'] = X['Age'].fillna(X['Age'].median())
# X['Fare'] = X['Fare'].fillna(X['Fare'].median())
# 
# print(f'Features: {features}')
# print(f'X shape: {X.shape}')

print()


# ============================================
# PASO 2: Baseline con DummyClassifier
# ============================================
print('--- Paso 2: Baseline (DummyClassifier) ---')

# Descomenta las siguientes líneas:
# # Estrategia: siempre predecir la clase más frecuente
# dummy_most_frequent = DummyClassifier(strategy='most_frequent')
# scores_mf = cross_val_score(dummy_most_frequent, X, y, cv=5, scoring='accuracy')
# print(f'Most Frequent: {scores_mf.mean():.4f} ± {scores_mf.std():.4f}')
# 
# # Estrategia: predecir proporcionalmente a la distribución
# dummy_stratified = DummyClassifier(strategy='stratified')
# scores_st = cross_val_score(dummy_stratified, X, y, cv=5, scoring='accuracy')
# print(f'Stratified: {scores_st.mean():.4f} ± {scores_st.std():.4f}')
# 
# # Estrategia: predecir uniformemente al azar
# dummy_uniform = DummyClassifier(strategy='uniform')
# scores_un = cross_val_score(dummy_uniform, X, y, cv=5, scoring='accuracy')
# print(f'Uniform: {scores_un.mean():.4f} ± {scores_un.std():.4f}')
# 
# baseline_score = scores_mf.mean()
# print(f'\n🎯 Baseline a superar: {baseline_score:.4f}')

print()


# ============================================
# PASO 3: Comparación Rápida de Modelos
# ============================================
print('--- Paso 3: Comparación de Modelos ---')

# Descomenta las siguientes líneas:
# models = {
#     'Logistic Regression': LogisticRegression(max_iter=1000),
#     'Decision Tree': DecisionTreeClassifier(random_state=42),
#     'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#     'Gradient Boosting': GradientBoostingClassifier(random_state=42),
#     'SVM': SVC(random_state=42),
#     'KNN': KNeighborsClassifier()
# }
# 
# results = []
# 
# for name, model in models.items():
#     start = time.time()
#     scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
#     elapsed = time.time() - start
#     
#     results.append({
#         'Model': name,
#         'CV Mean': scores.mean(),
#         'CV Std': scores.std(),
#         'Time': elapsed
#     })
#     
#     improvement = ((scores.mean() - baseline_score) / baseline_score) * 100
#     print(f'{name}: {scores.mean():.4f} ± {scores.std():.4f} | Time: {elapsed:.2f}s | Mejora: {improvement:+.1f}%')
# 
# results_df = pd.DataFrame(results).sort_values('CV Mean', ascending=False)
# print(f'\n=== Ranking de Modelos ===')
# print(results_df.to_string(index=False))

print()


# ============================================
# PASO 4: Visualizar Comparación
# ============================================
print('--- Paso 4: Visualizar Resultados ---')

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 
# # Barplot de accuracy
# colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
# bars = axes[0].barh(results_df['Model'], results_df['CV Mean'], 
#                     xerr=results_df['CV Std'], color=colors, capsize=5)
# axes[0].axvline(baseline_score, color='red', linestyle='--', label=f'Baseline: {baseline_score:.3f}')
# axes[0].set_xlabel('CV Accuracy')
# axes[0].set_title('Comparación de Modelos (CV 5-Fold)')
# axes[0].legend()
# axes[0].set_xlim(0.5, 0.9)
# 
# # Barplot de tiempo
# axes[1].barh(results_df['Model'], results_df['Time'], color='coral')
# axes[1].set_xlabel('Tiempo (segundos)')
# axes[1].set_title('Tiempo de Entrenamiento')
# 
# plt.tight_layout()
# plt.show()

print()


# ============================================
# PASO 5: Con Escalado (Pipeline)
# ============================================
print('--- Paso 5: Con Escalado ---')

# Descomenta las siguientes líneas:
# # Algunos modelos se benefician del escalado
# models_scaled = {
#     'Logistic Regression (scaled)': Pipeline([
#         ('scaler', StandardScaler()),
#         ('clf', LogisticRegression(max_iter=1000))
#     ]),
#     'SVM (scaled)': Pipeline([
#         ('scaler', StandardScaler()),
#         ('clf', SVC(random_state=42))
#     ]),
#     'KNN (scaled)': Pipeline([
#         ('scaler', StandardScaler()),
#         ('clf', KNeighborsClassifier())
#     ])
# }
# 
# print('=== Impacto del Escalado ===')
# for name, pipe in models_scaled.items():
#     scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
#     print(f'{name}: {scores.mean():.4f} ± {scores.std():.4f}')

print()


# ============================================
# PASO 6: Seleccionar Top 3
# ============================================
print('--- Paso 6: Seleccionar Top 3 ---')

# Descomenta las siguientes líneas:
# top_3 = results_df.head(3)['Model'].tolist()
# print(f'Top 3 modelos para optimizar: {top_3}')
# 
# print('''
# Próximos pasos:
# 1. Feature engineering más avanzado
# 2. GridSearchCV para optimizar hiperparámetros
# 3. Posible ensemble de los top modelos
# ''')

print()


# ============================================
# PASO 7: Análisis del Mejor Modelo
# ============================================
print('--- Paso 7: Análisis del Mejor Modelo ---')

# Descomenta las siguientes líneas:
# from sklearn.metrics import classification_report, confusion_matrix
# 
# # Train/test split para análisis
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # Entrenar mejor modelo
# best_model = GradientBoostingClassifier(random_state=42)
# best_model.fit(X_train, y_train)
# 
# # Evaluar
# y_pred = best_model.predict(X_val)
# 
# print('=== Classification Report ===')
# print(classification_report(y_val, y_pred, target_names=['No Survived', 'Survived']))
# 
# print('\n=== Confusion Matrix ===')
# print(confusion_matrix(y_val, y_pred))
# 
# # Feature importance
# print('\n=== Feature Importance ===')
# importance = pd.DataFrame({
#     'feature': features,
#     'importance': best_model.feature_importances_
# }).sort_values('importance', ascending=False)
# print(importance)

print()


# ============================================
# PASO 8: Guardar Resultados
# ============================================
print('--- Paso 8: Resumen ---')

# Descomenta las siguientes líneas:
# print('''
# ===========================================
# RESUMEN - BASELINE MODELS
# ===========================================
# 
# 🎯 Baseline (DummyClassifier): ~0.616
# 
# 📊 Top 3 Modelos:
# 1. Gradient Boosting: ~0.82
# 2. Random Forest: ~0.81
# 3. Logistic Regression: ~0.79
# 
# 💡 Observaciones:
# - Todos los modelos superan significativamente el baseline
# - Tree-based models funcionan mejor con features sin escalar
# - SVM y KNN mejoran con escalado
# 
# 🔜 Próximos pasos:
# - Feature engineering (FamilySize, Title, etc.)
# - GridSearchCV para hiperparámetros
# - Ensemble de modelos
# 
# ===========================================
# ''')

print()
print('=== Ejercicio completado ===')
