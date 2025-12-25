"""
Ejercicio 01: Cross-Validation en Práctica
==========================================
Aprende a usar diferentes estrategias de Cross-Validation.
"""

# ============================================
# PASO 1: Setup y Datos
# ============================================
print('--- Paso 1: Setup y Datos ---')

# Descomenta las siguientes líneas:
# from sklearn.datasets import load_breast_cancer
# import pandas as pd
# import numpy as np
# 
# # Cargar datos
# data = load_breast_cancer()
# X, y = data.data, data.target
# 
# print(f"Muestras: {X.shape[0]}")
# print(f"Features: {X.shape[1]}")
# print(f"Clases: {data.target_names}")
# print(f"Distribución: {pd.Series(y).value_counts().to_dict()}")

print()

# ============================================
# PASO 2: K-Fold Manual
# ============================================
print('--- Paso 2: K-Fold Manual ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import KFold
# from sklearn.linear_model import LogisticRegression
# 
# # Configurar K-Fold
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# model = LogisticRegression(max_iter=10000, random_state=42)
# 
# scores = []
# for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
#     # Dividir datos
#     X_train, X_val = X[train_idx], X[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]
#     
#     # Entrenar
#     model.fit(X_train, y_train)
#     
#     # Evaluar
#     score = model.score(X_val, y_val)
#     scores.append(score)
#     print(f"Fold {fold + 1}: Accuracy = {score:.4f}")
# 
# print(f"\nPromedio: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

print()

# ============================================
# PASO 3: cross_val_score
# ============================================
print('--- Paso 3: cross_val_score ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import cross_val_score
# 
# model = LogisticRegression(max_iter=10000, random_state=42)
# 
# # Una línea hace todo el trabajo
# scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
# 
# print(f"Scores por fold: {scores.round(4)}")
# print(f"Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

print()

# ============================================
# PASO 4: Stratified K-Fold
# ============================================
print('--- Paso 4: Stratified K-Fold ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import StratifiedKFold
# 
# # K-Fold regular
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# scores_kf = cross_val_score(model, X, y, cv=kf)
# 
# # Stratified K-Fold (mantiene proporción de clases)
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# scores_skf = cross_val_score(model, X, y, cv=skf)
# 
# print(f"K-Fold regular:     {scores_kf.mean():.4f} ± {scores_kf.std():.4f}")
# print(f"Stratified K-Fold:  {scores_skf.mean():.4f} ± {scores_skf.std():.4f}")
# 
# # Verificar proporción en cada fold
# print("\nProporción de clase 1 en cada fold (Stratified):")
# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#     prop_train = y[train_idx].mean()
#     prop_val = y[val_idx].mean()
#     print(f"  Fold {fold+1}: Train={prop_train:.3f}, Val={prop_val:.3f}")

print()

# ============================================
# PASO 5: Comparar Múltiples Modelos
# ============================================
print('--- Paso 5: Comparar Múltiples Modelos ---')

# Descomenta las siguientes líneas:
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# 
# models = {
#     'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
#     'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#     'SVM (RBF)': SVC(kernel='rbf', random_state=42),
#     'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
#     'Decision Tree': DecisionTreeClassifier(random_state=42)
# }
# 
# print("Comparación de modelos con 5-Fold CV:")
# print("-" * 50)
# 
# results = []
# for name, model in models.items():
#     scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
#     results.append({
#         'Modelo': name,
#         'Mean': scores.mean(),
#         'Std': scores.std()
#     })
#     print(f"{name:25s}: {scores.mean():.4f} ± {scores.std():.4f}")
# 
# # Mejor modelo
# best = max(results, key=lambda x: x['Mean'])
# print(f"\n✓ Mejor modelo: {best['Modelo']} ({best['Mean']:.4f})")

print()

# ============================================
# PASO 6: cross_validate (Múltiples Métricas)
# ============================================
print('--- Paso 6: cross_validate (Múltiples Métricas) ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import cross_validate
# 
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# 
# # Múltiples métricas a la vez
# scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
# 
# results = cross_validate(
#     model, X, y, cv=5,
#     scoring=scoring,
#     return_train_score=True,
#     n_jobs=-1
# )
# 
# print("Resultados con múltiples métricas:")
# print("-" * 50)
# for metric in scoring:
#     train_key = f'train_{metric}'
#     test_key = f'test_{metric}'
#     train_score = results[train_key].mean()
#     test_score = results[test_key].mean()
#     print(f"{metric:15s}: Train={train_score:.4f}, Test={test_score:.4f}")
# 
# print(f"\nTiempo de fit: {results['fit_time'].mean():.3f}s")

print()

# ============================================
# PASO 7: Impacto de K
# ============================================
print('--- Paso 7: Impacto de K ---')

# Descomenta las siguientes líneas:
# model = LogisticRegression(max_iter=10000, random_state=42)
# 
# print("Impacto del número de folds (K):")
# print("-" * 50)
# 
# k_values = [2, 3, 5, 10, 20]
# 
# for k in k_values:
#     scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')
#     print(f"K={k:2d}: {scores.mean():.4f} ± {scores.std():.4f}  (folds: {len(scores)})")
# 
# print("\nObservaciones:")
# print("- K pequeño: Más varianza, menos computación")
# print("- K grande: Menos varianza, más computación")
# print("- K=5 o K=10 son los valores más comunes")

print()

# ============================================
# PASO 8: Resumen y Conclusiones
# ============================================
print('--- Paso 8: Resumen ---')

# Descomenta las siguientes líneas:
# print("Resumen de Cross-Validation:")
# print("=" * 50)
# print("1. K-Fold: División aleatoria en K partes")
# print("2. Stratified: Mantiene proporción de clases")
# print("3. cross_val_score: Función simple para una métrica")
# print("4. cross_validate: Múltiples métricas + tiempos")
# print("5. K típico: 5 (rápido) o 10 (más estable)")
# print("=" * 50)

print()
print("¡Ejercicio completado!")
