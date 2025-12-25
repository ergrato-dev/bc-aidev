"""
Ejercicio 04: GridSearchCV y Optimización
=========================================
Aprende a optimizar hiperparámetros con GridSearch y RandomizedSearch.
"""

# ============================================
# PASO 1: GridSearchCV Básico
# ============================================
print('--- Paso 1: GridSearchCV Básico ---')

# Descomenta las siguientes líneas:
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# import warnings
# warnings.filterwarnings('ignore')
# 
# # Cargar datos
# data = load_breast_cancer()
# X, y = data.data, data.target
# 
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
# 
# print(f"Train: {len(X_train)}, Test: {len(X_test)}")
# 
# # Definir grid de hiperparámetros
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [5, 10, 20, None],
#     'min_samples_split': [2, 5, 10]
# }
# 
# total_combinations = 3 * 4 * 3
# print(f"\nTotal de combinaciones: {total_combinations}")
# print(f"Con 5-fold CV: {total_combinations * 5} entrenamientos")
# 
# # GridSearchCV
# model = RandomForestClassifier(random_state=42, n_jobs=-1)
# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=1,
#     return_train_score=True
# )
# 
# print("\nEjecutando GridSearchCV...")
# grid_search.fit(X_train, y_train)
# print("✓ Completado")

print()

# ============================================
# PASO 2: Analizar Resultados
# ============================================
print('--- Paso 2: Analizar Resultados ---')

# Descomenta las siguientes líneas:
# # Mejores hiperparámetros
# print("Mejores hiperparámetros encontrados:")
# for param, value in grid_search.best_params_.items():
#     print(f"  {param}: {value}")
# 
# print(f"\nMejor score CV: {grid_search.best_score_:.4f}")
# print(f"Score en test: {grid_search.score(X_test, y_test):.4f}")
# 
# # DataFrame de resultados
# results = pd.DataFrame(grid_search.cv_results_)
# 
# # Top 5 configuraciones
# print("\nTop 5 configuraciones:")
# cols = ['rank_test_score', 'mean_test_score', 'std_test_score', 
#         'param_n_estimators', 'param_max_depth', 'param_min_samples_split']
# print(results[cols].sort_values('rank_test_score').head())
# 
# # Visualizar resultados
# pivot = results.pivot_table(
#     values='mean_test_score',
#     index='param_max_depth',
#     columns='param_n_estimators',
#     aggfunc='mean'
# )
# 
# fig, ax = plt.subplots(figsize=(8, 6))
# im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
# ax.set_xticks(range(len(pivot.columns)))
# ax.set_xticklabels(pivot.columns)
# ax.set_yticks(range(len(pivot.index)))
# ax.set_yticklabels([str(x) for x in pivot.index])
# ax.set_xlabel('n_estimators')
# ax.set_ylabel('max_depth')
# plt.colorbar(im, label='Mean CV Score')
# plt.title('GridSearchCV: Accuracy por Hiperparámetros')
# plt.tight_layout()
# plt.savefig('gridsearch_results.png', dpi=100)
# plt.close()
# print("\n✓ Gráfica guardada: gridsearch_results.png")

print()

# ============================================
# PASO 3: RandomizedSearchCV
# ============================================
print('--- Paso 3: RandomizedSearchCV ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint, uniform
# 
# # Distribuciones de hiperparámetros (espacios más grandes)
# param_distributions = {
#     'n_estimators': randint(50, 500),
#     'max_depth': randint(5, 50),
#     'min_samples_split': randint(2, 20),
#     'min_samples_leaf': randint(1, 10),
#     'max_features': uniform(0.1, 0.9)
# }
# 
# model = RandomForestClassifier(random_state=42, n_jobs=-1)
# 
# random_search = RandomizedSearchCV(
#     estimator=model,
#     param_distributions=param_distributions,
#     n_iter=30,           # Solo 30 combinaciones aleatorias
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1,
#     random_state=42,
#     verbose=1
# )
# 
# print("Ejecutando RandomizedSearchCV (30 iteraciones)...")
# random_search.fit(X_train, y_train)
# print("✓ Completado")
# 
# print(f"\nMejores hiperparámetros:")
# for param, value in random_search.best_params_.items():
#     print(f"  {param}: {value}")
# 
# print(f"\nMejor score CV: {random_search.best_score_:.4f}")
# print(f"Score en test: {random_search.score(X_test, y_test):.4f}")
# 
# # Comparar
# print(f"\nComparación GridSearch vs RandomizedSearch:")
# print(f"  GridSearch (36 combinaciones):     {grid_search.best_score_:.4f}")
# print(f"  RandomSearch (30 combinaciones):   {random_search.best_score_:.4f}")

print()

# ============================================
# PASO 4: Pipeline con GridSearch
# ============================================
print('--- Paso 4: Pipeline con GridSearch ---')

# Descomenta las siguientes líneas:
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.svm import SVC
# 
# # Pipeline completo
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('pca', PCA()),
#     ('svm', SVC(random_state=42))
# ])
# 
# # Grid para el pipeline (usar nombre_paso__parametro)
# param_grid_pipeline = {
#     'pca__n_components': [5, 10, 15, 20],
#     'svm__C': [0.1, 1, 10],
#     'svm__kernel': ['rbf', 'linear'],
#     'svm__gamma': ['scale', 'auto']
# }
# 
# grid_pipeline = GridSearchCV(
#     pipeline,
#     param_grid_pipeline,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=1
# )
# 
# print("Optimizando pipeline completo...")
# grid_pipeline.fit(X_train, y_train)
# print("✓ Completado")
# 
# print(f"\nMejores parámetros del pipeline:")
# for param, value in grid_pipeline.best_params_.items():
#     print(f"  {param}: {value}")
# 
# print(f"\nScore CV: {grid_pipeline.best_score_:.4f}")
# print(f"Score test: {grid_pipeline.score(X_test, y_test):.4f}")

print()

# ============================================
# PASO 5: Nested Cross-Validation
# ============================================
print('--- Paso 5: Nested Cross-Validation ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import cross_val_score
# 
# # Sin Nested CV: optimistic bias
# # Con Nested CV: evaluación honesta
# 
# print("Nested CV: CV externo evalúa el proceso completo de GridSearch")
# print("-" * 60)
# 
# # GridSearch interno
# param_grid_nested = {
#     'n_estimators': [50, 100],
#     'max_depth': [5, 10, 20]
# }
# 
# inner_cv = 3  # CV interno para selección de hiperparámetros
# outer_cv = 5  # CV externo para evaluación
# 
# grid_inner = GridSearchCV(
#     RandomForestClassifier(random_state=42, n_jobs=-1),
#     param_grid_nested,
#     cv=inner_cv,
#     scoring='accuracy',
#     n_jobs=-1
# )
# 
# # Nested CV
# print("Ejecutando Nested CV...")
# nested_scores = cross_val_score(grid_inner, X, y, cv=outer_cv, scoring='accuracy')
# 
# print(f"\nScores por fold externo: {nested_scores.round(4)}")
# print(f"Nested CV Accuracy: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
# 
# # Comparar con evaluación simple
# grid_simple = GridSearchCV(
#     RandomForestClassifier(random_state=42, n_jobs=-1),
#     param_grid_nested,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1
# )
# grid_simple.fit(X_train, y_train)
# 
# print(f"\nComparación:")
# print(f"  GridSearch CV simple: {grid_simple.best_score_:.4f}")
# print(f"  Nested CV:            {nested_scores.mean():.4f}")
# print("\n→ Nested CV da una estimación más honesta del rendimiento real")

print()

# ============================================
# PASO 6: Diferentes Scorers
# ============================================
print('--- Paso 6: Diferentes Scorers ---')

# Descomenta las siguientes líneas:
# from sklearn.metrics import make_scorer, f1_score
# 
# param_grid_scoring = {
#     'n_estimators': [50, 100],
#     'max_depth': [5, 10]
# }
# 
# scorers = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
# 
# print("Optimización con diferentes métricas:")
# print("-" * 55)
# 
# for scorer in scorers:
#     grid = GridSearchCV(
#         RandomForestClassifier(random_state=42, n_jobs=-1),
#         param_grid_scoring,
#         cv=5,
#         scoring=scorer,
#         n_jobs=-1
#     )
#     grid.fit(X_train, y_train)
#     
#     print(f"\nOptimizando para: {scorer}")
#     print(f"  Mejor config: n_estimators={grid.best_params_['n_estimators']}, "
#           f"max_depth={grid.best_params_['max_depth']}")
#     print(f"  Mejor {scorer} CV: {grid.best_score_:.4f}")
# 
# print("\n→ La métrica de optimización puede cambiar los hiperparámetros óptimos")

print()

# ============================================
# RESUMEN
# ============================================
print('--- Resumen ---')

# Descomenta las siguientes líneas:
# print("=" * 55)
# print("RESUMEN DE OPTIMIZACIÓN DE HIPERPARÁMETROS")
# print("=" * 55)
# print("\nMétodo               | Cuándo usar")
# print("-" * 55)
# print("GridSearchCV         | Pocos hiperparámetros, búsqueda exhaustiva")
# print("RandomizedSearchCV   | Muchos hiperparámetros, espacios grandes")
# print("Nested CV            | Evaluación honesta sin sesgo optimista")
# print("-" * 55)
# print("\nConsejos:")
# print("1. Empieza con RandomizedSearch para explorar")
# print("2. Refina con GridSearch en el área prometedora")
# print("3. Usa Nested CV para reportar métricas finales")
# print("4. Paraleliza con n_jobs=-1")
# print("=" * 55)

print()
print("¡Ejercicio completado!")
