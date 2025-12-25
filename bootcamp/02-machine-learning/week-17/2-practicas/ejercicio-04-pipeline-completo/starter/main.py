"""
Ejercicio 04: Pipeline Completo
===============================
Pipeline de reducción dimensional + clasificación.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import time

# ============================================
# PASO 1: Cargar y Dividir Datos
# ============================================
print('--- Paso 1: Cargar Datos ---')

# Descomenta las siguientes líneas:
# digits = load_digits()
# X = digits.data
# y = digits.target
# 
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
# 
# print(f'Train: {X_train.shape}')
# print(f'Test: {X_test.shape}')
# print(f'Features: {X_train.shape[1]}')

print()


# ============================================
# PASO 2: Pipeline sin Reducción
# ============================================
print('--- Paso 2: Sin Reducción ---')

# Descomenta las siguientes líneas:
# # Pipeline solo con escalado y clasificador
# pipeline_no_red = Pipeline([
#     ('scaler', StandardScaler()),
#     ('classifier', SVC(kernel='rbf', random_state=42))
# ])
# 
# start = time.time()
# pipeline_no_red.fit(X_train, y_train)
# y_pred = pipeline_no_red.predict(X_test)
# elapsed = time.time() - start
# 
# acc_no_red = accuracy_score(y_test, y_pred)
# print(f'Sin reducción: Accuracy={acc_no_red:.4f}, Tiempo={elapsed:.2f}s')

print()


# ============================================
# PASO 3: Pipeline con PCA
# ============================================
print('--- Paso 3: Con PCA ---')

# Descomenta las siguientes líneas:
# # Probar diferentes números de componentes
# n_components_list = [10, 20, 30, 40, 50]
# results = []
# 
# for n_comp in n_components_list:
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('pca', PCA(n_components=n_comp)),
#         ('classifier', SVC(kernel='rbf', random_state=42))
#     ])
#     
#     start = time.time()
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     elapsed = time.time() - start
#     
#     acc = accuracy_score(y_test, y_pred)
#     results.append({'n_components': n_comp, 'accuracy': acc, 'time': elapsed})
#     
#     print(f'PCA({n_comp}): Accuracy={acc:.4f}, Tiempo={elapsed:.2f}s')
# 
# # Mejor resultado
# best = max(results, key=lambda x: x['accuracy'])
# print(f'\nMejor: PCA({best["n_components"]}) con {best["accuracy"]:.4f}')

print()


# ============================================
# PASO 4: PCA por Varianza
# ============================================
print('--- Paso 4: PCA por Varianza ---')

# Descomenta las siguientes líneas:
# variance_thresholds = [0.80, 0.90, 0.95, 0.99]
# 
# for var in variance_thresholds:
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('pca', PCA(n_components=var)),
#         ('classifier', SVC(kernel='rbf', random_state=42))
#     ])
#     
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     
#     acc = accuracy_score(y_test, y_pred)
#     n_comp = pipeline.named_steps['pca'].n_components_
#     
#     print(f'Varianza {var*100:.0f}%: {n_comp} componentes, Accuracy={acc:.4f}')

print()


# ============================================
# PASO 5: Comparar Clasificadores
# ============================================
print('--- Paso 5: Diferentes Clasificadores ---')

# Descomenta las siguientes líneas:
# classifiers = {
#     'SVM': SVC(kernel='rbf', random_state=42),
#     'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#     'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
# }
# 
# print('=== Sin Reducción ===')
# for name, clf in classifiers.items():
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('classifier', clf)
#     ])
#     scores = cross_val_score(pipeline, X_train, y_train, cv=5)
#     print(f'{name}: {scores.mean():.4f} ± {scores.std():.4f}')
# 
# print('\n=== Con PCA (95% varianza) ===')
# for name, clf in classifiers.items():
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('pca', PCA(n_components=0.95)),
#         ('classifier', clf)
#     ])
#     scores = cross_val_score(pipeline, X_train, y_train, cv=5)
#     print(f'{name}: {scores.mean():.4f} ± {scores.std():.4f}')

print()


# ============================================
# PASO 6: GridSearchCV para Optimizar
# ============================================
print('--- Paso 6: GridSearchCV ---')

# Descomenta las siguientes líneas:
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('pca', PCA()),
#     ('classifier', SVC(random_state=42))
# ])
# 
# param_grid = {
#     'pca__n_components': [20, 30, 40],
#     'classifier__kernel': ['rbf', 'linear'],
#     'classifier__C': [0.1, 1, 10]
# }
# 
# grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
# grid_search.fit(X_train, y_train)
# 
# print(f'Mejores parámetros: {grid_search.best_params_}')
# print(f'Mejor score CV: {grid_search.best_score_:.4f}')
# 
# # Evaluar en test
# y_pred = grid_search.predict(X_test)
# print(f'Score en test: {accuracy_score(y_test, y_pred):.4f}')

print()


# ============================================
# PASO 7: Visualizar Impacto de Reducción
# ============================================
print('--- Paso 7: Visualizar Impacto ---')

# Descomenta las siguientes líneas:
# # Gráfico de accuracy vs n_components
# n_components_range = range(5, 65, 5)
# accuracies = []
# 
# for n_comp in n_components_range:
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('pca', PCA(n_components=n_comp)),
#         ('classifier', SVC(kernel='rbf', random_state=42))
#     ])
#     pipeline.fit(X_train, y_train)
#     acc = accuracy_score(y_test, pipeline.predict(X_test))
#     accuracies.append(acc)
# 
# plt.figure(figsize=(10, 6))
# plt.plot(list(n_components_range), accuracies, 'bo-', linewidth=2, markersize=8)
# plt.axhline(y=acc_no_red, color='r', linestyle='--', label=f'Sin reducción ({acc_no_red:.3f})')
# plt.xlabel('Número de Componentes PCA')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs Número de Componentes')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()
# 
# # Mejor punto
# best_n = list(n_components_range)[np.argmax(accuracies)]
# print(f'Mejor n_components: {best_n} con accuracy {max(accuracies):.4f}')

print()


# ============================================
# PASO 8: Análisis de Tiempo de Entrenamiento
# ============================================
print('--- Paso 8: Benchmark de Tiempo ---')

# Descomenta las siguientes líneas:
# n_components_range = [10, 20, 30, 40, 50, None]  # None = sin PCA
# times_train = []
# times_predict = []
# 
# for n_comp in n_components_range:
#     if n_comp is None:
#         pipeline = Pipeline([
#             ('scaler', StandardScaler()),
#             ('classifier', SVC(kernel='rbf', random_state=42))
#         ])
#     else:
#         pipeline = Pipeline([
#             ('scaler', StandardScaler()),
#             ('pca', PCA(n_components=n_comp)),
#             ('classifier', SVC(kernel='rbf', random_state=42))
#         ])
#     
#     # Tiempo de entrenamiento
#     start = time.time()
#     pipeline.fit(X_train, y_train)
#     train_time = time.time() - start
#     
#     # Tiempo de predicción
#     start = time.time()
#     _ = pipeline.predict(X_test)
#     pred_time = time.time() - start
#     
#     times_train.append(train_time)
#     times_predict.append(pred_time)
#     
#     label = f'PCA({n_comp})' if n_comp else 'Sin PCA'
#     print(f'{label}: Train={train_time:.3f}s, Predict={pred_time:.3f}s')
# 
# # Gráfico
# labels = [f'PCA({n})' if n else 'Sin PCA' for n in n_components_range]
# x = np.arange(len(labels))
# 
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.bar(x - 0.2, times_train, 0.4, label='Train')
# ax.bar(x + 0.2, times_predict, 0.4, label='Predict')
# ax.set_ylabel('Tiempo (s)')
# ax.set_title('Tiempo de Entrenamiento y Predicción')
# ax.set_xticks(x)
# ax.set_xticklabels(labels, rotation=45)
# ax.legend()
# plt.tight_layout()
# plt.show()

print()
print('=== Ejercicio completado ===')
