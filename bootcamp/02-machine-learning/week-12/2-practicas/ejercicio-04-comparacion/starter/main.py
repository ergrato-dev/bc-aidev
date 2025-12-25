"""
Ejercicio 04: Comparación de Algoritmos
=======================================
Compara KNN, SVM y Naive Bayes en Wine dataset.
"""

# ============================================
# PASO 1: Cargar Dataset
# ============================================
print('--- Paso 1: Cargar Dataset ---')

# Descomenta las siguientes líneas:
# from sklearn.datasets import load_wine
#
# wine = load_wine()
# print(f"Dataset: Wine")
# print(f"Samples: {wine.data.shape[0]}")
# print(f"Features: {wine.data.shape[1]}")
# print(f"Clases: {list(wine.target_names)}")

print()

# ============================================
# PASO 2: Preparar Datos
# ============================================
print('--- Paso 2: Preparar Datos ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(
#     wine.data, wine.target,
#     test_size=0.2,
#     random_state=42,
#     stratify=wine.target
# )
#
# print(f"Train: {X_train.shape[0]} samples")
# print(f"Test: {X_test.shape[0]} samples")

print()

# ============================================
# PASO 3: Definir Modelos
# ============================================
print('--- Paso 3: Definir Modelos ---')

# Descomenta las siguientes líneas:
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
#
# models = {
#     'KNN': Pipeline([
#         ('scaler', StandardScaler()),
#         ('clf', KNeighborsClassifier(n_neighbors=5))
#     ]),
#     'SVM-RBF': Pipeline([
#         ('scaler', StandardScaler()),
#         ('clf', SVC(kernel='rbf', gamma='scale'))
#     ]),
#     'SVM-Linear': Pipeline([
#         ('scaler', StandardScaler()),
#         ('clf', SVC(kernel='linear'))
#     ]),
#     'Naive Bayes': GaussianNB()
# }
#
# print(f"Modelos a comparar: {list(models.keys())}")

print()

# ============================================
# PASO 4: Comparar con Cross-Validation
# ============================================
print('--- Paso 4: Comparar con Cross-Validation ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import cross_val_score
# import time
#
# results = {}
#
# print("Cross-Validation (5-fold):")
# print("-" * 50)
# for name, model in models.items():
#     start = time.time()
#     scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
#     train_time = time.time() - start
#
#     results[name] = {
#         'cv_mean': scores.mean(),
#         'cv_std': scores.std(),
#         'time': train_time
#     }
#
#     print(f"{name:12s}: {scores.mean():.4f} ± {scores.std():.4f} | {train_time:.4f}s")

print()

# ============================================
# PASO 5: Evaluación en Test
# ============================================
print('--- Paso 5: Evaluación en Test ---')

# Descomenta las siguientes líneas:
# from sklearn.metrics import accuracy_score
#
# print("Resultados en Test Set:")
# print("-" * 50)
#
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     test_acc = accuracy_score(y_test, y_pred)
#     results[name]['test_acc'] = test_acc
#     print(f"{name:12s}: {test_acc:.4f}")
#
# # Mejor modelo
# best_model = max(results.items(), key=lambda x: x[1]['test_acc'])
# print(f"\n🏆 Mejor modelo: {best_model[0]} (Test: {best_model[1]['test_acc']:.4f})")

print()

# ============================================
# PASO 6: Visualizar Comparación
# ============================================
print('--- Paso 6: Visualizar Comparación ---')

# Descomenta las siguientes líneas:
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # Crear DataFrame de resultados
# df_results = pd.DataFrame(results).T
# df_results = df_results.round(4)
# print("\nTabla de Resultados:")
# print(df_results)
#
# # Gráfico de barras
# names = list(results.keys())
# cv_scores = [results[n]['cv_mean'] for n in names]
# test_scores = [results[n]['test_acc'] for n in names]
#
# x = range(len(names))
# width = 0.35
#
# fig, ax = plt.subplots(figsize=(10, 6))
# bars1 = ax.bar([i - width/2 for i in x], cv_scores, width, label='CV Score', color='steelblue')
# bars2 = ax.bar([i + width/2 for i in x], test_scores, width, label='Test Score', color='darkorange')
#
# ax.set_ylabel('Accuracy', fontsize=12)
# ax.set_title('Comparación de Algoritmos: KNN vs SVM vs Naive Bayes', fontsize=14)
# ax.set_xticks(x)
# ax.set_xticklabels(names)
# ax.legend()
# ax.set_ylim(0.85, 1.02)
# ax.grid(axis='y', alpha=0.3)
#
# # Añadir valores sobre las barras
# for bar in bars1:
#     height = bar.get_height()
#     ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
#                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
# for bar in bars2:
#     height = bar.get_height()
#     ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
#                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
#
# plt.tight_layout()
# plt.savefig('algorithm_comparison.png', dpi=100, bbox_inches='tight')
# plt.show()
#
# print("\nGráfico guardado en: algorithm_comparison.png")

print()
print('=' * 50)
print('Ejercicio completado!')
