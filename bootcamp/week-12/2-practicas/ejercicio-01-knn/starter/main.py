"""
Ejercicio 01: K-Nearest Neighbors con Iris
==========================================
Implementa KNN para clasificación del dataset Iris.
"""

# ============================================
# PASO 1: Cargar y Explorar Datos
# ============================================
print('--- Paso 1: Cargar y Explorar Datos ---')

# Descomenta las siguientes líneas:
# from sklearn.datasets import load_iris
# import pandas as pd
#
# iris = load_iris()
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# df['target'] = iris.target
# print(df.head())
# print(f"\nDistribución de clases:")
# print(df['target'].value_counts())
# print(f"\nClases: {iris.target_names}")

print()

# ============================================
# PASO 2: Dividir y Normalizar
# ============================================
print('--- Paso 2: Dividir y Normalizar ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# X_train, X_test, y_train, y_test = train_test_split(
#     iris.data, iris.target,
#     test_size=0.2,
#     random_state=42,
#     stratify=iris.target
# )
#
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# print(f"Train: {X_train_scaled.shape}")
# print(f"Test: {X_test_scaled.shape}")
# print(f"Media post-scaling: {X_train_scaled.mean(axis=0).round(2)}")

print()

# ============================================
# PASO 3: KNN Básico
# ============================================
print('--- Paso 3: KNN Básico ---')

# Descomenta las siguientes líneas:
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
#
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train_scaled, y_train)
# y_pred = knn.predict(X_test_scaled)
#
# print(f"k=5 Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print()

# ============================================
# PASO 4: Encontrar k Óptimo
# ============================================
print('--- Paso 4: Encontrar k Óptimo ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import cross_val_score
# import matplotlib.pyplot as plt
#
# k_range = range(1, 21)
# k_scores = []
#
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
#     k_scores.append(scores.mean())
#
# # Visualizar
# plt.figure(figsize=(10, 6))
# plt.plot(k_range, k_scores, 'b-o', linewidth=2, markersize=8)
# plt.xlabel('Valor de k', fontsize=12)
# plt.ylabel('CV Accuracy', fontsize=12)
# plt.title('Elección de k óptimo para KNN', fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.xticks(k_range)
# plt.savefig('k_optimization.png', dpi=100, bbox_inches='tight')
# plt.show()
#
# best_k = list(k_range)[k_scores.index(max(k_scores))]
# print(f"Mejor k: {best_k}")
# print(f"Mejor CV Accuracy: {max(k_scores):.4f}")

print()

# ============================================
# PASO 5: Comparar Métricas de Distancia
# ============================================
print('--- Paso 5: Comparar Métricas de Distancia ---')

# Descomenta las siguientes líneas:
# metrics = ['euclidean', 'manhattan', 'chebyshev']
#
# print("Comparación de métricas de distancia:")
# for metric in metrics:
#     knn = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
#     scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
#     print(f"  {metric:12s}: {scores.mean():.4f} ± {scores.std():.4f}")

print()

# ============================================
# PASO 6: Modelo Final con Pipeline
# ============================================
print('--- Paso 6: Modelo Final con Pipeline ---')

# Descomenta las siguientes líneas:
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report
#
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('knn', KNeighborsClassifier(n_neighbors=best_k, weights='distance'))
# ])
#
# # Entrenar con datos originales (pipeline normaliza internamente)
# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
#
# print(f"Test Accuracy: {pipeline.score(X_test, y_test):.4f}")
# print(f"\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=iris.target_names))

print()
print('=' * 50)
print('Ejercicio completado!')
