# ============================================
# EJERCICIO 01: K-Means desde Cero
# ============================================
# Implementación paso a paso del algoritmo K-Means
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("EJERCICIO 01: K-Means desde Cero")
print("=" * 60)

# ============================================
# PASO 1: Generar Datos de Prueba
# ============================================
print("\n--- Paso 1: Generar Datos de Prueba ---")

# Descomenta las siguientes líneas:
# np.random.seed(42)
# X, y_true = make_blobs(n_samples=300, centers=4,
#                        cluster_std=0.6, random_state=42)
# print(f"Datos generados: {X.shape}")
# print(f"Clusters reales: {np.unique(y_true)}")

print()

# ============================================
# PASO 2: Implementar Cálculo de Distancia
# ============================================
print("--- Paso 2: Cálculo de Distancia Euclidiana ---")

# Descomenta las siguientes líneas:
# def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
#     """
#     Calculate Euclidean distance between two points.
#     
#     Args:
#         x1: First point
#         x2: Second point
#     
#     Returns:
#         Euclidean distance
#     """
#     return np.sqrt(np.sum((x1 - x2) ** 2))
#
# # Probar la función
# p1 = np.array([0, 0])
# p2 = np.array([3, 4])
# dist = euclidean_distance(p1, p2)
# print(f"Distancia entre {p1} y {p2}: {dist}")  # Debe ser 5.0

print()

# ============================================
# PASO 3: Inicialización de Centroides
# ============================================
print("--- Paso 3: Inicialización de Centroides ---")

# Descomenta las siguientes líneas:
# def initialize_centroids(X: np.ndarray, k: int) -> np.ndarray:
#     """
#     Randomly select k points as initial centroids.
#     
#     Args:
#         X: Data matrix
#         k: Number of clusters
#     
#     Returns:
#         Initial centroids
#     """
#     indices = np.random.choice(X.shape[0], k, replace=False)
#     return X[indices].copy()
#
# # Probar inicialización
# k = 4
# initial_centroids = initialize_centroids(X, k)
# print(f"Centroides iniciales (shape): {initial_centroids.shape}")
# print(f"Primer centroide: {initial_centroids[0]}")

print()

# ============================================
# PASO 4: Asignación de Clusters
# ============================================
print("--- Paso 4: Asignación de Clusters ---")

# Descomenta las siguientes líneas:
# def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
#     """
#     Assign each point to the nearest centroid.
#     
#     Args:
#         X: Data matrix
#         centroids: Current centroids
#     
#     Returns:
#         Cluster labels for each point
#     """
#     n_samples = X.shape[0]
#     labels = np.zeros(n_samples, dtype=int)
#     
#     for i in range(n_samples):
#         # Calculate distance to each centroid
#         distances = np.array([euclidean_distance(X[i], c) for c in centroids])
#         # Assign to nearest centroid
#         labels[i] = np.argmin(distances)
#     
#     return labels
#
# # Probar asignación
# labels = assign_clusters(X, initial_centroids)
# print(f"Labels shape: {labels.shape}")
# print(f"Distribución inicial: {np.bincount(labels)}")

print()

# ============================================
# PASO 5: Actualización de Centroides
# ============================================
print("--- Paso 5: Actualización de Centroides ---")

# Descomenta las siguientes líneas:
# def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
#     """
#     Update centroids as mean of assigned points.
#     
#     Args:
#         X: Data matrix
#         labels: Current cluster assignments
#         k: Number of clusters
#     
#     Returns:
#         Updated centroids
#     """
#     n_features = X.shape[1]
#     centroids = np.zeros((k, n_features))
#     
#     for i in range(k):
#         cluster_points = X[labels == i]
#         if len(cluster_points) > 0:
#             centroids[i] = cluster_points.mean(axis=0)
#         else:
#             # Handle empty cluster: reinitialize randomly
#             centroids[i] = X[np.random.randint(X.shape[0])]
#     
#     return centroids
#
# # Probar actualización
# new_centroids = update_centroids(X, labels, k)
# print(f"Centroides actualizados: {new_centroids.shape}")

print()

# ============================================
# PASO 6: Calcular Inercia (WCSS)
# ============================================
print("--- Paso 6: Calcular Inercia ---")

# Descomenta las siguientes líneas:
# def calculate_inertia(X: np.ndarray, centroids: np.ndarray, 
#                       labels: np.ndarray) -> float:
#     """
#     Calculate Within-Cluster Sum of Squares (WCSS).
#     
#     Args:
#         X: Data matrix
#         centroids: Cluster centroids
#         labels: Cluster assignments
#     
#     Returns:
#         Total inertia
#     """
#     inertia = 0.0
#     for i in range(len(centroids)):
#         cluster_points = X[labels == i]
#         if len(cluster_points) > 0:
#             inertia += np.sum((cluster_points - centroids[i]) ** 2)
#     return inertia
#
# # Probar cálculo
# inertia = calculate_inertia(X, new_centroids, labels)
# print(f"Inercia inicial: {inertia:.2f}")

print()

# ============================================
# PASO 7: Algoritmo K-Means Completo
# ============================================
print("--- Paso 7: K-Means Completo ---")

# Descomenta las siguientes líneas:
# def kmeans_scratch(X: np.ndarray, k: int, max_iters: int = 100, 
#                    tol: float = 1e-4, verbose: bool = True) -> tuple:
#     """
#     K-Means clustering implementation from scratch.
#     
#     Args:
#         X: Data matrix
#         k: Number of clusters
#         max_iters: Maximum iterations
#         tol: Convergence tolerance
#         verbose: Print progress
#     
#     Returns:
#         centroids: Final centroids
#         labels: Cluster assignments
#         history: Inertia history
#     """
#     # Initialize
#     centroids = initialize_centroids(X, k)
#     history = []
#     
#     for iteration in range(max_iters):
#         # Assign clusters
#         labels = assign_clusters(X, centroids)
#         
#         # Update centroids
#         new_centroids = update_centroids(X, labels, k)
#         
#         # Calculate inertia
#         inertia = calculate_inertia(X, new_centroids, labels)
#         history.append(inertia)
#         
#         # Check convergence
#         shift = np.sum((new_centroids - centroids) ** 2)
#         
#         if verbose and iteration % 10 == 0:
#             print(f"  Iteración {iteration}: Inercia = {inertia:.2f}, Shift = {shift:.6f}")
#         
#         if shift < tol:
#             if verbose:
#                 print(f"  ✓ Convergencia en iteración {iteration}")
#             break
#         
#         centroids = new_centroids
#     
#     return centroids, labels, history
#
# # Ejecutar K-Means
# print("\nEjecutando K-Means (k=4):")
# centroids, labels, history = kmeans_scratch(X, k=4)
# print(f"\nResultados:")
# print(f"  - Clusters encontrados: {len(np.unique(labels))}")
# print(f"  - Distribución: {np.bincount(labels)}")
# print(f"  - Inercia final: {history[-1]:.2f}")

print()

# ============================================
# PASO 8: Método del Codo
# ============================================
print("--- Paso 8: Método del Codo ---")

# Descomenta las siguientes líneas:
# def elbow_method(X: np.ndarray, k_range: range) -> list:
#     """
#     Apply elbow method to find optimal K.
#     
#     Args:
#         X: Data matrix
#         k_range: Range of K values to try
#     
#     Returns:
#         List of inertias for each K
#     """
#     inertias = []
#     print("Calculando inercia para diferentes K:")
#     
#     for k in k_range:
#         centroids, labels, _ = kmeans_scratch(X, k, verbose=False)
#         inertia = calculate_inertia(X, centroids, labels)
#         inertias.append(inertia)
#         print(f"  K={k}: Inercia = {inertia:.2f}")
#     
#     return inertias
#
# # Calcular inercias
# k_range = range(1, 10)
# inertias = elbow_method(X, k_range)
#
# # Graficar método del codo
# plt.figure(figsize=(10, 6))
# plt.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
# plt.xlabel('Número de Clusters (K)', fontsize=12)
# plt.ylabel('Inercia (WCSS)', fontsize=12)
# plt.title('Método del Codo para Selección de K', fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.xticks(list(k_range))
# plt.tight_layout()
# plt.savefig('elbow_method.png', dpi=150)
# plt.show()
# print("\n✓ Gráfico guardado: elbow_method.png")

print()

# ============================================
# PASO 9: Visualización de Clusters
# ============================================
print("--- Paso 9: Visualización de Clusters ---")

# Descomenta las siguientes líneas:
# def visualize_kmeans(X: np.ndarray, labels: np.ndarray, 
#                      centroids: np.ndarray, title: str) -> None:
#     """
#     Visualize K-Means clustering results.
#     
#     Args:
#         X: Data matrix (2D)
#         labels: Cluster assignments
#         centroids: Cluster centroids
#         title: Plot title
#     """
#     plt.figure(figsize=(10, 8))
#     
#     # Plot points
#     scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, 
#                          cmap='viridis', alpha=0.6, s=50)
#     
#     # Plot centroids
#     plt.scatter(centroids[:, 0], centroids[:, 1], 
#                c='red', marker='X', s=200, edgecolors='black',
#                linewidths=2, label='Centroides')
#     
#     plt.colorbar(scatter, label='Cluster')
#     plt.xlabel('Feature 1', fontsize=12)
#     plt.ylabel('Feature 2', fontsize=12)
#     plt.title(title, fontsize=14)
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#
# # Visualizar resultados
# visualize_kmeans(X, labels, centroids, 
#                  'K-Means desde Cero (K=4)')
# plt.savefig('kmeans_scratch.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: kmeans_scratch.png")

print()

# ============================================
# PASO 10: Comparación con Scikit-learn
# ============================================
print("--- Paso 10: Comparación con Scikit-learn ---")

# Descomenta las siguientes líneas:
# # K-Means de sklearn
# kmeans_sklearn = KMeans(n_clusters=4, random_state=42, n_init=10)
# labels_sklearn = kmeans_sklearn.fit_predict(X)
# centroids_sklearn = kmeans_sklearn.cluster_centers_
# inertia_sklearn = kmeans_sklearn.inertia_
#
# print("\nComparación de resultados:")
# print(f"{'Métrica':<25} {'Scratch':<15} {'Sklearn':<15}")
# print("-" * 55)
# print(f"{'Inercia':<25} {history[-1]:<15.2f} {inertia_sklearn:<15.2f}")
# print(f"{'Clusters':<25} {len(np.unique(labels)):<15} {len(np.unique(labels_sklearn)):<15}")
#
# # Visualizar comparación
# fig, axes = plt.subplots(1, 2, figsize=(16, 6))
#
# # Scratch
# axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
# axes[0].scatter(centroids[:, 0], centroids[:, 1], 
#                c='red', marker='X', s=200, edgecolors='black')
# axes[0].set_title(f'K-Means Scratch\nInercia: {history[-1]:.2f}')
# axes[0].set_xlabel('Feature 1')
# axes[0].set_ylabel('Feature 2')
#
# # Sklearn
# axes[1].scatter(X[:, 0], X[:, 1], c=labels_sklearn, cmap='viridis', alpha=0.6)
# axes[1].scatter(centroids_sklearn[:, 0], centroids_sklearn[:, 1], 
#                c='red', marker='X', s=200, edgecolors='black')
# axes[1].set_title(f'K-Means Sklearn\nInercia: {inertia_sklearn:.2f}')
# axes[1].set_xlabel('Feature 1')
# axes[1].set_ylabel('Feature 2')
#
# plt.tight_layout()
# plt.savefig('kmeans_comparison.png', dpi=150)
# plt.show()
# print("\n✓ Gráfico guardado: kmeans_comparison.png")

print()

# ============================================
# PASO 11: Evolución de Clusters
# ============================================
print("--- Paso 11: Evolución de Clusters (Bonus) ---")

# Descomenta las siguientes líneas:
# def kmeans_with_visualization(X: np.ndarray, k: int, 
#                               max_iters: int = 10) -> None:
#     """
#     K-Means with step-by-step visualization.
#     """
#     centroids = initialize_centroids(X, k)
#     
#     fig, axes = plt.subplots(2, 5, figsize=(20, 8))
#     axes = axes.flatten()
#     
#     for iteration in range(min(max_iters, 10)):
#         labels = assign_clusters(X, centroids)
#         
#         # Plot current state
#         axes[iteration].scatter(X[:, 0], X[:, 1], c=labels, 
#                                cmap='viridis', alpha=0.6, s=30)
#         axes[iteration].scatter(centroids[:, 0], centroids[:, 1], 
#                                c='red', marker='X', s=150, edgecolors='black')
#         axes[iteration].set_title(f'Iteración {iteration}')
#         axes[iteration].axis('off')
#         
#         # Update centroids
#         centroids = update_centroids(X, labels, k)
#     
#     plt.suptitle('Evolución del Algoritmo K-Means', fontsize=16)
#     plt.tight_layout()
#     plt.savefig('kmeans_evolution.png', dpi=150)
#     plt.show()
#     print("✓ Gráfico guardado: kmeans_evolution.png")
#
# # Ejecutar visualización de evolución
# np.random.seed(42)
# kmeans_with_visualization(X, k=4)

print()

# ============================================
# RESUMEN
# ============================================
print("=" * 60)
print("RESUMEN DEL EJERCICIO")
print("=" * 60)
print("""
En este ejercicio aprendiste a:

1. ✓ Implementar K-Means desde cero
2. ✓ Calcular distancias euclidianas
3. ✓ Inicializar centroides aleatoriamente
4. ✓ Asignar puntos al cluster más cercano
5. ✓ Actualizar centroides como media del cluster
6. ✓ Calcular inercia (WCSS)
7. ✓ Aplicar el método del codo
8. ✓ Visualizar clusters y centroides
9. ✓ Comparar con scikit-learn

Archivos generados:
- elbow_method.png
- kmeans_scratch.png
- kmeans_comparison.png
- kmeans_evolution.png
""")
