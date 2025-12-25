# ============================================
# EJERCICIO 02: DBSCAN - Clustering por Densidad
# ============================================
# Implementación paso a paso de DBSCAN
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("EJERCICIO 02: DBSCAN desde Cero")
print("=" * 60)

# ============================================
# PASO 1: Generar Datos con Formas Complejas
# ============================================
print("\n--- Paso 1: Generar Datos Complejos ---")

# Descomenta las siguientes líneas:
# np.random.seed(42)
#
# # Datos en forma de lunas (moons)
# X_moons, y_moons = make_moons(n_samples=300, noise=0.08, random_state=42)
# print(f"Datos moons: {X_moons.shape}")
#
# # Datos en forma de círculos concéntricos
# X_circles, y_circles = make_circles(n_samples=300, noise=0.05, 
#                                     factor=0.5, random_state=42)
# print(f"Datos circles: {X_circles.shape}")
#
# # Datos con outliers
# X_blobs, _ = make_blobs(n_samples=250, centers=3, 
#                         cluster_std=0.5, random_state=42)
# outliers = np.random.uniform(low=-10, high=10, size=(50, 2))
# X_outliers = np.vstack([X_blobs, outliers])
# print(f"Datos con outliers: {X_outliers.shape}")
#
# # Visualizar los tres datasets
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c='blue', alpha=0.6)
# axes[0].set_title('Moons Dataset')
# axes[1].scatter(X_circles[:, 0], X_circles[:, 1], c='blue', alpha=0.6)
# axes[1].set_title('Circles Dataset')
# axes[2].scatter(X_outliers[:, 0], X_outliers[:, 1], c='blue', alpha=0.6)
# axes[2].set_title('Blobs con Outliers')
# plt.tight_layout()
# plt.savefig('datasets_complex.png', dpi=150)
# plt.show()
# print("\n✓ Gráfico guardado: datasets_complex.png")

print()

# ============================================
# PASO 2: Implementar Cálculo de Distancia
# ============================================
print("--- Paso 2: Funciones de Distancia ---")

# Descomenta las siguientes líneas:
# def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
#     """Calculate Euclidean distance between two points."""
#     return np.sqrt(np.sum((x1 - x2) ** 2))
#
# def euclidean_distance_vectorized(X: np.ndarray, point: np.ndarray) -> np.ndarray:
#     """Calculate distances from all points in X to a single point."""
#     return np.sqrt(np.sum((X - point) ** 2, axis=1))
#
# # Probar funciones
# p1 = np.array([0, 0])
# p2 = np.array([3, 4])
# print(f"Distancia escalar: {euclidean_distance(p1, p2):.2f}")
# print(f"Distancia vectorizada: {euclidean_distance_vectorized(np.array([p2]), p1)[0]:.2f}")

print()

# ============================================
# PASO 3: Implementar Region Query
# ============================================
print("--- Paso 3: Region Query (Vecindario) ---")

# Descomenta las siguientes líneas:
# def region_query(X: np.ndarray, point_idx: int, epsilon: float) -> list:
#     """
#     Find all points within epsilon distance of a point.
#     
#     Args:
#         X: Data matrix
#         point_idx: Index of the query point
#         epsilon: Neighborhood radius
#     
#     Returns:
#         List of indices of neighboring points
#     """
#     # Versión vectorizada (más eficiente)
#     distances = euclidean_distance_vectorized(X, X[point_idx])
#     neighbors = np.where(distances <= epsilon)[0].tolist()
#     return neighbors
#
# # Probar region query
# X = X_moons.copy()
# epsilon = 0.3
# test_point = 0
# neighbors = region_query(X, test_point, epsilon)
# print(f"Punto {test_point}: {len(neighbors)} vecinos dentro de ε={epsilon}")
# print(f"Primeros vecinos: {neighbors[:5]}")

print()

# ============================================
# PASO 4: Clasificar Tipos de Puntos
# ============================================
print("--- Paso 4: Core, Border, Noise ---")

# Descomenta las siguientes líneas:
# def classify_points(X: np.ndarray, epsilon: float, 
#                     min_samples: int) -> tuple:
#     """
#     Classify each point as core, border, or noise.
#     
#     Returns:
#         core_points: indices of core points
#         border_points: indices of border points
#         noise_points: indices of noise points
#     """
#     n_samples = X.shape[0]
#     neighbor_counts = np.zeros(n_samples, dtype=int)
#     neighbors_list = []
#     
#     # Count neighbors for each point
#     for i in range(n_samples):
#         neighbors = region_query(X, i, epsilon)
#         neighbor_counts[i] = len(neighbors)
#         neighbors_list.append(neighbors)
#     
#     # Classify points
#     is_core = neighbor_counts >= min_samples
#     core_points = np.where(is_core)[0]
#     
#     # Find border points (not core, but neighbor of core)
#     border_points = []
#     for i in range(n_samples):
#         if not is_core[i]:
#             for neighbor in neighbors_list[i]:
#                 if is_core[neighbor]:
#                     border_points.append(i)
#                     break
#     border_points = np.array(border_points)
#     
#     # Noise points (neither core nor border)
#     is_border = np.isin(np.arange(n_samples), border_points)
#     noise_points = np.where(~is_core & ~is_border)[0]
#     
#     return core_points, border_points, noise_points, neighbors_list
#
# # Clasificar puntos
# epsilon = 0.3
# min_samples = 5
# core, border, noise, _ = classify_points(X, epsilon, min_samples)
# print(f"Core points: {len(core)}")
# print(f"Border points: {len(border)}")
# print(f"Noise points: {len(noise)}")
#
# # Visualizar clasificación
# plt.figure(figsize=(10, 8))
# plt.scatter(X[core, 0], X[core, 1], c='green', s=100, 
#            label=f'Core ({len(core)})', alpha=0.7)
# plt.scatter(X[border, 0], X[border, 1], c='yellow', s=60, 
#            label=f'Border ({len(border)})', alpha=0.7)
# plt.scatter(X[noise, 0], X[noise, 1], c='red', s=40, marker='x',
#            label=f'Noise ({len(noise)})', alpha=0.7)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title(f'Clasificación de Puntos (ε={epsilon}, min_samples={min_samples})')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.savefig('point_classification.png', dpi=150)
# plt.show()
# print("\n✓ Gráfico guardado: point_classification.png")

print()

# ============================================
# PASO 5: Expandir Cluster
# ============================================
print("--- Paso 5: Expandir Cluster ---")

# Descomenta las siguientes líneas:
# def expand_cluster(X: np.ndarray, labels: np.ndarray, point_idx: int,
#                    neighbors: list, cluster_id: int, epsilon: float,
#                    min_samples: int) -> None:
#     """
#     Expand cluster by adding all density-reachable points.
#     
#     Args:
#         X: Data matrix
#         labels: Array of cluster labels (modified in place)
#         point_idx: Starting core point index
#         neighbors: Initial neighbors of the point
#         cluster_id: Current cluster ID
#         epsilon: Neighborhood radius
#         min_samples: Minimum neighbors to be core
#     """
#     # Assign starting point to cluster
#     labels[point_idx] = cluster_id
#     
#     # Process neighbors iteratively
#     i = 0
#     while i < len(neighbors):
#         neighbor_idx = neighbors[i]
#         
#         if labels[neighbor_idx] == -1:
#             # Was marked as noise, but is reachable from core
#             labels[neighbor_idx] = cluster_id
#         
#         elif labels[neighbor_idx] == 0:
#             # Unvisited point
#             labels[neighbor_idx] = cluster_id
#             
#             # Check if this neighbor is also a core point
#             neighbor_neighbors = region_query(X, neighbor_idx, epsilon)
#             if len(neighbor_neighbors) >= min_samples:
#                 # Add new neighbors to process
#                 neighbors.extend(neighbor_neighbors)
#         
#         i += 1
#
# print("Función expand_cluster definida correctamente.")

print()

# ============================================
# PASO 6: Algoritmo DBSCAN Completo
# ============================================
print("--- Paso 6: DBSCAN Completo ---")

# Descomenta las siguientes líneas:
# def dbscan_scratch(X: np.ndarray, epsilon: float, 
#                    min_samples: int, verbose: bool = True) -> np.ndarray:
#     """
#     DBSCAN clustering algorithm from scratch.
#     
#     Args:
#         X: Data matrix
#         epsilon: Neighborhood radius
#         min_samples: Minimum neighbors to be core point
#         verbose: Print progress
#     
#     Returns:
#         labels: Cluster labels (-1 for noise)
#     
#     Label meanings:
#         -1: Noise
#          0: Unvisited
#         >0: Cluster ID
#     """
#     n_samples = X.shape[0]
#     labels = np.zeros(n_samples, dtype=int)  # 0 = unvisited
#     cluster_id = 0
#     
#     for point_idx in range(n_samples):
#         if labels[point_idx] != 0:
#             continue  # Already processed
#         
#         # Find neighbors
#         neighbors = region_query(X, point_idx, epsilon)
#         
#         if len(neighbors) < min_samples:
#             # Not enough neighbors - mark as noise (for now)
#             labels[point_idx] = -1
#         else:
#             # Core point - start new cluster
#             cluster_id += 1
#             expand_cluster(X, labels, point_idx, neighbors,
#                           cluster_id, epsilon, min_samples)
#             
#             if verbose:
#                 points_in_cluster = np.sum(labels == cluster_id)
#                 print(f"  Cluster {cluster_id}: {points_in_cluster} puntos")
#     
#     return labels
#
# # Ejecutar DBSCAN
# print("\nEjecutando DBSCAN (ε=0.3, min_samples=5):")
# labels = dbscan_scratch(X_moons, epsilon=0.3, min_samples=5)
#
# # Estadísticas
# unique_labels = np.unique(labels)
# n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
# n_noise = np.sum(labels == -1)
# print(f"\nResultados:")
# print(f"  - Clusters encontrados: {n_clusters}")
# print(f"  - Puntos de ruido: {n_noise}")

print()

# ============================================
# PASO 7: Selección de Epsilon (K-Distance Graph)
# ============================================
print("--- Paso 7: K-Distance Graph ---")

# Descomenta las siguientes líneas:
# def k_distance_graph(X: np.ndarray, min_samples: int) -> np.ndarray:
#     """
#     Plot k-distance graph to find optimal epsilon.
#     
#     The "elbow" in this graph suggests a good epsilon value.
#     
#     Args:
#         X: Data matrix
#         min_samples: k value for k-distance
#     
#     Returns:
#         k_distances: Sorted k-distances
#     """
#     # Find k nearest neighbors
#     neigh = NearestNeighbors(n_neighbors=min_samples)
#     neigh.fit(X)
#     distances, _ = neigh.kneighbors(X)
#     
#     # Sort distances to k-th neighbor (ascending)
#     k_distances = np.sort(distances[:, -1])
#     
#     # Plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(k_distances)), k_distances, 'b-', linewidth=2)
#     plt.xlabel('Puntos (ordenados por distancia)', fontsize=12)
#     plt.ylabel(f'{min_samples}-distancia', fontsize=12)
#     plt.title(f'K-Distance Graph (k={min_samples})\n'
#              'El "codo" indica el epsilon óptimo', fontsize=14)
#     plt.grid(True, alpha=0.3)
#     
#     # Add annotation for elbow region
#     plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.7, 
#                label='Posible ε ≈ 0.2')
#     plt.legend()
#     
#     plt.tight_layout()
#     plt.savefig('k_distance_graph.png', dpi=150)
#     plt.show()
#     print("\n✓ Gráfico guardado: k_distance_graph.png")
#     
#     return k_distances
#
# # Generar k-distance graph
# k_distances = k_distance_graph(X_moons, min_samples=5)
# print(f"\nRango de k-distancias: [{k_distances.min():.3f}, {k_distances.max():.3f}]")
# print(f"Mediana: {np.median(k_distances):.3f}")

print()

# ============================================
# PASO 8: Comparación DBSCAN vs K-Means
# ============================================
print("--- Paso 8: DBSCAN vs K-Means ---")

# Descomenta las siguientes líneas:
# def compare_algorithms(X: np.ndarray, title: str, 
#                        epsilon: float = 0.2, min_samples: int = 5) -> None:
#     """
#     Compare DBSCAN and K-Means on the same dataset.
#     """
#     # DBSCAN
#     labels_dbscan = dbscan_scratch(X, epsilon, min_samples, verbose=False)
#     n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
#     
#     # K-Means (usando el número de clusters encontrado por DBSCAN)
#     if n_clusters_dbscan > 0:
#         kmeans = KMeans(n_clusters=max(2, n_clusters_dbscan), random_state=42)
#         labels_kmeans = kmeans.fit_predict(X)
#     else:
#         labels_kmeans = np.zeros(len(X))
#     
#     # Visualizar
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#     
#     # DBSCAN
#     colors_dbscan = labels_dbscan.copy().astype(float)
#     colors_dbscan[labels_dbscan == -1] = -1  # Ruido en color especial
#     scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=colors_dbscan, 
#                               cmap='viridis', alpha=0.6, s=50)
#     # Marcar ruido
#     noise_mask = labels_dbscan == -1
#     if np.any(noise_mask):
#         axes[0].scatter(X[noise_mask, 0], X[noise_mask, 1], 
#                        c='red', marker='x', s=50, label='Ruido')
#     axes[0].set_title(f'DBSCAN (ε={epsilon}, min={min_samples})\n'
#                      f'Clusters: {n_clusters_dbscan}, Ruido: {np.sum(noise_mask)}')
#     axes[0].legend()
#     
#     # K-Means
#     axes[1].scatter(X[:, 0], X[:, 1], c=labels_kmeans, 
#                    cmap='viridis', alpha=0.6, s=50)
#     axes[1].set_title(f'K-Means (K={max(2, n_clusters_dbscan)})\n'
#                      f'Clusters: {len(np.unique(labels_kmeans))}')
#     
#     for ax in axes:
#         ax.set_xlabel('Feature 1')
#         ax.set_ylabel('Feature 2')
#         ax.grid(True, alpha=0.3)
#     
#     plt.suptitle(title, fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(f'comparison_{title.lower().replace(" ", "_")}.png', dpi=150)
#     plt.show()
#
# # Comparar en diferentes datasets
# print("\n1. Comparación en Moons:")
# compare_algorithms(X_moons, "Moons Dataset", epsilon=0.2, min_samples=5)
#
# print("\n2. Comparación en Circles:")
# compare_algorithms(X_circles, "Circles Dataset", epsilon=0.2, min_samples=5)
#
# print("\n3. Comparación con Outliers:")
# compare_algorithms(X_outliers, "Dataset con Outliers", epsilon=0.8, min_samples=5)
#
# print("\n✓ Gráficos guardados")

print()

# ============================================
# PASO 9: Sensibilidad a Parámetros
# ============================================
print("--- Paso 9: Sensibilidad a Parámetros ---")

# Descomenta las siguientes líneas:
# def explore_parameters(X: np.ndarray, epsilons: list, 
#                        min_samples_list: list) -> None:
#     """
#     Explore how different parameters affect DBSCAN results.
#     """
#     n_eps = len(epsilons)
#     n_mins = len(min_samples_list)
#     
#     fig, axes = plt.subplots(n_mins, n_eps, figsize=(4*n_eps, 4*n_mins))
#     
#     for i, min_samples in enumerate(min_samples_list):
#         for j, epsilon in enumerate(epsilons):
#             ax = axes[i, j] if n_mins > 1 else axes[j]
#             
#             labels = dbscan_scratch(X, epsilon, min_samples, verbose=False)
#             n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#             n_noise = np.sum(labels == -1)
#             
#             colors = labels.astype(float)
#             ax.scatter(X[:, 0], X[:, 1], c=colors, cmap='viridis', 
#                       alpha=0.6, s=30)
#             
#             # Marcar ruido
#             noise_mask = labels == -1
#             if np.any(noise_mask):
#                 ax.scatter(X[noise_mask, 0], X[noise_mask, 1], 
#                           c='red', marker='x', s=30)
#             
#             ax.set_title(f'ε={epsilon}, min={min_samples}\n'
#                         f'C={n_clusters}, N={n_noise}')
#             ax.set_xticks([])
#             ax.set_yticks([])
#     
#     plt.suptitle('Exploración de Parámetros DBSCAN', fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig('parameter_exploration.png', dpi=150)
#     plt.show()
#     print("✓ Gráfico guardado: parameter_exploration.png")
#
# # Explorar parámetros
# epsilons = [0.1, 0.2, 0.3, 0.5]
# min_samples_list = [3, 5, 10]
# explore_parameters(X_moons, epsilons, min_samples_list)

print()

# ============================================
# PASO 10: Comparación con Sklearn
# ============================================
print("--- Paso 10: Comparación con Sklearn ---")

# Descomenta las siguientes líneas:
# # DBSCAN desde cero
# labels_scratch = dbscan_scratch(X_moons, epsilon=0.2, min_samples=5, verbose=False)
#
# # DBSCAN de sklearn
# dbscan_sklearn = DBSCAN(eps=0.2, min_samples=5)
# labels_sklearn = dbscan_sklearn.fit_predict(X_moons)
#
# # Comparar resultados
# print("\nComparación de resultados:")
# print(f"{'Métrica':<25} {'Scratch':<15} {'Sklearn':<15}")
# print("-" * 55)
#
# n_clusters_scratch = len(set(labels_scratch)) - (1 if -1 in labels_scratch else 0)
# n_clusters_sklearn = len(set(labels_sklearn)) - (1 if -1 in labels_sklearn else 0)
# print(f"{'Clusters':<25} {n_clusters_scratch:<15} {n_clusters_sklearn:<15}")
#
# n_noise_scratch = np.sum(labels_scratch == -1)
# n_noise_sklearn = np.sum(labels_sklearn == -1)
# print(f"{'Puntos de ruido':<25} {n_noise_scratch:<15} {n_noise_sklearn:<15}")
#
# # Visualizar comparación
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# for ax, labels, title in zip(axes, 
#                              [labels_scratch, labels_sklearn],
#                              ['DBSCAN Scratch', 'DBSCAN Sklearn']):
#     colors = labels.astype(float)
#     ax.scatter(X_moons[:, 0], X_moons[:, 1], c=colors, cmap='viridis', alpha=0.6)
#     noise_mask = labels == -1
#     if np.any(noise_mask):
#         ax.scatter(X_moons[noise_mask, 0], X_moons[noise_mask, 1], 
#                   c='red', marker='x', s=50, label='Ruido')
#     ax.set_title(title)
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig('dbscan_comparison.png', dpi=150)
# plt.show()
# print("\n✓ Gráfico guardado: dbscan_comparison.png")

print()

# ============================================
# RESUMEN
# ============================================
print("=" * 60)
print("RESUMEN DEL EJERCICIO")
print("=" * 60)
print("""
En este ejercicio aprendiste:

1. ✓ Cómo funciona DBSCAN basado en densidad
2. ✓ Diferencia entre puntos core, border y noise
3. ✓ Implementar region query para encontrar vecinos
4. ✓ Expandir clusters desde puntos core
5. ✓ Usar k-distance graph para elegir epsilon
6. ✓ Ventajas de DBSCAN sobre K-Means para formas complejas
7. ✓ Detectar automáticamente outliers
8. ✓ Sensibilidad a parámetros epsilon y min_samples

Archivos generados:
- datasets_complex.png
- point_classification.png
- k_distance_graph.png
- comparison_*.png
- parameter_exploration.png
- dbscan_comparison.png

Ventajas de DBSCAN:
✓ No necesita especificar K
✓ Encuentra clusters de cualquier forma
✓ Detecta outliers automáticamente
✓ Robusto a ruido

Desventajas:
✗ Sensible a epsilon y min_samples
✗ Problemas con densidades variables
✗ Complejidad O(n²) sin optimización
""")
