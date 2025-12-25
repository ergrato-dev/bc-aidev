# ============================================
# EJERCICIO 03: Clustering Jerárquico
# ============================================
# Implementación de clustering aglomerativo y dendrogramas
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

print("=" * 60)
print("EJERCICIO 03: Clustering Jerárquico")
print("=" * 60)

# ============================================
# PASO 1: Generar Datos
# ============================================
print("\n--- Paso 1: Generar Datos ---")

# Descomenta las siguientes líneas:
# np.random.seed(42)
#
# # Dataset con 4 clusters bien separados
# X, y_true = make_blobs(n_samples=150, centers=4, 
#                        cluster_std=0.6, random_state=42)
# print(f"Datos generados: {X.shape}")
# print(f"Clusters reales: {np.unique(y_true)}")
#
# # Visualizar datos
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Dataset Original (4 clusters)')
# plt.colorbar(label='Cluster real')
# plt.savefig('hierarchical_data.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: hierarchical_data.png")

print()

# ============================================
# PASO 2: Calcular Matriz de Distancias
# ============================================
print("--- Paso 2: Matriz de Distancias ---")

# Descomenta las siguientes líneas:
# # Distancias condensadas (formato para scipy)
# distances_condensed = pdist(X, metric='euclidean')
# print(f"Distancias condensadas: {distances_condensed.shape}")
#
# # Matriz cuadrada de distancias
# distance_matrix = squareform(distances_condensed)
# print(f"Matriz de distancias: {distance_matrix.shape}")
#
# # Visualizar matriz de distancias
# plt.figure(figsize=(10, 8))
# plt.imshow(distance_matrix, cmap='viridis')
# plt.colorbar(label='Distancia Euclidiana')
# plt.title('Matriz de Distancias entre Puntos')
# plt.xlabel('Índice de punto')
# plt.ylabel('Índice de punto')
# plt.savefig('distance_matrix.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: distance_matrix.png")
#
# # Estadísticas de distancias
# print(f"\nEstadísticas de distancias:")
# print(f"  Min: {distances_condensed.min():.2f}")
# print(f"  Max: {distances_condensed.max():.2f}")
# print(f"  Media: {distances_condensed.mean():.2f}")
# print(f"  Mediana: {np.median(distances_condensed):.2f}")

print()

# ============================================
# PASO 3: Clustering Jerárquico con Ward
# ============================================
print("--- Paso 3: Linkage Ward ---")

# Descomenta las siguientes líneas:
# # Calcular linkage con método Ward
# Z_ward = linkage(X, method='ward')
# print(f"Matriz de linkage: {Z_ward.shape}")
# print(f"  - Cada fila: [cluster1, cluster2, distancia, n_elementos]")
#
# # Mostrar primeras fusiones
# print("\nPrimeras 5 fusiones:")
# print(f"{'Cluster 1':<12} {'Cluster 2':<12} {'Distancia':<12} {'N elementos':<12}")
# print("-" * 48)
# for i in range(5):
#     print(f"{int(Z_ward[i, 0]):<12} {int(Z_ward[i, 1]):<12} "
#           f"{Z_ward[i, 2]:<12.2f} {int(Z_ward[i, 3]):<12}")

print()

# ============================================
# PASO 4: Crear Dendrograma
# ============================================
print("--- Paso 4: Dendrograma ---")

# Descomenta las siguientes líneas:
# def plot_dendrogram(Z: np.ndarray, title: str, 
#                     cut_height: float = None) -> None:
#     """
#     Plot dendrogram with optional cut line.
#     
#     Args:
#         Z: Linkage matrix
#         title: Plot title
#         cut_height: Height to draw horizontal cut line
#     """
#     plt.figure(figsize=(14, 6))
#     
#     dendrogram(Z, 
#                truncate_mode='lastp',  # Mostrar últimos p clusters
#                p=30,                    # Número de hojas
#                leaf_rotation=90,
#                leaf_font_size=8,
#                show_contracted=True)
#     
#     if cut_height is not None:
#         plt.axhline(y=cut_height, color='r', linestyle='--', 
#                    label=f'Corte en altura={cut_height}')
#         plt.legend()
#     
#     plt.title(title, fontsize=14)
#     plt.xlabel('Índice de muestra (o tamaño del cluster)', fontsize=12)
#     plt.ylabel('Distancia', fontsize=12)
#     plt.grid(True, alpha=0.3, axis='y')
#     plt.tight_layout()
#
# # Dendrograma completo
# plot_dendrogram(Z_ward, 'Dendrograma - Método Ward', cut_height=10)
# plt.savefig('dendrogram_ward.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: dendrogram_ward.png")

print()

# ============================================
# PASO 5: Comparar Métodos de Linkage
# ============================================
print("--- Paso 5: Comparar Métodos de Linkage ---")

# Descomenta las siguientes líneas:
# methods = ['single', 'complete', 'average', 'ward']
# descriptions = {
#     'single': 'Mínima distancia (chain effect)',
#     'complete': 'Máxima distancia (compacto)',
#     'average': 'Distancia promedio',
#     'ward': 'Minimiza varianza (más usado)'
# }
#
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# axes = axes.flatten()
#
# linkage_matrices = {}
#
# for ax, method in zip(axes, methods):
#     Z = linkage(X, method=method)
#     linkage_matrices[method] = Z
#     
#     plt.sca(ax)
#     dendrogram(Z, truncate_mode='lastp', p=20, 
#                leaf_rotation=90, leaf_font_size=8)
#     ax.set_title(f'{method.capitalize()}\n{descriptions[method]}', fontsize=12)
#     ax.set_xlabel('Índice/Tamaño')
#     ax.set_ylabel('Distancia')
#     ax.grid(True, alpha=0.3, axis='y')
#
# plt.suptitle('Comparación de Métodos de Linkage', fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.savefig('linkage_comparison.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: linkage_comparison.png")

print()

# ============================================
# PASO 6: Cortar el Dendrograma
# ============================================
print("--- Paso 6: Cortar Dendrograma ---")

# Descomenta las siguientes líneas:
# # Cortar por distancia
# cut_distance = 7
# labels_by_distance = fcluster(Z_ward, t=cut_distance, criterion='distance')
# n_clusters_dist = len(np.unique(labels_by_distance))
# print(f"Corte por distancia (t={cut_distance}): {n_clusters_dist} clusters")
#
# # Cortar por número de clusters
# n_clusters_target = 4
# labels_by_k = fcluster(Z_ward, t=n_clusters_target, criterion='maxclust')
# n_clusters_k = len(np.unique(labels_by_k))
# print(f"Corte por maxclust (t={n_clusters_target}): {n_clusters_k} clusters")
#
# # Visualizar ambos cortes
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=labels_by_distance, 
#                           cmap='viridis', alpha=0.7)
# axes[0].set_title(f'Corte por distancia (t={cut_distance})\n'
#                  f'{n_clusters_dist} clusters')
# plt.colorbar(scatter1, ax=axes[0])
#
# scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=labels_by_k, 
#                           cmap='viridis', alpha=0.7)
# axes[1].set_title(f'Corte por maxclust (t={n_clusters_target})\n'
#                  f'{n_clusters_k} clusters')
# plt.colorbar(scatter2, ax=axes[1])
#
# for ax in axes:
#     ax.set_xlabel('Feature 1')
#     ax.set_ylabel('Feature 2')
#     ax.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig('dendrogram_cuts.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: dendrogram_cuts.png")

print()

# ============================================
# PASO 7: Scikit-learn AgglomerativeClustering
# ============================================
print("--- Paso 7: Scikit-learn AgglomerativeClustering ---")

# Descomenta las siguientes líneas:
# # Especificando número de clusters
# agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
# labels_sklearn = agg_clustering.fit_predict(X)
#
# print(f"Clusters encontrados: {len(np.unique(labels_sklearn))}")
# print(f"Distribución: {np.bincount(labels_sklearn)}")
#
# # Sin especificar n_clusters (usando distance_threshold)
# agg_clustering_auto = AgglomerativeClustering(
#     n_clusters=None,
#     distance_threshold=7,
#     linkage='ward'
# )
# labels_auto = agg_clustering_auto.fit_predict(X)
# print(f"\nCon distance_threshold=7: {len(np.unique(labels_auto))} clusters")
#
# # Visualizar
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=labels_sklearn, 
#                           cmap='viridis', alpha=0.7)
# axes[0].set_title(f'AgglomerativeClustering (n_clusters=4)\n'
#                  f'Distribución: {np.bincount(labels_sklearn)}')
#
# scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=labels_auto, 
#                           cmap='viridis', alpha=0.7)
# axes[1].set_title(f'AgglomerativeClustering (distance_threshold=7)\n'
#                  f'{len(np.unique(labels_auto))} clusters')
#
# for ax in axes:
#     ax.set_xlabel('Feature 1')
#     ax.set_ylabel('Feature 2')
#
# plt.tight_layout()
# plt.savefig('sklearn_agg_clustering.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: sklearn_agg_clustering.png")

print()

# ============================================
# PASO 8: Efecto del Linkage en Clusters
# ============================================
print("--- Paso 8: Efecto del Linkage ---")

# Descomenta las siguientes líneas:
# # Comparar clusters finales con diferentes linkages
# fig, axes = plt.subplots(2, 2, figsize=(14, 12))
# axes = axes.flatten()
#
# for ax, method in zip(axes, methods):
#     # Clustering con 4 clusters
#     agg = AgglomerativeClustering(n_clusters=4, linkage=method)
#     labels = agg.fit_predict(X)
#     
#     scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
#     ax.set_title(f'Linkage: {method.capitalize()}\n'
#                 f'Distribución: {np.bincount(labels)}')
#     ax.set_xlabel('Feature 1')
#     ax.set_ylabel('Feature 2')
#     ax.grid(True, alpha=0.3)
#
# plt.suptitle('Efecto del Método de Linkage en Clusters (K=4)', 
#             fontsize=14, fontweight='bold')
# plt.tight_layout()
# plt.savefig('linkage_effect.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: linkage_effect.png")

print()

# ============================================
# PASO 9: Single Linkage y Chain Effect
# ============================================
print("--- Paso 9: Chain Effect (Single Linkage) ---")

# Descomenta las siguientes líneas:
# # Datos en forma de cadena (donde single linkage falla)
# X_chain = np.vstack([
#     np.random.randn(50, 2) + [0, 0],
#     np.random.randn(50, 2) + [3, 0],
#     np.random.randn(10, 2) + [1.5, 0]  # Puntos puente
# ])
#
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#
# # Datos originales
# axes[0].scatter(X_chain[:, 0], X_chain[:, 1], alpha=0.6)
# axes[0].set_title('Datos con puntos puente')
#
# # Single linkage (chain effect)
# labels_single = AgglomerativeClustering(n_clusters=2, 
#                                         linkage='single').fit_predict(X_chain)
# axes[1].scatter(X_chain[:, 0], X_chain[:, 1], c=labels_single, 
#                cmap='viridis', alpha=0.6)
# axes[1].set_title('Single Linkage\n(chain effect - conecta por puente)')
#
# # Complete linkage (evita chain effect)
# labels_complete = AgglomerativeClustering(n_clusters=2, 
#                                           linkage='complete').fit_predict(X_chain)
# axes[2].scatter(X_chain[:, 0], X_chain[:, 1], c=labels_complete, 
#                cmap='viridis', alpha=0.6)
# axes[2].set_title('Complete Linkage\n(más robusto)')
#
# for ax in axes:
#     ax.set_xlabel('Feature 1')
#     ax.set_ylabel('Feature 2')
#
# plt.tight_layout()
# plt.savefig('chain_effect.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: chain_effect.png")

print()

# ============================================
# PASO 10: Comparación con K-Means
# ============================================
print("--- Paso 10: Jerárquico vs K-Means ---")

# Descomenta las siguientes líneas:
# # K-Means
# kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
# labels_kmeans = kmeans.fit_predict(X)
#
# # Jerárquico (Ward)
# labels_hier = AgglomerativeClustering(n_clusters=4, 
#                                       linkage='ward').fit_predict(X)
#
# # Visualizar comparación
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# axes[0].scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis', alpha=0.7)
# axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#                c='red', marker='X', s=200, edgecolors='black', label='Centroides')
# axes[0].set_title(f'K-Means (K=4)\nDistribución: {np.bincount(labels_kmeans)}')
# axes[0].legend()
#
# axes[1].scatter(X[:, 0], X[:, 1], c=labels_hier, cmap='viridis', alpha=0.7)
# axes[1].set_title(f'Jerárquico Ward (K=4)\nDistribución: {np.bincount(labels_hier)}')
#
# for ax in axes:
#     ax.set_xlabel('Feature 1')
#     ax.set_ylabel('Feature 2')
#     ax.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig('hierarchical_vs_kmeans.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: hierarchical_vs_kmeans.png")

print()

# ============================================
# PASO 11: Dendrograma con Colores por Cluster
# ============================================
print("--- Paso 11: Dendrograma Coloreado ---")

# Descomenta las siguientes líneas:
# def colored_dendrogram(Z: np.ndarray, n_clusters: int, title: str) -> None:
#     """
#     Create dendrogram with colored clusters.
#     """
#     plt.figure(figsize=(14, 8))
#     
#     # Calcular umbral de color basado en número de clusters
#     # El umbral está en la altura que produce n_clusters
#     threshold = Z[-(n_clusters-1), 2]
#     
#     dendrogram(Z, 
#                truncate_mode='lastp',
#                p=40,
#                leaf_rotation=90,
#                leaf_font_size=8,
#                color_threshold=threshold,
#                above_threshold_color='gray')
#     
#     plt.axhline(y=threshold, color='r', linestyle='--', 
#                label=f'Umbral para {n_clusters} clusters')
#     plt.title(title, fontsize=14)
#     plt.xlabel('Índice de muestra', fontsize=12)
#     plt.ylabel('Distancia', fontsize=12)
#     plt.legend()
#     plt.grid(True, alpha=0.3, axis='y')
#     plt.tight_layout()
#
# # Crear dendrograma coloreado
# colored_dendrogram(Z_ward, n_clusters=4, 
#                   title='Dendrograma Ward - 4 Clusters Coloreados')
# plt.savefig('dendrogram_colored.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: dendrogram_colored.png")

print()

# ============================================
# PASO 12: Implementación Básica desde Cero
# ============================================
print("--- Paso 12: Implementación Básica (Bonus) ---")

# Descomenta las siguientes líneas:
# def single_linkage_simple(X: np.ndarray, n_clusters: int) -> np.ndarray:
#     """
#     Simple implementation of single linkage hierarchical clustering.
#     
#     This is a naive O(n³) implementation for educational purposes.
#     """
#     n_samples = X.shape[0]
#     
#     # Cada punto empieza en su propio cluster
#     labels = np.arange(n_samples)
#     active_clusters = set(range(n_samples))
#     
#     # Calcular matriz de distancias
#     distances = squareform(pdist(X))
#     np.fill_diagonal(distances, np.inf)  # Ignorar auto-distancias
#     
#     # Fusionar hasta tener n_clusters
#     while len(active_clusters) > n_clusters:
#         # Encontrar par de clusters más cercanos
#         min_dist = np.inf
#         merge_i, merge_j = -1, -1
#         
#         for i in active_clusters:
#             for j in active_clusters:
#                 if i < j:
#                     # Encontrar mínima distancia entre puntos de ambos clusters
#                     points_i = np.where(labels == i)[0]
#                     points_j = np.where(labels == j)[0]
#                     
#                     min_pair_dist = np.min(distances[np.ix_(points_i, points_j)])
#                     
#                     if min_pair_dist < min_dist:
#                         min_dist = min_pair_dist
#                         merge_i, merge_j = i, j
#         
#         # Fusionar clusters (j se une a i)
#         labels[labels == merge_j] = merge_i
#         active_clusters.remove(merge_j)
#     
#     # Renumerar clusters de 0 a n_clusters-1
#     unique_labels = np.unique(labels)
#     new_labels = np.zeros_like(labels)
#     for new_id, old_id in enumerate(unique_labels):
#         new_labels[labels == old_id] = new_id
#     
#     return new_labels
#
# # Probar implementación
# print("Ejecutando single linkage desde cero...")
# labels_scratch = single_linkage_simple(X, n_clusters=4)
# print(f"Clusters: {len(np.unique(labels_scratch))}")
# print(f"Distribución: {np.bincount(labels_scratch)}")
#
# # Comparar con sklearn
# labels_sklearn_single = AgglomerativeClustering(
#     n_clusters=4, linkage='single').fit_predict(X)
#
# # Visualizar
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# axes[0].scatter(X[:, 0], X[:, 1], c=labels_scratch, cmap='viridis', alpha=0.7)
# axes[0].set_title(f'Single Linkage (scratch)\nDistribución: {np.bincount(labels_scratch)}')
#
# axes[1].scatter(X[:, 0], X[:, 1], c=labels_sklearn_single, cmap='viridis', alpha=0.7)
# axes[1].set_title(f'Single Linkage (sklearn)\nDistribución: {np.bincount(labels_sklearn_single)}')
#
# for ax in axes:
#     ax.set_xlabel('Feature 1')
#     ax.set_ylabel('Feature 2')
#
# plt.tight_layout()
# plt.savefig('hierarchical_scratch.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: hierarchical_scratch.png")

print()

# ============================================
# RESUMEN
# ============================================
print("=" * 60)
print("RESUMEN DEL EJERCICIO")
print("=" * 60)
print("""
En este ejercicio aprendiste:

1. ✓ Calcular matrices de distancias con scipy
2. ✓ Crear dendrogramas con diferentes linkages
3. ✓ Interpretar la estructura jerárquica
4. ✓ Cortar dendrograma por distancia o número de clusters
5. ✓ Usar AgglomerativeClustering de sklearn
6. ✓ Diferencias entre single, complete, average y ward
7. ✓ Entender el chain effect en single linkage
8. ✓ Comparar con K-Means

Archivos generados:
- hierarchical_data.png
- distance_matrix.png
- dendrogram_ward.png
- linkage_comparison.png
- dendrogram_cuts.png
- sklearn_agg_clustering.png
- linkage_effect.png
- chain_effect.png
- hierarchical_vs_kmeans.png
- dendrogram_colored.png
- hierarchical_scratch.png

Métodos de Linkage:
- Single: Mínima distancia (chain effect)
- Complete: Máxima distancia (compacto)
- Average: Promedio de distancias
- Ward: Minimiza varianza (recomendado)

Ventajas del Clustering Jerárquico:
✓ No necesita K de antemano
✓ Proporciona jerarquía completa
✓ Dendrograma interpretable
✓ Diferentes cortes = diferentes K

Desventajas:
✗ Complejidad O(n²) o O(n³)
✗ No escala bien a datasets grandes
✗ Decisiones de fusión son irreversibles
""")
