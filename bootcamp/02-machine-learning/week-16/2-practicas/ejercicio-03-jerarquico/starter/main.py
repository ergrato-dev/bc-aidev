"""
Ejercicio 03: Clustering Jerárquico
===================================
Aprende clustering jerárquico y dendrogramas.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# ============================================
# PASO 1: Generar Datos
# ============================================
print('--- Paso 1: Generar Datos ---')

# Descomenta las siguientes líneas:
# X, y_true = make_blobs(n_samples=100, centers=4, cluster_std=0.8, random_state=42)
# X_scaled = StandardScaler().fit_transform(X)
# 
# plt.figure(figsize=(8, 5))
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', alpha=0.6)
# plt.title('Datos de ejemplo (4 clusters)')
# plt.show()

print()


# ============================================
# PASO 2: Crear Dendrograma con Scipy
# ============================================
print('--- Paso 2: Dendrograma ---')

# Descomenta las siguientes líneas:
# from scipy.cluster.hierarchy import dendrogram, linkage
# 
# # Calcular linkage matrix
# Z = linkage(X_scaled, method='ward')
# 
# # Visualizar dendrograma
# plt.figure(figsize=(14, 6))
# dendrogram(Z, 
#            truncate_mode='lastp',
#            p=30,
#            leaf_rotation=90,
#            leaf_font_size=8)
# plt.title('Dendrograma (Ward Linkage)')
# plt.xlabel('Muestras')
# plt.ylabel('Distancia')
# plt.axhline(y=5, color='r', linestyle='--', label='Corte sugerido')
# plt.legend()
# plt.tight_layout()
# plt.show()
# 
# print('El dendrograma muestra la jerarquía de fusiones')
# print('El corte horizontal determina el número de clusters')

print()


# ============================================
# PASO 3: Extraer Clusters del Dendrograma
# ============================================
print('--- Paso 3: Extraer Clusters ---')

# Descomenta las siguientes líneas:
# from scipy.cluster.hierarchy import fcluster
# 
# # Método 1: Por número de clusters
# labels_k4 = fcluster(Z, t=4, criterion='maxclust')
# 
# # Método 2: Por distancia de corte
# labels_d5 = fcluster(Z, t=5, criterion='distance')
# 
# print(f'Clusters (K=4): {np.unique(labels_k4)}')
# print(f'Clusters (dist=5): {np.unique(labels_d5)}')
# print(f'Tamaños (K=4): {np.bincount(labels_k4)}')

print()


# ============================================
# PASO 4: Clustering con Sklearn
# ============================================
print('--- Paso 4: AgglomerativeClustering ---')

# Descomenta las siguientes líneas:
# from sklearn.cluster import AgglomerativeClustering
# 
# agg = AgglomerativeClustering(
#     n_clusters=4,
#     linkage='ward'
# )
# labels = agg.fit_predict(X_scaled)
# 
# plt.figure(figsize=(10, 6))
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50)
# plt.title('Clustering Jerárquico Aglomerativo (K=4, Ward)')
# plt.colorbar(label='Cluster')
# plt.show()

print()


# ============================================
# PASO 5: Comparar Métodos de Linkage
# ============================================
print('--- Paso 5: Comparar Linkages ---')

# Descomenta las siguientes líneas:
# linkages = ['ward', 'complete', 'average', 'single']
# 
# fig, axes = plt.subplots(2, 4, figsize=(16, 8))
# 
# for i, method in enumerate(linkages):
#     # Dendrograma
#     Z_method = linkage(X_scaled, method=method)
#     dendrogram(Z_method, ax=axes[0, i], truncate_mode='lastp', p=20,
#                leaf_rotation=90, leaf_font_size=6)
#     axes[0, i].set_title(f'{method.capitalize()} Linkage')
#     
#     # Clusters
#     if method == 'ward':
#         agg = AgglomerativeClustering(n_clusters=4, linkage=method)
#     else:
#         agg = AgglomerativeClustering(n_clusters=4, linkage=method)
#     lbls = agg.fit_predict(X_scaled)
#     axes[1, i].scatter(X_scaled[:, 0], X_scaled[:, 1], c=lbls, cmap='viridis', s=30)
#     axes[1, i].set_title(f'Clusters ({method})')
# 
# axes[0, 0].set_ylabel('Dendrogramas')
# axes[1, 0].set_ylabel('Clusters')
# plt.tight_layout()
# plt.show()
# 
# print('Ward: clusters compactos, esféricos')
# print('Single: puede crear "cadenas" (chaining)')
# print('Complete: clusters más grandes')
# print('Average: balance entre single y complete')

print()


# ============================================
# PASO 6: Encontrar K Óptimo
# ============================================
print('--- Paso 6: Silhouette para Elegir K ---')

# Descomenta las siguientes líneas:
# from sklearn.metrics import silhouette_score
# 
# silhouette_scores = []
# K_range = range(2, 10)
# 
# for k in K_range:
#     agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
#     lbls = agg.fit_predict(X_scaled)
#     score = silhouette_score(X_scaled, lbls)
#     silhouette_scores.append(score)
#     print(f'K={k}: Silhouette={score:.4f}')
# 
# plt.figure(figsize=(10, 5))
# plt.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
# plt.xlabel('Número de Clusters (K)')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score para Clustering Jerárquico')
# plt.grid(True, alpha=0.3)
# 
# best_k = K_range[np.argmax(silhouette_scores)]
# plt.axvline(x=best_k, color='r', linestyle='--', label=f'Mejor K={best_k}')
# plt.legend()
# plt.show()

print()


# ============================================
# PASO 7: Visualización Completa
# ============================================
print('--- Paso 7: Visualización Completa ---')

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# 
# # Ground Truth
# axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis')
# axes[0].set_title('Ground Truth')
# 
# # Dendrograma
# Z_final = linkage(X_scaled, method='ward')
# dendrogram(Z_final, ax=axes[1], truncate_mode='lastp', p=15,
#            leaf_rotation=90, leaf_font_size=8)
# axes[1].axhline(y=5, color='r', linestyle='--')
# axes[1].set_title('Dendrograma (Ward)')
# 
# # Clustering final
# final_labels = fcluster(Z_final, t=4, criterion='maxclust')
# axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=final_labels, cmap='viridis')
# axes[2].set_title(f'Clustering Jerárquico (K=4)')
# 
# plt.tight_layout()
# plt.show()

print()
print('=== Ejercicio completado ===')
