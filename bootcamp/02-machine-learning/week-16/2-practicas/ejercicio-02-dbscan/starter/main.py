"""
Ejercicio 02: DBSCAN Clustering
===============================
Aprende clustering basado en densidad con DBSCAN.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# ============================================
# PASO 1: Datos donde K-Means Falla
# ============================================
print('--- Paso 1: Datos Moons ---')

# Descomenta las siguientes líneas:
# X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)
# X_scaled = StandardScaler().fit_transform(X_moons)
# 
# plt.figure(figsize=(8, 5))
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_moons, cmap='viridis', alpha=0.6)
# plt.title('Datos Moons (Ground Truth)')
# plt.show()
# 
# print('Estos datos tienen forma no convexa - K-Means fallará')

print()


# ============================================
# PASO 2: DBSCAN Básico
# ============================================
print('--- Paso 2: DBSCAN Básico ---')

# Descomenta las siguientes líneas:
# dbscan = DBSCAN(
#     eps=0.3,           # Radio de vecindad
#     min_samples=5,     # Mínimo de puntos para ser core
#     metric='euclidean'
# )
# 
# labels = dbscan.fit_predict(X_scaled)
# 
# # Contar clusters y noise
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise = list(labels).count(-1)
# 
# print(f'Clusters encontrados: {n_clusters}')
# print(f'Puntos de ruido: {n_noise}')
# print(f'Core points: {len(dbscan.core_sample_indices_)}')

print()


# ============================================
# PASO 3: Visualizar Tipos de Puntos
# ============================================
print('--- Paso 3: Visualizar Core/Border/Noise ---')

# Descomenta las siguientes líneas:
# # Identificar tipos
# core_mask = np.zeros_like(labels, dtype=bool)
# core_mask[dbscan.core_sample_indices_] = True
# 
# noise_mask = labels == -1
# border_mask = ~core_mask & ~noise_mask
# 
# plt.figure(figsize=(10, 6))
# 
# # Core points
# plt.scatter(X_scaled[core_mask, 0], X_scaled[core_mask, 1], 
#            c=labels[core_mask], cmap='viridis', s=100, 
#            edgecolors='black', linewidths=1, label='Core')
# 
# # Border points
# plt.scatter(X_scaled[border_mask, 0], X_scaled[border_mask, 1],
#            c=labels[border_mask], cmap='viridis', s=50,
#            alpha=0.6, label='Border')
# 
# # Noise points
# plt.scatter(X_scaled[noise_mask, 0], X_scaled[noise_mask, 1],
#            c='red', marker='x', s=50, label='Noise')
# 
# plt.title(f'DBSCAN (eps=0.3, min_samples=5)\nClusters: {n_clusters}, Noise: {n_noise}')
# plt.legend()
# plt.show()

print()


# ============================================
# PASO 4: Comparar DBSCAN vs K-Means
# ============================================
print('--- Paso 4: DBSCAN vs K-Means ---')

# Descomenta las siguientes líneas:
# from sklearn.cluster import KMeans
# 
# # K-Means
# kmeans = KMeans(n_clusters=2, random_state=42)
# labels_km = kmeans.fit_predict(X_scaled)
# 
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# 
# # Ground Truth
# axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_moons, cmap='viridis')
# axes[0].set_title('Ground Truth')
# 
# # K-Means
# axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_km, cmap='viridis')
# axes[1].set_title('K-Means (INCORRECTO)')
# 
# # DBSCAN
# axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
# axes[2].set_title('DBSCAN (CORRECTO)')
# 
# plt.tight_layout()
# plt.show()
# 
# print('DBSCAN detecta correctamente las formas no convexas')

print()


# ============================================
# PASO 5: Selección de eps (k-distance graph)
# ============================================
print('--- Paso 5: Encontrar eps óptimo ---')

# Descomenta las siguientes líneas:
# from sklearn.neighbors import NearestNeighbors
# 
# # Calcular distancia al k-ésimo vecino
# k = 5  # min_samples
# nn = NearestNeighbors(n_neighbors=k)
# nn.fit(X_scaled)
# distances, _ = nn.kneighbors(X_scaled)
# 
# # Ordenar distancias (al k-ésimo vecino)
# k_distances = np.sort(distances[:, k-1])[::-1]
# 
# plt.figure(figsize=(10, 5))
# plt.plot(k_distances)
# plt.xlabel('Puntos (ordenados)')
# plt.ylabel(f'Distancia al {k}-ésimo vecino')
# plt.title('k-Distance Graph para selección de eps')
# plt.axhline(y=0.3, color='r', linestyle='--', label='eps=0.3')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()
# 
# print('El "codo" en la gráfica sugiere un buen valor de eps')

print()


# ============================================
# PASO 6: Efecto de eps y min_samples
# ============================================
print('--- Paso 6: Variar Parámetros ---')

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# 
# # Variar eps
# for i, eps in enumerate([0.1, 0.3, 0.5]):
#     db = DBSCAN(eps=eps, min_samples=5)
#     lbls = db.fit_predict(X_scaled)
#     n_clust = len(set(lbls)) - (1 if -1 in lbls else 0)
#     n_noi = (lbls == -1).sum()
#     
#     axes[0, i].scatter(X_scaled[:, 0], X_scaled[:, 1], c=lbls, cmap='viridis')
#     axes[0, i].set_title(f'eps={eps}\nClusters: {n_clust}, Noise: {n_noi}')
# 
# # Variar min_samples
# for i, min_s in enumerate([3, 5, 10]):
#     db = DBSCAN(eps=0.3, min_samples=min_s)
#     lbls = db.fit_predict(X_scaled)
#     n_clust = len(set(lbls)) - (1 if -1 in lbls else 0)
#     n_noi = (lbls == -1).sum()
#     
#     axes[1, i].scatter(X_scaled[:, 0], X_scaled[:, 1], c=lbls, cmap='viridis')
#     axes[1, i].set_title(f'min_samples={min_s}\nClusters: {n_clust}, Noise: {n_noi}')
# 
# axes[0, 0].set_ylabel('Variar eps')
# axes[1, 0].set_ylabel('Variar min_samples')
# plt.suptitle('Efecto de los parámetros de DBSCAN')
# plt.tight_layout()
# plt.show()

print()


# ============================================
# PASO 7: DBSCAN con Outliers
# ============================================
print('--- Paso 7: Detección de Outliers ---')

# Descomenta las siguientes líneas:
# # Crear datos con outliers
# X_blobs, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
# 
# # Añadir outliers
# outliers = np.random.uniform(low=-8, high=8, size=(15, 2))
# X_with_outliers = np.vstack([X_blobs, outliers])
# X_scaled_out = StandardScaler().fit_transform(X_with_outliers)
# 
# # DBSCAN
# db = DBSCAN(eps=0.5, min_samples=5)
# labels_out = db.fit_predict(X_scaled_out)
# 
# # Visualizar
# plt.figure(figsize=(10, 6))
# 
# # Clusters
# mask = labels_out >= 0
# plt.scatter(X_scaled_out[mask, 0], X_scaled_out[mask, 1], 
#            c=labels_out[mask], cmap='viridis', s=50, label='Clusters')
# 
# # Outliers (noise)
# plt.scatter(X_scaled_out[~mask, 0], X_scaled_out[~mask, 1],
#            c='red', marker='x', s=100, linewidths=2, label='Outliers detectados')
# 
# n_outliers = (~mask).sum()
# plt.title(f'DBSCAN: Detección de Outliers\nOutliers detectados: {n_outliers}')
# plt.legend()
# plt.show()
# 
# print(f'DBSCAN detectó {n_outliers} outliers automáticamente')

print()
print('=== Ejercicio completado ===')
