"""
Ejercicio 01: K-Means Clustering
================================
Aprende a implementar y usar K-Means para clustering.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# ============================================
# PASO 1: Generar Datos de Ejemplo
# ============================================
print('--- Paso 1: Generar Datos ---')

# Descomenta las siguientes líneas:
# X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
# print(f'Datos generados: {X.shape}')
# print(f'Clusters reales: {len(np.unique(y_true))}')

# # Escalar datos (IMPORTANTE para clustering)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# print('Datos escalados')

# # Visualizar datos originales
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
# plt.title('Datos con etiquetas reales')
# plt.subplot(1, 2, 2)
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.6)
# plt.title('Datos escalados (sin etiquetas)')
# plt.tight_layout()
# plt.show()

print()


# ============================================
# PASO 2: K-Means desde Cero
# ============================================
print('--- Paso 2: K-Means desde Cero ---')

# Descomenta las siguientes líneas:
# class SimpleKMeans:
#     def __init__(self, n_clusters=3, max_iters=100, random_state=None):
#         self.n_clusters = n_clusters
#         self.max_iters = max_iters
#         self.random_state = random_state
#     
#     def fit(self, X):
#         np.random.seed(self.random_state)
#         
#         # Inicializar centroides aleatoriamente
#         idx = np.random.choice(len(X), self.n_clusters, replace=False)
#         self.centroids = X[idx].copy()
#         
#         for i in range(self.max_iters):
#             # Asignar puntos al centroide más cercano
#             self.labels_ = self._assign_clusters(X)
#             
#             # Calcular nuevos centroides
#             new_centroids = self._update_centroids(X)
#             
#             # Verificar convergencia
#             if np.allclose(self.centroids, new_centroids):
#                 print(f'Convergió en iteración {i+1}')
#                 break
#             
#             self.centroids = new_centroids
#         
#         self.cluster_centers_ = self.centroids
#         return self
#     
#     def _assign_clusters(self, X):
#         distances = np.zeros((len(X), self.n_clusters))
#         for k in range(self.n_clusters):
#             distances[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)
#         return np.argmin(distances, axis=1)
#     
#     def _update_centroids(self, X):
#         centroids = np.zeros((self.n_clusters, X.shape[1]))
#         for k in range(self.n_clusters):
#             mask = self.labels_ == k
#             if mask.sum() > 0:
#                 centroids[k] = X[mask].mean(axis=0)
#         return centroids
#     
#     def predict(self, X):
#         return self._assign_clusters(X)

# # Probar nuestra implementación
# my_kmeans = SimpleKMeans(n_clusters=4, random_state=42)
# my_kmeans.fit(X_scaled)
# 
# print(f'Clusters encontrados: {len(np.unique(my_kmeans.labels_))}')

print()


# ============================================
# PASO 3: K-Means con Scikit-learn
# ============================================
print('--- Paso 3: K-Means con Scikit-learn ---')

# Descomenta las siguientes líneas:
# from sklearn.cluster import KMeans
# 
# # Crear modelo
# kmeans = KMeans(
#     n_clusters=4,
#     init='k-means++',      # Inicialización inteligente
#     n_init=10,             # Ejecutar 10 veces con diferentes inicializaciones
#     max_iter=300,
#     random_state=42
# )
# 
# # Ajustar
# labels = kmeans.fit_predict(X_scaled)
# 
# print(f'Inercia: {kmeans.inertia_:.2f}')
# print(f'Iteraciones: {kmeans.n_iter_}')
# print(f'Tamaño clusters: {np.bincount(labels)}')

print()


# ============================================
# PASO 4: Visualizar Resultados
# ============================================
print('--- Paso 4: Visualizar Resultados ---')

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# 
# # 1. Datos originales con etiquetas reales
# axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', alpha=0.6)
# axes[0].set_title('Ground Truth')
# 
# # 2. Nuestra implementación
# axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=my_kmeans.labels_, cmap='viridis', alpha=0.6)
# axes[1].scatter(my_kmeans.cluster_centers_[:, 0], my_kmeans.cluster_centers_[:, 1],
#                 c='red', marker='X', s=200, edgecolors='black')
# axes[1].set_title('Nuestra Implementación')
# 
# # 3. Sklearn K-Means
# axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', alpha=0.6)
# axes[2].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#                 c='red', marker='X', s=200, edgecolors='black')
# axes[2].set_title(f'Sklearn K-Means (Inercia: {kmeans.inertia_:.2f})')
# 
# plt.tight_layout()
# plt.show()

print()


# ============================================
# PASO 5: Comparar Inicializaciones
# ============================================
print('--- Paso 5: Comparar Inicializaciones ---')

# Descomenta las siguientes líneas:
# # K-Means++ vs Random
# kmeans_pp = KMeans(n_clusters=4, init='k-means++', n_init=1, random_state=42)
# kmeans_random = KMeans(n_clusters=4, init='random', n_init=1, random_state=42)
# 
# labels_pp = kmeans_pp.fit_predict(X_scaled)
# labels_random = kmeans_random.fit_predict(X_scaled)
# 
# print(f'K-Means++: Inercia={kmeans_pp.inertia_:.2f}, Iters={kmeans_pp.n_iter_}')
# print(f'Random:    Inercia={kmeans_random.inertia_:.2f}, Iters={kmeans_random.n_iter_}')
# 
# # Visualizar
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 
# axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_pp, cmap='viridis', alpha=0.6)
# axes[0].scatter(kmeans_pp.cluster_centers_[:, 0], kmeans_pp.cluster_centers_[:, 1],
#                 c='red', marker='X', s=200)
# axes[0].set_title(f'K-Means++ (Inercia: {kmeans_pp.inertia_:.2f})')
# 
# axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_random, cmap='viridis', alpha=0.6)
# axes[1].scatter(kmeans_random.cluster_centers_[:, 0], kmeans_random.cluster_centers_[:, 1],
#                 c='red', marker='X', s=200)
# axes[1].set_title(f'Random Init (Inercia: {kmeans_random.inertia_:.2f})')
# 
# plt.tight_layout()
# plt.show()

print()


# ============================================
# PASO 6: K-Means con Diferentes K
# ============================================
print('--- Paso 6: Probar Diferentes K ---')

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# axes = axes.flatten()
# 
# for i, k in enumerate([2, 3, 4, 5, 6, 7]):
#     km = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels_k = km.fit_predict(X_scaled)
#     
#     axes[i].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_k, cmap='viridis', alpha=0.6)
#     axes[i].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
#                     c='red', marker='X', s=150)
#     axes[i].set_title(f'K={k}, Inercia={km.inertia_:.2f}')
# 
# plt.suptitle('K-Means con diferentes valores de K')
# plt.tight_layout()
# plt.show()

print()


# ============================================
# PASO 7: Limitación - Formas No Esféricas
# ============================================
print('--- Paso 7: Limitación con Formas Irregulares ---')

# Descomenta las siguientes líneas:
# from sklearn.datasets import make_moons
# 
# # Datos con forma de lunas
# X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)
# X_moons_scaled = StandardScaler().fit_transform(X_moons)
# 
# # K-Means
# km_moons = KMeans(n_clusters=2, random_state=42)
# labels_moons = km_moons.fit_predict(X_moons_scaled)
# 
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 
# axes[0].scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=y_moons, cmap='viridis')
# axes[0].set_title('Ground Truth (Moons)')
# 
# axes[1].scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=labels_moons, cmap='viridis')
# axes[1].scatter(km_moons.cluster_centers_[:, 0], km_moons.cluster_centers_[:, 1],
#                 c='red', marker='X', s=200)
# axes[1].set_title('K-Means (FALLA con formas no esféricas)')
# 
# plt.tight_layout()
# plt.show()
# 
# print('K-Means asume clusters esféricos, por eso falla con moons.')
# print('Solución: usar DBSCAN (próximo ejercicio)')

print()
print('=== Ejercicio completado ===')
