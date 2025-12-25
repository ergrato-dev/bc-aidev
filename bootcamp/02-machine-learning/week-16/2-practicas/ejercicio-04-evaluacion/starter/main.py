"""
Ejercicio 04: Evaluación de Clustering
======================================
Aprende a evaluar y comparar algoritmos de clustering.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================================
# PASO 1: Generar Datos
# ============================================
print('--- Paso 1: Generar Datos ---')

# Descomenta las siguientes líneas:
# X, y_true = make_blobs(n_samples=400, centers=5, cluster_std=1.0, random_state=42)
# X_scaled = StandardScaler().fit_transform(X)
# 
# plt.figure(figsize=(8, 5))
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.6)
# plt.title('Datos sin etiquetas - ¿Cuántos clusters hay?')
# plt.show()

print()


# ============================================
# PASO 2: Método del Codo (Elbow)
# ============================================
print('--- Paso 2: Método del Codo ---')

# Descomenta las siguientes líneas:
# inertias = []
# K_range = range(1, 11)
# 
# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(X_scaled)
#     inertias.append(kmeans.inertia_)
#     print(f'K={k}: Inercia={kmeans.inertia_:.2f}')
# 
# plt.figure(figsize=(10, 5))
# plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
# plt.xlabel('Número de Clusters (K)')
# plt.ylabel('Inercia (WCSS)')
# plt.title('Método del Codo')
# plt.grid(True, alpha=0.3)
# plt.axvline(x=5, color='r', linestyle='--', label='Codo sugerido K=5')
# plt.legend()
# plt.show()
# 
# print('El "codo" indica donde añadir más clusters deja de mejorar significativamente')

print()


# ============================================
# PASO 3: Silhouette Score
# ============================================
print('--- Paso 3: Silhouette Score ---')

# Descomenta las siguientes líneas:
# from sklearn.metrics import silhouette_score, silhouette_samples
# 
# silhouette_scores = []
# K_range = range(2, 11)  # Silhouette requiere K >= 2
# 
# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(X_scaled)
#     score = silhouette_score(X_scaled, labels)
#     silhouette_scores.append(score)
#     print(f'K={k}: Silhouette={score:.4f}')
# 
# plt.figure(figsize=(10, 5))
# plt.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
# plt.xlabel('Número de Clusters (K)')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score por K (mayor es mejor)')
# plt.grid(True, alpha=0.3)
# 
# best_k = K_range[np.argmax(silhouette_scores)]
# plt.axvline(x=best_k, color='r', linestyle='--', label=f'Mejor K={best_k}')
# plt.legend()
# plt.show()

print()


# ============================================
# PASO 4: Silhouette Plot Detallado
# ============================================
print('--- Paso 4: Silhouette Plot ---')

# Descomenta las siguientes líneas:
# import matplotlib.cm as cm
# 
# def silhouette_plot(X, n_clusters):
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(X)
#     
#     sample_silhouette = silhouette_samples(X, labels)
#     avg_score = silhouette_score(X, labels)
#     
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#     
#     y_lower = 10
#     for i in range(n_clusters):
#         cluster_sil = np.sort(sample_silhouette[labels == i])
#         y_upper = y_lower + len(cluster_sil)
#         
#         color = cm.viridis(float(i) / n_clusters)
#         ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
#                          facecolor=color, alpha=0.7)
#         ax1.text(-0.05, y_lower + 0.5 * len(cluster_sil), str(i))
#         y_lower = y_upper + 10
#     
#     ax1.axvline(x=avg_score, color='red', linestyle='--', 
#                 label=f'Promedio: {avg_score:.3f}')
#     ax1.set_xlabel('Silhouette Coefficient')
#     ax1.set_ylabel('Cluster')
#     ax1.set_title(f'Silhouette Plot (K={n_clusters})')
#     ax1.legend()
#     
#     scatter = ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
#     ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#                 c='red', marker='X', s=200)
#     ax2.set_title(f'Clusters (K={n_clusters})')
#     
#     plt.tight_layout()
#     plt.show()
# 
# # Probar con diferentes K
# for k in [3, 5, 7]:
#     silhouette_plot(X_scaled, k)

print()


# ============================================
# PASO 5: Otras Métricas
# ============================================
print('--- Paso 5: Davies-Bouldin y Calinski-Harabasz ---')

# Descomenta las siguientes líneas:
# from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
# 
# db_scores = []
# ch_scores = []
# K_range = range(2, 11)
# 
# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(X_scaled)
#     
#     db_scores.append(davies_bouldin_score(X_scaled, labels))
#     ch_scores.append(calinski_harabasz_score(X_scaled, labels))
# 
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 
# # Davies-Bouldin (menor es mejor)
# axes[0].plot(K_range, db_scores, 'ro-', linewidth=2, markersize=8)
# axes[0].set_xlabel('K')
# axes[0].set_ylabel('Davies-Bouldin Index')
# axes[0].set_title('Davies-Bouldin (menor es mejor)')
# best_k_db = K_range[np.argmin(db_scores)]
# axes[0].axvline(x=best_k_db, color='blue', linestyle='--', label=f'Mejor K={best_k_db}')
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)
# 
# # Calinski-Harabasz (mayor es mejor)
# axes[1].plot(K_range, ch_scores, 'bo-', linewidth=2, markersize=8)
# axes[1].set_xlabel('K')
# axes[1].set_ylabel('Calinski-Harabasz Index')
# axes[1].set_title('Calinski-Harabasz (mayor es mejor)')
# best_k_ch = K_range[np.argmax(ch_scores)]
# axes[1].axvline(x=best_k_ch, color='red', linestyle='--', label=f'Mejor K={best_k_ch}')
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)
# 
# plt.tight_layout()
# plt.show()

print()


# ============================================
# PASO 6: Comparar Todos los Métodos
# ============================================
print('--- Paso 6: Resumen de Métricas ---')

# Descomenta las siguientes líneas:
# import pandas as pd
# 
# results = []
# K_range = range(2, 11)
# 
# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(X_scaled)
#     
#     results.append({
#         'K': k,
#         'Inercia': kmeans.inertia_,
#         'Silhouette': silhouette_score(X_scaled, labels),
#         'Davies-Bouldin': davies_bouldin_score(X_scaled, labels),
#         'Calinski-Harabasz': calinski_harabasz_score(X_scaled, labels)
#     })
# 
# df = pd.DataFrame(results)
# print('\n=== Resumen de Métricas ===')
# print(df.to_string(index=False))
# 
# # Mejor K según cada métrica
# print('\n=== K Óptimo por Métrica ===')
# print(f'Silhouette (max):      K = {df.loc[df["Silhouette"].idxmax(), "K"]}')
# print(f'Davies-Bouldin (min):  K = {df.loc[df["Davies-Bouldin"].idxmin(), "K"]}')
# print(f'Calinski-Harabasz (max): K = {df.loc[df["Calinski-Harabasz"].idxmax(), "K"]}')

print()


# ============================================
# PASO 7: Comparar Algoritmos
# ============================================
print('--- Paso 7: Comparar Algoritmos ---')

# Descomenta las siguientes líneas:
# from sklearn.cluster import DBSCAN, AgglomerativeClustering
# 
# # K-Means
# km = KMeans(n_clusters=5, random_state=42)
# labels_km = km.fit_predict(X_scaled)
# 
# # Aglomerativo
# agg = AgglomerativeClustering(n_clusters=5, linkage='ward')
# labels_agg = agg.fit_predict(X_scaled)
# 
# # DBSCAN
# db = DBSCAN(eps=0.4, min_samples=5)
# labels_db = db.fit_predict(X_scaled)
# 
# # Evaluar
# def evaluate(X, labels, name):
#     mask = labels >= 0
#     n_clusters = len(set(labels[mask]))
#     if n_clusters > 1:
#         sil = silhouette_score(X[mask], labels[mask])
#         db = davies_bouldin_score(X[mask], labels[mask])
#         print(f'{name}: Clusters={n_clusters}, Silhouette={sil:.4f}, DB={db:.4f}')
#     else:
#         print(f'{name}: Solo {n_clusters} cluster')
# 
# evaluate(X_scaled, labels_km, 'K-Means')
# evaluate(X_scaled, labels_agg, 'Aglomerativo')
# evaluate(X_scaled, labels_db, 'DBSCAN')
# 
# # Visualizar
# fig, axes = plt.subplots(1, 4, figsize=(18, 4))
# 
# axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis')
# axes[0].set_title('Ground Truth')
# 
# axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_km, cmap='viridis')
# axes[1].set_title('K-Means')
# 
# axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_agg, cmap='viridis')
# axes[2].set_title('Aglomerativo')
# 
# axes[3].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_db, cmap='viridis')
# axes[3].set_title('DBSCAN')
# 
# plt.tight_layout()
# plt.show()

print()
print('=== Ejercicio completado ===')
