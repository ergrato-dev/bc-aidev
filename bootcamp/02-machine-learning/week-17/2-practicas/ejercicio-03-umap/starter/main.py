"""
Ejercicio 03: UMAP y Comparaciones
==================================
Aprende UMAP y compara con otras técnicas.

Requiere: pip install umap-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
import time

# ============================================
# PASO 1: Instalar e Importar UMAP
# ============================================
print('--- Paso 1: Importar UMAP ---')

# Descomenta las siguientes líneas:
# try:
#     import umap
#     print('UMAP importado correctamente')
# except ImportError:
#     print('Instala UMAP con: pip install umap-learn')
#     raise

print()


# ============================================
# PASO 2: Cargar Datos
# ============================================
print('--- Paso 2: Cargar Datos ---')

# Descomenta las siguientes líneas:
# digits = load_digits()
# X = digits.data
# y = digits.target
# 
# X_scaled = StandardScaler().fit_transform(X)
# 
# print(f'Shape: {X_scaled.shape}')
# print(f'Clases: {np.unique(y)}')

print()


# ============================================
# PASO 3: UMAP Básico
# ============================================
print('--- Paso 3: UMAP Básico ---')

# Descomenta las siguientes líneas:
# reducer = umap.UMAP(
#     n_components=2,
#     n_neighbors=15,      # Vecinos para estructura local
#     min_dist=0.1,        # Distancia mínima entre puntos
#     metric='euclidean',
#     random_state=42
# )
# 
# start = time.time()
# X_umap = reducer.fit_transform(X_scaled)
# elapsed = time.time() - start
# 
# print(f'Tiempo UMAP: {elapsed:.2f}s')
# print(f'Shape: {X_umap.shape}')
# 
# # Visualizar
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.7, s=30)
# plt.colorbar(scatter, label='Dígito')
# plt.xlabel('UMAP 1')
# plt.ylabel('UMAP 2')
# plt.title('UMAP de Dígitos')
# plt.show()

print()


# ============================================
# PASO 4: Efecto de n_neighbors
# ============================================
print('--- Paso 4: Efecto de n_neighbors ---')

# Descomenta las siguientes líneas:
# n_neighbors_list = [5, 15, 50, 100]
# 
# fig, axes = plt.subplots(2, 2, figsize=(12, 12))
# axes = axes.ravel()
# 
# for ax, nn in zip(axes, n_neighbors_list):
#     reducer = umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=42)
#     X_embedded = reducer.fit_transform(X_scaled)
#     
#     ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='tab10', alpha=0.6, s=15)
#     ax.set_title(f'n_neighbors = {nn}')
#     ax.set_xticks([])
#     ax.set_yticks([])
# 
# plt.suptitle('Efecto de n_neighbors en UMAP', fontsize=14)
# plt.tight_layout()
# plt.show()
# 
# print('n_neighbors bajo: más énfasis en estructura local')
# print('n_neighbors alto: más énfasis en estructura global')

print()


# ============================================
# PASO 5: Efecto de min_dist
# ============================================
print('--- Paso 5: Efecto de min_dist ---')

# Descomenta las siguientes líneas:
# min_dist_list = [0.0, 0.1, 0.5, 0.99]
# 
# fig, axes = plt.subplots(2, 2, figsize=(12, 12))
# axes = axes.ravel()
# 
# for ax, md in zip(axes, min_dist_list):
#     reducer = umap.UMAP(n_neighbors=15, min_dist=md, random_state=42)
#     X_embedded = reducer.fit_transform(X_scaled)
#     
#     ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='tab10', alpha=0.6, s=15)
#     ax.set_title(f'min_dist = {md}')
#     ax.set_xticks([])
#     ax.set_yticks([])
# 
# plt.suptitle('Efecto de min_dist en UMAP', fontsize=14)
# plt.tight_layout()
# plt.show()
# 
# print('min_dist bajo: clusters más compactos')
# print('min_dist alto: puntos más dispersos')

print()


# ============================================
# PASO 6: Comparación PCA vs t-SNE vs UMAP
# ============================================
print('--- Paso 6: Comparación de Métodos ---')

# Descomenta las siguientes líneas:
# methods = {
#     'PCA': PCA(n_components=2),
#     't-SNE': TSNE(n_components=2, perplexity=30, random_state=42),
#     'UMAP': umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
# }
# 
# fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# results = {}
# 
# for ax, (name, method) in zip(axes, methods.items()):
#     start = time.time()
#     X_embedded = method.fit_transform(X_scaled)
#     elapsed = time.time() - start
#     
#     trust = trustworthiness(X_scaled, X_embedded, n_neighbors=15)
#     
#     results[name] = {'time': elapsed, 'trust': trust, 'embedding': X_embedded}
#     
#     ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='tab10', alpha=0.6, s=15)
#     ax.set_title(f'{name}\nTime: {elapsed:.2f}s, Trust: {trust:.3f}')
#     ax.set_xticks([])
#     ax.set_yticks([])
# 
# plt.suptitle('Comparación: PCA vs t-SNE vs UMAP', fontsize=14)
# plt.tight_layout()
# plt.show()
# 
# print('\n=== Resumen ===')
# for name, res in results.items():
#     print(f'{name}: Tiempo={res["time"]:.2f}s, Trustworthiness={res["trust"]:.4f}')

print()


# ============================================
# PASO 7: Transform para Nuevos Datos
# ============================================
print('--- Paso 7: Transform Nuevos Datos ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import train_test_split
# 
# # Dividir datos
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# 
# # Entrenar UMAP solo en train
# reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
# reducer.fit(X_train)
# 
# # Transformar train y test
# X_train_umap = reducer.transform(X_train)
# X_test_umap = reducer.transform(X_test)  # ¡Nuevos datos!
# 
# # Visualizar
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 
# axes[0].scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap='tab10', alpha=0.6, s=20)
# axes[0].set_title('Train Set')
# 
# axes[1].scatter(X_test_umap[:, 0], X_test_umap[:, 1], c=y_test, cmap='tab10', alpha=0.6, s=20)
# axes[1].set_title('Test Set (nuevos datos transformados)')
# 
# plt.suptitle('UMAP: Transform para Nuevos Datos', fontsize=14)
# plt.tight_layout()
# plt.show()
# 
# print('UMAP puede transformar nuevos datos (a diferencia de t-SNE)')

print()


# ============================================
# PASO 8: UMAP Supervisado
# ============================================
print('--- Paso 8: UMAP Supervisado ---')

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 
# # UMAP no supervisado
# reducer_unsup = umap.UMAP(n_components=2, random_state=42)
# X_unsup = reducer_unsup.fit_transform(X_scaled)
# 
# axes[0].scatter(X_unsup[:, 0], X_unsup[:, 1], c=y, cmap='tab10', alpha=0.6, s=15)
# axes[0].set_title('UMAP No Supervisado')
# 
# # UMAP supervisado (usa etiquetas)
# reducer_sup = umap.UMAP(n_components=2, random_state=42)
# X_sup = reducer_sup.fit_transform(X_scaled, y=y)
# 
# axes[1].scatter(X_sup[:, 0], X_sup[:, 1], c=y, cmap='tab10', alpha=0.6, s=15)
# axes[1].set_title('UMAP Supervisado (usa etiquetas)')
# 
# plt.suptitle('UMAP: No Supervisado vs Supervisado', fontsize=14)
# plt.tight_layout()
# plt.show()
# 
# # Comparar trustworthiness
# trust_unsup = trustworthiness(X_scaled, X_unsup, n_neighbors=15)
# trust_sup = trustworthiness(X_scaled, X_sup, n_neighbors=15)
# print(f'Trust no supervisado: {trust_unsup:.4f}')
# print(f'Trust supervisado: {trust_sup:.4f}')
# print('\nEl supervisado separa mejor las clases usando las etiquetas')

print()
print('=== Ejercicio completado ===')
