"""
Ejercicio 02: t-SNE para Visualización
======================================
Aprende a usar t-SNE para visualización de datos.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, trustworthiness

# ============================================
# PASO 1: Generar Datos de Alta Dimensión
# ============================================
print('--- Paso 1: Generar Datos ---')

# Descomenta las siguientes líneas:
# # Crear datos con clusters en 50 dimensiones
# np.random.seed(42)
# X, y = make_blobs(n_samples=500, n_features=50, centers=5, 
#                   cluster_std=2.0, random_state=42)
# 
# X_scaled = StandardScaler().fit_transform(X)
# 
# print(f'Shape: {X_scaled.shape}')
# print(f'Clases: {np.unique(y)}')

print()


# ============================================
# PASO 2: t-SNE Básico
# ============================================
print('--- Paso 2: t-SNE Básico ---')

# Descomenta las siguientes líneas:
# tsne = TSNE(
#     n_components=2,
#     perplexity=30,       # Vecinos efectivos
#     random_state=42,
#     n_iter=1000
# )
# 
# X_tsne = tsne.fit_transform(X_scaled)
# 
# print(f'Shape t-SNE: {X_tsne.shape}')
# print(f'KL Divergence: {tsne.kl_divergence_:.4f}')
# 
# # Visualizar
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7, s=50)
# plt.colorbar(scatter, label='Cluster')
# plt.xlabel('t-SNE 1')
# plt.ylabel('t-SNE 2')
# plt.title('t-SNE (perplexity=30)')
# plt.show()

print()


# ============================================
# PASO 3: Efecto de Perplexity
# ============================================
print('--- Paso 3: Comparar Perplexity ---')

# Descomenta las siguientes líneas:
# perplexities = [5, 15, 30, 50]
# 
# fig, axes = plt.subplots(2, 2, figsize=(12, 12))
# axes = axes.ravel()
# 
# for ax, perp in zip(axes, perplexities):
#     tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
#     X_embedded = tsne.fit_transform(X_scaled)
#     
#     ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.7, s=30)
#     ax.set_title(f'Perplexity = {perp}\nKL = {tsne.kl_divergence_:.3f}')
#     ax.set_xticks([])
#     ax.set_yticks([])
# 
# plt.suptitle('Efecto de Perplexity en t-SNE', fontsize=14)
# plt.tight_layout()
# plt.show()
# 
# print('Perplexity bajo: clusters más compactos, más local')
# print('Perplexity alto: más estructura global visible')

print()


# ============================================
# PASO 4: Evaluar con Trustworthiness
# ============================================
print('--- Paso 4: Trustworthiness ---')

# Descomenta las siguientes líneas:
# perplexities = [5, 15, 30, 50]
# 
# print('=== Trustworthiness por Perplexity ===')
# for perp in perplexities:
#     tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
#     X_embedded = tsne.fit_transform(X_scaled)
#     
#     trust = trustworthiness(X_scaled, X_embedded, n_neighbors=15)
#     print(f'Perplexity {perp:2d}: Trust = {trust:.4f}')
# 
# print('\nTrustworthiness mide qué tan bien se preservan vecinos cercanos')
# print('Valor cercano a 1.0 es mejor')

print()


# ============================================
# PASO 5: t-SNE en Dataset Real (Digits)
# ============================================
print('--- Paso 5: t-SNE en Digits ---')

# Descomenta las siguientes líneas:
# # Cargar dígitos
# digits = load_digits()
# X_digits = digits.data
# y_digits = digits.target
# 
# print(f'Digits shape: {X_digits.shape}')
# 
# # Escalar
# X_digits_scaled = StandardScaler().fit_transform(X_digits)
# 
# # t-SNE
# tsne_digits = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1500)
# X_digits_tsne = tsne_digits.fit_transform(X_digits_scaled)
# 
# # Visualizar
# plt.figure(figsize=(12, 10))
# scatter = plt.scatter(X_digits_tsne[:, 0], X_digits_tsne[:, 1],
#                       c=y_digits, cmap='tab10', alpha=0.7, s=30)
# plt.colorbar(scatter, label='Dígito')
# plt.xlabel('t-SNE 1')
# plt.ylabel('t-SNE 2')
# plt.title('t-SNE de Dígitos Escritos a Mano (64D → 2D)')
# 
# # Anotar algunos dígitos
# for digit in range(10):
#     idx = np.where(y_digits == digit)[0][0]
#     plt.annotate(str(digit), (X_digits_tsne[idx, 0], X_digits_tsne[idx, 1]),
#                  fontsize=12, fontweight='bold', color='red')
# 
# plt.show()

print()


# ============================================
# PASO 6: Visualización 3D
# ============================================
print('--- Paso 6: t-SNE 3D ---')

# Descomenta las siguientes líneas:
# from mpl_toolkits.mplot3d import Axes3D
# 
# # t-SNE 3D
# tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42)
# X_tsne_3d = tsne_3d.fit_transform(X_digits_scaled)
# 
# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection='3d')
# 
# scatter = ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2],
#                      c=y_digits, cmap='tab10', alpha=0.6, s=20)
# 
# ax.set_xlabel('t-SNE 1')
# ax.set_ylabel('t-SNE 2')
# ax.set_zlabel('t-SNE 3')
# ax.set_title('t-SNE 3D de Dígitos')
# 
# plt.colorbar(scatter, label='Dígito', shrink=0.5)
# plt.show()

print()


# ============================================
# PASO 7: Comparar con PCA
# ============================================
print('--- Paso 7: PCA vs t-SNE ---')

# Descomenta las siguientes líneas:
# from sklearn.decomposition import PCA
# 
# # PCA 2D
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_digits_scaled)
# 
# # Comparar visualmente
# fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# 
# # PCA
# scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='tab10', alpha=0.6, s=20)
# axes[0].set_title(f'PCA (var: {pca.explained_variance_ratio_.sum()*100:.1f}%)')
# axes[0].set_xlabel('PC1')
# axes[0].set_ylabel('PC2')
# 
# # t-SNE
# scatter2 = axes[1].scatter(X_digits_tsne[:, 0], X_digits_tsne[:, 1], c=y_digits, cmap='tab10', alpha=0.6, s=20)
# axes[1].set_title('t-SNE (perplexity=30)')
# axes[1].set_xlabel('t-SNE 1')
# axes[1].set_ylabel('t-SNE 2')
# 
# plt.colorbar(scatter2, ax=axes, label='Dígito', shrink=0.8)
# plt.suptitle('PCA vs t-SNE en Dígitos', fontsize=14)
# plt.tight_layout()
# plt.show()
# 
# # Trustworthiness de ambos
# trust_pca = trustworthiness(X_digits_scaled, X_pca, n_neighbors=15)
# trust_tsne = trustworthiness(X_digits_scaled, X_digits_tsne, n_neighbors=15)
# print(f'Trustworthiness PCA:   {trust_pca:.4f}')
# print(f'Trustworthiness t-SNE: {trust_tsne:.4f}')

print()


# ============================================
# PASO 8: Estabilidad de t-SNE
# ============================================
print('--- Paso 8: Estabilidad (random_state) ---')

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# 
# for ax, seed in zip(axes, [1, 42, 123]):
#     tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
#     X_embedded = tsne.fit_transform(X_scaled)
#     
#     ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.7, s=30)
#     ax.set_title(f'random_state = {seed}')
#     ax.set_xticks([])
#     ax.set_yticks([])
# 
# plt.suptitle('t-SNE: Diferentes Semillas → Diferentes Resultados', fontsize=14)
# plt.tight_layout()
# plt.show()
# 
# print('t-SNE es estocástico: mismo dataset puede dar diferentes visualizaciones')
# print('Siempre usar random_state para reproducibilidad')

print()
print('=== Ejercicio completado ===')
