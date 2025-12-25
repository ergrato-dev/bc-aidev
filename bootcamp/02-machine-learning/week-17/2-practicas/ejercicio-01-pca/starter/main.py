"""
Ejercicio 01: PCA desde Cero y Sklearn
======================================
Implementa PCA manualmente y compara con sklearn.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler

# ============================================
# PASO 1: Cargar y Preparar Datos
# ============================================
print('--- Paso 1: Cargar Datos ---')

# Descomenta las siguientes líneas:
# iris = load_iris()
# X = iris.data
# y = iris.target
# feature_names = iris.feature_names
# 
# print(f'Shape original: {X.shape}')
# print(f'Features: {feature_names}')
# 
# # Escalar datos (importante para PCA)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# print(f'Media después de escalar: {X_scaled.mean(axis=0).round(6)}')
# print(f'Std después de escalar: {X_scaled.std(axis=0).round(2)}')

print()


# ============================================
# PASO 2: PCA desde Cero
# ============================================
print('--- Paso 2: PCA Manual con NumPy ---')

# Descomenta las siguientes líneas:
# def pca_from_scratch(X, n_components):
#     """Implementación de PCA desde cero."""
#     # Paso 1: Centrar datos (ya escalados, pero asegurar)
#     X_centered = X - X.mean(axis=0)
#     
#     # Paso 2: Matriz de covarianza
#     cov_matrix = np.cov(X_centered.T)
#     print(f'Matriz de covarianza shape: {cov_matrix.shape}')
#     
#     # Paso 3: Autovalores y autovectores
#     eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
#     
#     # Convertir a real (pueden ser complejos por errores numéricos)
#     eigenvalues = eigenvalues.real
#     eigenvectors = eigenvectors.real
#     
#     # Paso 4: Ordenar por autovalor descendente
#     idx = np.argsort(eigenvalues)[::-1]
#     eigenvalues = eigenvalues[idx]
#     eigenvectors = eigenvectors[:, idx]
#     
#     print(f'Autovalores: {eigenvalues.round(3)}')
#     
#     # Calcular varianza explicada
#     total_var = eigenvalues.sum()
#     var_ratio = eigenvalues / total_var
#     print(f'Varianza explicada: {(var_ratio * 100).round(2)}%')
#     
#     # Paso 5: Seleccionar top-k componentes
#     components = eigenvectors[:, :n_components]
#     
#     # Paso 6: Proyectar
#     X_pca = X_centered @ components
#     
#     return X_pca, components, eigenvalues, var_ratio
# 
# # Aplicar PCA manual
# X_pca_manual, components, eigenvalues, var_ratio = pca_from_scratch(X_scaled, n_components=2)
# print(f'\nShape después de PCA: {X_pca_manual.shape}')
# print(f'Varianza total explicada (2 comp): {var_ratio[:2].sum()*100:.2f}%')

print()


# ============================================
# PASO 3: PCA con Sklearn
# ============================================
print('--- Paso 3: PCA con Sklearn ---')

# Descomenta las siguientes líneas:
# from sklearn.decomposition import PCA
# 
# # PCA con sklearn
# pca = PCA(n_components=2)
# X_pca_sklearn = pca.fit_transform(X_scaled)
# 
# print(f'Shape sklearn: {X_pca_sklearn.shape}')
# print(f'Varianza explicada sklearn: {pca.explained_variance_ratio_.round(4)}')
# print(f'Total varianza: {pca.explained_variance_ratio_.sum()*100:.2f}%')
# 
# # Comparar resultados
# print(f'\n¿Resultados similares? (puede diferir en signo)')
# print(f'Manual PC1 rango: [{X_pca_manual[:, 0].min():.2f}, {X_pca_manual[:, 0].max():.2f}]')
# print(f'Sklearn PC1 rango: [{X_pca_sklearn[:, 0].min():.2f}, {X_pca_sklearn[:, 0].max():.2f}]')

print()


# ============================================
# PASO 4: Visualizar Componentes
# ============================================
print('--- Paso 4: Visualización ---')

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 
# # PCA Manual
# scatter1 = axes[0].scatter(X_pca_manual[:, 0], X_pca_manual[:, 1], 
#                            c=y, cmap='viridis', alpha=0.7, s=50)
# axes[0].set_xlabel('PC1')
# axes[0].set_ylabel('PC2')
# axes[0].set_title('PCA Manual (NumPy)')
# plt.colorbar(scatter1, ax=axes[0], label='Clase')
# 
# # PCA Sklearn
# scatter2 = axes[1].scatter(X_pca_sklearn[:, 0], X_pca_sklearn[:, 1],
#                            c=y, cmap='viridis', alpha=0.7, s=50)
# axes[1].set_xlabel('PC1')
# axes[1].set_ylabel('PC2')
# axes[1].set_title('PCA Sklearn')
# plt.colorbar(scatter2, ax=axes[1], label='Clase')
# 
# plt.tight_layout()
# plt.show()

print()


# ============================================
# PASO 5: Scree Plot
# ============================================
print('--- Paso 5: Scree Plot ---')

# Descomenta las siguientes líneas:
# # PCA completo para scree plot
# pca_full = PCA()
# pca_full.fit(X_scaled)
# 
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 
# # Varianza individual
# axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
#             pca_full.explained_variance_ratio_ * 100, 
#             color='steelblue', alpha=0.7)
# axes[0].set_xlabel('Componente Principal')
# axes[0].set_ylabel('Varianza Explicada (%)')
# axes[0].set_title('Varianza por Componente')
# 
# # Varianza acumulada
# cumsum = np.cumsum(pca_full.explained_variance_ratio_) * 100
# axes[1].plot(range(1, len(cumsum) + 1), cumsum, 'go-', linewidth=2, markersize=8)
# axes[1].axhline(y=95, color='r', linestyle='--', label='95% varianza')
# axes[1].set_xlabel('Número de Componentes')
# axes[1].set_ylabel('Varianza Acumulada (%)')
# axes[1].set_title('Varianza Acumulada')
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)
# 
# plt.tight_layout()
# plt.show()
# 
# # Encontrar componentes para 95%
# n_95 = np.argmax(cumsum >= 95) + 1
# print(f'Componentes para 95% varianza: {n_95}')

print()


# ============================================
# PASO 6: Interpretación de Componentes
# ============================================
print('--- Paso 6: Interpretación ---')

# Descomenta las siguientes líneas:
# print('=== Contribución de Features a cada PC ===')
# for i, component in enumerate(pca.components_):
#     print(f'\nPC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}% var):')
#     for name, weight in sorted(zip(feature_names, component), key=lambda x: abs(x[1]), reverse=True):
#         print(f'  {name}: {weight:.3f}')

print()


# ============================================
# PASO 7: Reconstrucción de Datos
# ============================================
print('--- Paso 7: Reconstrucción ---')

# Descomenta las siguientes líneas:
# # Reconstruir datos desde 2 componentes
# X_reconstructed = pca.inverse_transform(X_pca_sklearn)
# 
# # Error de reconstrucción
# mse = np.mean((X_scaled - X_reconstructed) ** 2)
# print(f'Error de reconstrucción (MSE): {mse:.4f}')
# 
# # Comparar con más componentes
# for n in [1, 2, 3, 4]:
#     pca_n = PCA(n_components=n)
#     X_reduced = pca_n.fit_transform(X_scaled)
#     X_recon = pca_n.inverse_transform(X_reduced)
#     mse_n = np.mean((X_scaled - X_recon) ** 2)
#     var_n = pca_n.explained_variance_ratio_.sum() * 100
#     print(f'n={n}: MSE={mse_n:.4f}, Var={var_n:.1f}%')

print()


# ============================================
# PASO 8: PCA en Dataset Más Grande
# ============================================
print('--- Paso 8: PCA en Digits ---')

# Descomenta las siguientes líneas:
# # Cargar digits (64 dimensiones)
# digits = load_digits()
# X_digits = digits.data
# y_digits = digits.target
# 
# print(f'Digits shape: {X_digits.shape}')  # 8x8 pixels = 64 features
# 
# # Escalar
# X_digits_scaled = StandardScaler().fit_transform(X_digits)
# 
# # PCA para encontrar componentes óptimos
# pca_digits = PCA()
# pca_digits.fit(X_digits_scaled)
# 
# cumsum_digits = np.cumsum(pca_digits.explained_variance_ratio_) * 100
# n_95_digits = np.argmax(cumsum_digits >= 95) + 1
# print(f'Componentes para 95% varianza: {n_95_digits} de 64')
# 
# # Visualizar en 2D
# pca_2d = PCA(n_components=2)
# X_digits_2d = pca_2d.fit_transform(X_digits_scaled)
# 
# plt.figure(figsize=(12, 10))
# scatter = plt.scatter(X_digits_2d[:, 0], X_digits_2d[:, 1],
#                       c=y_digits, cmap='tab10', alpha=0.6, s=20)
# plt.colorbar(scatter, label='Dígito')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title(f'PCA de Dígitos (64D → 2D, {pca_2d.explained_variance_ratio_.sum()*100:.1f}% var)')
# plt.show()

print()
print('=== Ejercicio completado ===')
