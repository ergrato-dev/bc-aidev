# ============================================
# EJERCICIO 04: Evaluación de Clustering
# ============================================
# Métricas y técnicas para evaluar calidad de clusters
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

print("=" * 60)
print("EJERCICIO 04: Evaluación de Clustering")
print("=" * 60)

# ============================================
# PASO 1: Generar Datos de Prueba
# ============================================
print("\n--- Paso 1: Generar Datos ---")

# Descomenta las siguientes líneas:
# np.random.seed(42)
#
# # Dataset 1: Clusters bien separados
# X_good, y_good = make_blobs(n_samples=300, centers=4, 
#                             cluster_std=0.5, random_state=42)
# print(f"Dataset bueno: {X_good.shape}")
#
# # Dataset 2: Clusters superpuestos
# X_overlap, y_overlap = make_blobs(n_samples=300, centers=4, 
#                                   cluster_std=2.0, random_state=42)
# print(f"Dataset superpuesto: {X_overlap.shape}")
#
# # Dataset 3: Clusters complejos (moons)
# X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
# print(f"Dataset moons: {X_moons.shape}")
#
# # Visualizar los tres datasets
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#
# axes[0].scatter(X_good[:, 0], X_good[:, 1], c=y_good, cmap='viridis', alpha=0.6)
# axes[0].set_title('Clusters Bien Separados')
#
# axes[1].scatter(X_overlap[:, 0], X_overlap[:, 1], c=y_overlap, cmap='viridis', alpha=0.6)
# axes[1].set_title('Clusters Superpuestos')
#
# axes[2].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', alpha=0.6)
# axes[2].set_title('Clusters No Convexos (Moons)')
#
# for ax in axes:
#     ax.set_xlabel('Feature 1')
#     ax.set_ylabel('Feature 2')
#
# plt.tight_layout()
# plt.savefig('evaluation_datasets.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: evaluation_datasets.png")

print()

# ============================================
# PASO 2: Silhouette Score
# ============================================
print("--- Paso 2: Silhouette Score ---")

# Descomenta las siguientes líneas:
# # K-Means en dataset bueno
# kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
# labels_good = kmeans.fit_predict(X_good)
#
# # Calcular silhouette score
# sil_score = silhouette_score(X_good, labels_good)
# print(f"Silhouette Score (clusters buenos): {sil_score:.3f}")
#
# # K-Means en dataset superpuesto
# labels_overlap = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_overlap)
# sil_overlap = silhouette_score(X_overlap, labels_overlap)
# print(f"Silhouette Score (superpuestos): {sil_overlap:.3f}")
#
# # Interpretación
# print("\nInterpretación del Silhouette Score:")
# print("  > 0.7: Estructura fuerte de clusters")
# print("  0.5 - 0.7: Estructura razonable")
# print("  0.25 - 0.5: Estructura débil")
# print("  < 0.25: Sin estructura clara")

print()

# ============================================
# PASO 3: Silhouette Plot por Cluster
# ============================================
print("--- Paso 3: Silhouette Plot ---")

# Descomenta las siguientes líneas:
# def silhouette_plot(X: np.ndarray, labels: np.ndarray, 
#                     title: str, save_name: str) -> None:
#     """
#     Create a silhouette plot showing score distribution per cluster.
#     
#     Args:
#         X: Data matrix
#         labels: Cluster labels
#         title: Plot title
#         save_name: Filename to save
#     """
#     sample_scores = silhouette_samples(X, labels)
#     n_clusters = len(np.unique(labels[labels >= 0]))  # Excluir noise (-1)
#     avg_score = silhouette_score(X, labels)
#     
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#     
#     # Silhouette plot
#     y_lower = 10
#     colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
#     
#     for i in range(n_clusters):
#         cluster_scores = sample_scores[labels == i]
#         cluster_scores.sort()
#         
#         cluster_size = len(cluster_scores)
#         y_upper = y_lower + cluster_size
#         
#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                          0, cluster_scores,
#                          facecolor=colors[i], alpha=0.7)
#         ax1.text(-0.05, y_lower + 0.5 * cluster_size, str(i))
#         
#         y_lower = y_upper + 10
#     
#     ax1.axvline(x=avg_score, color='red', linestyle='--',
#                label=f'Score promedio: {avg_score:.3f}')
#     ax1.set_title(f'Silhouette Plot\n{title}')
#     ax1.set_xlabel('Silhouette Score')
#     ax1.set_ylabel('Cluster')
#     ax1.legend()
#     ax1.set_xlim([-0.1, 1])
#     
#     # Scatter plot con colores
#     scatter = ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
#     ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#                c='red', marker='X', s=200, edgecolors='black')
#     ax2.set_title(f'Clusters (Silhouette: {avg_score:.3f})')
#     ax2.set_xlabel('Feature 1')
#     ax2.set_ylabel('Feature 2')
#     plt.colorbar(scatter, ax=ax2, label='Cluster')
#     
#     plt.tight_layout()
#     plt.savefig(save_name, dpi=150)
#     plt.show()
#
# # Crear silhouette plots
# silhouette_plot(X_good, labels_good, 
#                'Clusters Bien Separados', 'silhouette_good.png')
# print("✓ Gráfico guardado: silhouette_good.png")
#
# silhouette_plot(X_overlap, labels_overlap, 
#                'Clusters Superpuestos', 'silhouette_overlap.png')
# print("✓ Gráfico guardado: silhouette_overlap.png")

print()

# ============================================
# PASO 4: Encontrar K Óptimo con Silhouette
# ============================================
print("--- Paso 4: K Óptimo con Silhouette ---")

# Descomenta las siguientes líneas:
# def find_optimal_k_silhouette(X: np.ndarray, k_range: range) -> tuple:
#     """
#     Find optimal K using silhouette score.
#     
#     Returns:
#         best_k: Optimal number of clusters
#         scores: List of scores for each K
#     """
#     scores = []
#     print("Calculando silhouette para diferentes K:")
#     
#     for k in k_range:
#         kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#         labels = kmeans.fit_predict(X)
#         score = silhouette_score(X, labels)
#         scores.append(score)
#         print(f"  K={k}: Silhouette = {score:.3f}")
#     
#     best_k = k_range[np.argmax(scores)]
#     return best_k, scores
#
# # Encontrar K óptimo
# k_range = range(2, 10)
# best_k, sil_scores = find_optimal_k_silhouette(X_good, k_range)
# print(f"\n✓ K óptimo por Silhouette: {best_k}")
#
# # Graficar
# plt.figure(figsize=(10, 6))
# plt.plot(list(k_range), sil_scores, 'bo-', linewidth=2, markersize=8)
# plt.axvline(x=best_k, color='r', linestyle='--', label=f'K óptimo = {best_k}')
# plt.xlabel('Número de Clusters (K)', fontsize=12)
# plt.ylabel('Silhouette Score', fontsize=12)
# plt.title('Método del Silhouette para Selección de K', fontsize=14)
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.xticks(list(k_range))
# plt.tight_layout()
# plt.savefig('silhouette_optimal_k.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: silhouette_optimal_k.png")

print()

# ============================================
# PASO 5: Davies-Bouldin Index
# ============================================
print("--- Paso 5: Davies-Bouldin Index ---")

# Descomenta las siguientes líneas:
# # Calcular DBI para diferentes K
# db_scores = []
# print("Calculando Davies-Bouldin para diferentes K:")
#
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(X_good)
#     dbi = davies_bouldin_score(X_good, labels)
#     db_scores.append(dbi)
#     print(f"  K={k}: DBI = {dbi:.3f}")
#
# best_k_dbi = k_range[np.argmin(db_scores)]
# print(f"\n✓ K óptimo por Davies-Bouldin: {best_k_dbi}")
#
# # Graficar DBI
# plt.figure(figsize=(10, 6))
# plt.plot(list(k_range), db_scores, 'go-', linewidth=2, markersize=8)
# plt.axvline(x=best_k_dbi, color='r', linestyle='--', label=f'K óptimo = {best_k_dbi}')
# plt.xlabel('Número de Clusters (K)', fontsize=12)
# plt.ylabel('Davies-Bouldin Index (menor = mejor)', fontsize=12)
# plt.title('Davies-Bouldin Index para Selección de K', fontsize=14)
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.xticks(list(k_range))
# plt.tight_layout()
# plt.savefig('davies_bouldin_k.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: davies_bouldin_k.png")

print()

# ============================================
# PASO 6: Calinski-Harabasz Index
# ============================================
print("--- Paso 6: Calinski-Harabasz Index ---")

# Descomenta las siguientes líneas:
# # Calcular CH para diferentes K
# ch_scores = []
# print("Calculando Calinski-Harabasz para diferentes K:")
#
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(X_good)
#     ch = calinski_harabasz_score(X_good, labels)
#     ch_scores.append(ch)
#     print(f"  K={k}: CH = {ch:.1f}")
#
# best_k_ch = k_range[np.argmax(ch_scores)]
# print(f"\n✓ K óptimo por Calinski-Harabasz: {best_k_ch}")
#
# # Graficar CH
# plt.figure(figsize=(10, 6))
# plt.plot(list(k_range), ch_scores, 'mo-', linewidth=2, markersize=8)
# plt.axvline(x=best_k_ch, color='r', linestyle='--', label=f'K óptimo = {best_k_ch}')
# plt.xlabel('Número de Clusters (K)', fontsize=12)
# plt.ylabel('Calinski-Harabasz Index (mayor = mejor)', fontsize=12)
# plt.title('Calinski-Harabasz Index para Selección de K', fontsize=14)
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.xticks(list(k_range))
# plt.tight_layout()
# plt.savefig('calinski_harabasz_k.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: calinski_harabasz_k.png")

print()

# ============================================
# PASO 7: Comparar Todas las Métricas
# ============================================
print("--- Paso 7: Comparar Métricas ---")

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 3, figsize=(16, 4))
#
# # Silhouette
# axes[0].plot(list(k_range), sil_scores, 'bo-', linewidth=2)
# axes[0].axvline(x=best_k, color='r', linestyle='--')
# axes[0].set_title(f'Silhouette (K óptimo: {best_k})')
# axes[0].set_xlabel('K')
# axes[0].set_ylabel('Score (mayor = mejor)')
#
# # Davies-Bouldin
# axes[1].plot(list(k_range), db_scores, 'go-', linewidth=2)
# axes[1].axvline(x=best_k_dbi, color='r', linestyle='--')
# axes[1].set_title(f'Davies-Bouldin (K óptimo: {best_k_dbi})')
# axes[1].set_xlabel('K')
# axes[1].set_ylabel('Score (menor = mejor)')
#
# # Calinski-Harabasz
# axes[2].plot(list(k_range), ch_scores, 'mo-', linewidth=2)
# axes[2].axvline(x=best_k_ch, color='r', linestyle='--')
# axes[2].set_title(f'Calinski-Harabasz (K óptimo: {best_k_ch})')
# axes[2].set_xlabel('K')
# axes[2].set_ylabel('Score (mayor = mejor)')
#
# for ax in axes:
#     ax.grid(True, alpha=0.3)
#     ax.set_xticks(list(k_range))
#
# plt.suptitle('Comparación de Métricas Internas', fontsize=14, fontweight='bold')
# plt.tight_layout()
# plt.savefig('metrics_comparison.png', dpi=150)
# plt.show()
# print("✓ Gráfico guardado: metrics_comparison.png")

print()

# ============================================
# PASO 8: Métricas Externas (Con Ground Truth)
# ============================================
print("--- Paso 8: Métricas Externas ---")

# Descomenta las siguientes líneas:
# def evaluate_external_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
#                               name: str) -> dict:
#     """
#     Calculate all external clustering metrics.
#     
#     Args:
#         y_true: Ground truth labels
#         y_pred: Predicted labels
#         name: Name of the clustering method
#     
#     Returns:
#         Dictionary of metrics
#     """
#     metrics = {
#         'Method': name,
#         'ARI': adjusted_rand_score(y_true, y_pred),
#         'NMI': normalized_mutual_info_score(y_true, y_pred),
#         'Homogeneity': homogeneity_score(y_true, y_pred),
#         'Completeness': completeness_score(y_true, y_pred),
#         'V-Measure': v_measure_score(y_true, y_pred)
#     }
#     return metrics
#
# # Evaluar K-Means
# labels_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_good)
# metrics_kmeans = evaluate_external_metrics(y_good, labels_kmeans, 'K-Means')
#
# print("\nMétricas Externas para K-Means:")
# print(f"{'Métrica':<15} {'Valor':<10} {'Interpretación'}")
# print("-" * 50)
# for metric, value in metrics_kmeans.items():
#     if metric != 'Method':
#         interp = "Excelente" if value > 0.9 else "Bueno" if value > 0.7 else "Regular"
#         print(f"{metric:<15} {value:<10.3f} {interp}")

print()

# ============================================
# PASO 9: Comparar Algoritmos de Clustering
# ============================================
print("--- Paso 9: Comparar Algoritmos ---")

# Descomenta las siguientes líneas:
# def compare_clustering_algorithms(X: np.ndarray, y_true: np.ndarray) -> None:
#     """
#     Compare multiple clustering algorithms using various metrics.
#     """
#     algorithms = [
#         ('K-Means', KMeans(n_clusters=4, random_state=42, n_init=10)),
#         ('Hierarchical', AgglomerativeClustering(n_clusters=4, linkage='ward')),
#         ('DBSCAN', DBSCAN(eps=0.5, min_samples=5))
#     ]
#     
#     results = []
#     
#     print("\nComparando algoritmos de clustering:")
#     print("=" * 80)
#     
#     for name, algo in algorithms:
#         labels = algo.fit_predict(X)
#         
#         # Manejar caso de DBSCAN con ruido (-1)
#         valid_mask = labels >= 0
#         if np.sum(valid_mask) < 2:
#             print(f"{name}: No suficientes puntos válidos")
#             continue
#         
#         # Métricas internas (solo puntos válidos)
#         if len(np.unique(labels[valid_mask])) > 1:
#             sil = silhouette_score(X[valid_mask], labels[valid_mask])
#             dbi = davies_bouldin_score(X[valid_mask], labels[valid_mask])
#             ch = calinski_harabasz_score(X[valid_mask], labels[valid_mask])
#         else:
#             sil, dbi, ch = 0, 0, 0
#         
#         # Métricas externas
#         ari = adjusted_rand_score(y_true, labels)
#         nmi = normalized_mutual_info_score(y_true, labels)
#         
#         n_clusters = len(np.unique(labels[valid_mask]))
#         n_noise = np.sum(labels == -1)
#         
#         results.append({
#             'Method': name,
#             'Clusters': n_clusters,
#             'Noise': n_noise,
#             'Silhouette': sil,
#             'DBI': dbi,
#             'CH': ch,
#             'ARI': ari,
#             'NMI': nmi
#         })
#     
#     # Imprimir resultados
#     print(f"\n{'Método':<15} {'Clusters':<10} {'Noise':<8} {'Silhouette':<12} "
#           f"{'DBI':<10} {'ARI':<10} {'NMI':<10}")
#     print("-" * 85)
#     
#     for r in results:
#         print(f"{r['Method']:<15} {r['Clusters']:<10} {r['Noise']:<8} "
#               f"{r['Silhouette']:<12.3f} {r['DBI']:<10.3f} "
#               f"{r['ARI']:<10.3f} {r['NMI']:<10.3f}")
#     
#     return results
#
# # Comparar en dataset bueno
# print("\n📊 Dataset: Clusters Bien Separados")
# results_good = compare_clustering_algorithms(X_good, y_good)

print()

# ============================================
# PASO 10: Visualización de Comparación
# ============================================
print("--- Paso 10: Visualización Comparativa ---")

# Descomenta las siguientes líneas:
# def visualize_algorithm_comparison(X: np.ndarray, y_true: np.ndarray,
#                                    title: str) -> None:
#     """
#     Visualize clustering results from different algorithms.
#     """
#     algorithms = [
#         ('Ground Truth', None, y_true),
#         ('K-Means', KMeans(n_clusters=4, random_state=42, n_init=10), None),
#         ('Hierarchical', AgglomerativeClustering(n_clusters=4), None),
#         ('DBSCAN', DBSCAN(eps=0.5, min_samples=5), None)
#     ]
#     
#     fig, axes = plt.subplots(1, 4, figsize=(18, 4))
#     
#     for ax, (name, algo, labels) in zip(axes, algorithms):
#         if labels is None:
#             labels = algo.fit_predict(X)
#         
#         # Colorear (ruido en gris)
#         colors = labels.astype(float)
#         scatter = ax.scatter(X[:, 0], X[:, 1], c=colors, cmap='viridis', alpha=0.6)
#         
#         # Marcar ruido si existe
#         noise_mask = labels == -1
#         if np.any(noise_mask):
#             ax.scatter(X[noise_mask, 0], X[noise_mask, 1], 
#                       c='gray', marker='x', s=50, label='Noise')
#             ax.legend()
#         
#         # Calcular métricas
#         if name != 'Ground Truth':
#             valid_mask = labels >= 0
#             if np.sum(valid_mask) > 1 and len(np.unique(labels[valid_mask])) > 1:
#                 sil = silhouette_score(X[valid_mask], labels[valid_mask])
#                 ari = adjusted_rand_score(y_true, labels)
#                 ax.set_title(f'{name}\nSil: {sil:.2f}, ARI: {ari:.2f}')
#             else:
#                 ax.set_title(f'{name}\nN/A')
#         else:
#             ax.set_title(name)
#         
#         ax.set_xlabel('Feature 1')
#         ax.set_ylabel('Feature 2')
#     
#     plt.suptitle(title, fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(f'comparison_{title.lower().replace(" ", "_")}.png', dpi=150)
#     plt.show()
#
# # Comparación visual
# visualize_algorithm_comparison(X_good, y_good, 'Clusters Bien Separados')
# print("✓ Gráfico guardado: comparison_clusters_bien_separados.png")
#
# visualize_algorithm_comparison(X_moons, y_moons, 'Moons Dataset')
# print("✓ Gráfico guardado: comparison_moons_dataset.png")

print()

# ============================================
# PASO 11: Análisis de Robustez
# ============================================
print("--- Paso 11: Análisis de Robustez ---")

# Descomenta las siguientes líneas:
# def analyze_robustness(X: np.ndarray, n_runs: int = 10) -> None:
#     """
#     Analyze how stable are clustering results across runs.
#     """
#     sil_scores = []
#     ari_scores = []
#     
#     print(f"Ejecutando K-Means {n_runs} veces...")
#     
#     # Referencia con seed fijo
#     kmeans_ref = KMeans(n_clusters=4, random_state=42, n_init=10)
#     labels_ref = kmeans_ref.fit_predict(X)
#     
#     for i in range(n_runs):
#         # K-Means con diferentes inicializaciones
#         kmeans = KMeans(n_clusters=4, random_state=i, n_init=1)
#         labels = kmeans.fit_predict(X)
#         
#         sil = silhouette_score(X, labels)
#         ari = adjusted_rand_score(labels_ref, labels)
#         
#         sil_scores.append(sil)
#         ari_scores.append(ari)
#     
#     # Estadísticas
#     print(f"\nEstabilidad de K-Means:")
#     print(f"  Silhouette: {np.mean(sil_scores):.3f} ± {np.std(sil_scores):.3f}")
#     print(f"  ARI (vs ref): {np.mean(ari_scores):.3f} ± {np.std(ari_scores):.3f}")
#     
#     # Visualizar distribución
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#     
#     axes[0].hist(sil_scores, bins=15, edgecolor='black', alpha=0.7)
#     axes[0].axvline(np.mean(sil_scores), color='r', linestyle='--',
#                    label=f'Media: {np.mean(sil_scores):.3f}')
#     axes[0].set_title('Distribución de Silhouette Score')
#     axes[0].set_xlabel('Silhouette Score')
#     axes[0].legend()
#     
#     axes[1].hist(ari_scores, bins=15, edgecolor='black', alpha=0.7)
#     axes[1].axvline(np.mean(ari_scores), color='r', linestyle='--',
#                    label=f'Media: {np.mean(ari_scores):.3f}')
#     axes[1].set_title('Distribución de ARI')
#     axes[1].set_xlabel('ARI')
#     axes[1].legend()
#     
#     plt.tight_layout()
#     plt.savefig('robustness_analysis.png', dpi=150)
#     plt.show()
#     print("✓ Gráfico guardado: robustness_analysis.png")
#
# analyze_robustness(X_good, n_runs=20)

print()

# ============================================
# PASO 12: Resumen de Métricas
# ============================================
print("--- Paso 12: Tabla Resumen de Métricas ---")

# Descomenta las siguientes líneas:
# print("\n" + "=" * 80)
# print("RESUMEN DE MÉTRICAS DE CLUSTERING")
# print("=" * 80)
#
# metrics_summary = """
# ┌─────────────────────┬───────────┬─────────┬────────────────────────────────────┐
# │ Métrica             │ Rango     │ Mejor   │ Descripción                        │
# ├─────────────────────┼───────────┼─────────┼────────────────────────────────────┤
# │ MÉTRICAS INTERNAS (Sin Ground Truth)                                          │
# ├─────────────────────┼───────────┼─────────┼────────────────────────────────────┤
# │ Silhouette Score    │ [-1, 1]   │ Mayor   │ Cohesión vs separación             │
# │ Davies-Bouldin      │ [0, ∞)    │ Menor   │ Ratio dispersión/separación        │
# │ Calinski-Harabasz   │ [0, ∞)    │ Mayor   │ Varianza inter/intra cluster       │
# ├─────────────────────┼───────────┼─────────┼────────────────────────────────────┤
# │ MÉTRICAS EXTERNAS (Con Ground Truth)                                          │
# ├─────────────────────┼───────────┼─────────┼────────────────────────────────────┤
# │ Adjusted Rand Index │ [-1, 1]   │ Mayor   │ Concordancia de pares ajustada     │
# │ Normalized MI       │ [0, 1]    │ Mayor   │ Información mutua normalizada      │
# │ Homogeneity         │ [0, 1]    │ Mayor   │ Pureza de clusters                 │
# │ Completeness        │ [0, 1]    │ Mayor   │ Todos de clase juntos              │
# │ V-Measure           │ [0, 1]    │ Mayor   │ Media armónica Hom+Com             │
# └─────────────────────┴───────────┴─────────┴────────────────────────────────────┘
# """
# print(metrics_summary)

print()

# ============================================
# RESUMEN
# ============================================
print("=" * 60)
print("RESUMEN DEL EJERCICIO")
print("=" * 60)
print("""
En este ejercicio aprendiste:

1. ✓ Calcular e interpretar Silhouette Score
2. ✓ Crear Silhouette Plots por cluster
3. ✓ Usar Silhouette para encontrar K óptimo
4. ✓ Aplicar Davies-Bouldin Index
5. ✓ Aplicar Calinski-Harabasz Index
6. ✓ Comparar métricas internas
7. ✓ Evaluar con métricas externas (ARI, NMI)
8. ✓ Comparar algoritmos objetivamente
9. ✓ Analizar robustez de clustering

Archivos generados:
- evaluation_datasets.png
- silhouette_good.png
- silhouette_overlap.png
- silhouette_optimal_k.png
- davies_bouldin_k.png
- calinski_harabasz_k.png
- metrics_comparison.png
- comparison_*.png
- robustness_analysis.png

Recomendaciones:
• Usar múltiples métricas, no solo una
• Silhouette es la más interpretable
• ARI/NMI solo si hay ground truth
• Considerar robustez (múltiples runs)
• Visualizar siempre los clusters
""")
