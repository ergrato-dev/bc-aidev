# ============================================
# PROYECTO: Segmentación de Clientes - SOLUCIÓN
# ============================================
# Aplicar K-Means, DBSCAN y Jerárquico para
# segmentar clientes de un e-commerce
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy import stats

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("=" * 70)
print("PROYECTO: Segmentación de Clientes para Marketing - SOLUCIÓN")
print("=" * 70)

# ============================================
# FASE 1: PREPARACIÓN DE DATOS
# ============================================
print("\n" + "=" * 70)
print("FASE 1: PREPARACIÓN DE DATOS")
print("=" * 70)

# --------------------------------------------
# 1.1 Generar Dataset Sintético de Clientes
# --------------------------------------------
print("\n--- 1.1 Generación de Dataset ---")

def generate_customer_data(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer data with RFM-like features.
    
    Creates distinct customer segments:
    - Champions: High frequency, high monetary, low recency
    - Loyal: Medium-high frequency, medium monetary
    - At Risk: Low frequency, high recency
    - New: Low tenure, medium values
    - Lost: Very high recency, low frequency
    
    Args:
        n_samples: Total number of customers
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with customer features
    """
    np.random.seed(random_state)
    
    # Distribución de clientes por segmento
    n_champions = int(n_samples * 0.15)
    n_loyal = int(n_samples * 0.25)
    n_at_risk = int(n_samples * 0.20)
    n_new = int(n_samples * 0.25)
    n_lost = n_samples - n_champions - n_loyal - n_at_risk - n_new
    
    segments = []
    
    # Champions: Mejores clientes
    champions = pd.DataFrame({
        'recency': np.random.exponential(10, n_champions) + 5,
        'frequency': np.random.normal(50, 10, n_champions),
        'monetary': np.random.normal(5000, 1000, n_champions),
        'avg_basket': np.random.normal(100, 20, n_champions),
        'purchase_variety': np.random.normal(15, 3, n_champions),
        'tenure': np.random.normal(800, 100, n_champions)
    })
    segments.append(champions)
    
    # Loyal: Clientes fieles
    loyal = pd.DataFrame({
        'recency': np.random.exponential(20, n_loyal) + 10,
        'frequency': np.random.normal(30, 8, n_loyal),
        'monetary': np.random.normal(2500, 600, n_loyal),
        'avg_basket': np.random.normal(80, 15, n_loyal),
        'purchase_variety': np.random.normal(10, 2, n_loyal),
        'tenure': np.random.normal(500, 150, n_loyal)
    })
    segments.append(loyal)
    
    # At Risk: En riesgo de perder
    at_risk = pd.DataFrame({
        'recency': np.random.normal(90, 20, n_at_risk),
        'frequency': np.random.normal(15, 5, n_at_risk),
        'monetary': np.random.normal(1500, 400, n_at_risk),
        'avg_basket': np.random.normal(100, 25, n_at_risk),
        'purchase_variety': np.random.normal(8, 2, n_at_risk),
        'tenure': np.random.normal(600, 200, n_at_risk)
    })
    segments.append(at_risk)
    
    # New: Clientes nuevos
    new = pd.DataFrame({
        'recency': np.random.exponential(15, n_new) + 5,
        'frequency': np.random.exponential(3, n_new) + 1,
        'monetary': np.random.normal(300, 100, n_new),
        'avg_basket': np.random.normal(70, 20, n_new),
        'purchase_variety': np.random.normal(3, 1, n_new),
        'tenure': np.random.exponential(60, n_new) + 10
    })
    segments.append(new)
    
    # Lost: Clientes perdidos
    lost = pd.DataFrame({
        'recency': np.random.normal(200, 50, n_lost),
        'frequency': np.random.exponential(5, n_lost) + 1,
        'monetary': np.random.normal(500, 200, n_lost),
        'avg_basket': np.random.normal(50, 15, n_lost),
        'purchase_variety': np.random.normal(4, 1, n_lost),
        'tenure': np.random.normal(400, 150, n_lost)
    })
    segments.append(lost)
    
    # Combinar y mezclar
    df = pd.concat(segments, ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Asegurar valores positivos
    for col in df.columns:
        df[col] = df[col].clip(lower=1)
    
    # Redondear valores
    df['recency'] = df['recency'].round(0).astype(int)
    df['frequency'] = df['frequency'].round(0).astype(int)
    df['monetary'] = df['monetary'].round(2)
    df['avg_basket'] = df['avg_basket'].round(2)
    df['purchase_variety'] = df['purchase_variety'].round(0).astype(int)
    df['tenure'] = df['tenure'].round(0).astype(int)
    
    return df

# Generar datos
df = generate_customer_data(n_samples=500)
print(f"Dataset generado: {df.shape[0]} clientes, {df.shape[1]} features")
print(f"\nColumnas: {list(df.columns)}")


# --------------------------------------------
# 1.2 Exploración de Datos (EDA)
# --------------------------------------------
print("\n--- 1.2 Exploración de Datos ---")

def exploratory_analysis(df: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis on customer data.
    """
    # Estadísticas descriptivas
    print("\n📊 Estadísticas Descriptivas:")
    print(df.describe().round(2))
    
    # Visualizaciones
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(df.columns):
        axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribución: {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frecuencia')
        
        # Añadir líneas de media y mediana
        axes[idx].axvline(df[col].mean(), color='red', linestyle='--', 
                         label=f'Media: {df[col].mean():.1f}')
        axes[idx].axvline(df[col].median(), color='green', linestyle='-', 
                         label=f'Mediana: {df[col].median():.1f}')
        axes[idx].legend(fontsize=8)
    
    plt.suptitle('Distribución de Variables de Cliente', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Gráfico guardado: eda_distributions.png")
    
    # Matriz de correlación
    plt.figure(figsize=(10, 8))
    correlation = df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, cmap='RdBu_r', 
                center=0, fmt='.2f', square=True)
    plt.title('Matriz de Correlación', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Gráfico guardado: eda_correlation.png")

exploratory_analysis(df)


# --------------------------------------------
# 1.3 Preprocesamiento
# --------------------------------------------
print("\n--- 1.3 Preprocesamiento ---")

def preprocess_data(df: pd.DataFrame, remove_outliers: bool = True,
                    outlier_threshold: float = 3.0) -> tuple:
    """
    Preprocess customer data for clustering.
    """
    df_clean = df.copy()
    
    if remove_outliers:
        # Calcular z-scores
        z_scores = np.abs(stats.zscore(df_clean))
        
        # Crear máscara de outliers
        outlier_mask = (z_scores < outlier_threshold).all(axis=1)
        n_outliers = (~outlier_mask).sum()
        
        df_clean = df_clean[outlier_mask].reset_index(drop=True)
        print(f"Outliers eliminados: {n_outliers} ({n_outliers/len(df)*100:.1f}%)")
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    print(f"Datos después de preprocesamiento: {X_scaled.shape}")
    print(f"Media de features (escaladas): {X_scaled.mean(axis=0).round(4)}")
    print(f"Std de features (escaladas): {X_scaled.std(axis=0).round(4)}")
    
    return X_scaled, scaler, df_clean

X_scaled, scaler, df_clean = preprocess_data(df, remove_outliers=True)
feature_names = df.columns.tolist()


# ============================================
# FASE 2: CLUSTERING
# ============================================
print("\n" + "=" * 70)
print("FASE 2: APLICACIÓN DE ALGORITMOS DE CLUSTERING")
print("=" * 70)

# --------------------------------------------
# 2.1 K-Means Clustering
# --------------------------------------------
print("\n--- 2.1 K-Means ---")

def find_optimal_k(X: np.ndarray, k_range: range) -> tuple:
    """
    Find optimal K using elbow method and silhouette score.
    """
    inertias = []
    silhouettes = []
    
    print("Buscando K óptimo...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X, labels)
        silhouettes.append(sil)
        print(f"  K={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={sil:.3f}")
    
    # Encontrar mejor K por silhouette
    best_k = k_range[np.argmax(silhouettes)]
    
    # Visualizar
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow plot
    axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Número de Clusters (K)', fontsize=12)
    axes[0].set_ylabel('Inercia', fontsize=12)
    axes[0].set_title('Método del Codo', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette plot
    axes[1].plot(list(k_range), silhouettes, 'go-', linewidth=2, markersize=8)
    axes[1].axvline(x=best_k, color='r', linestyle='--', 
                   label=f'K óptimo = {best_k}')
    axes[1].set_xlabel('Número de Clusters (K)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Método del Silhouette', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Selección del Número Óptimo de Clusters', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('elbow_silhouette.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Gráfico guardado: elbow_silhouette.png")
    print(f"✓ K óptimo por Silhouette: {best_k}")
    
    return best_k, inertias, silhouettes


def apply_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> tuple:
    """
    Apply K-Means clustering.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    print(f"\nK-Means con K={n_clusters}:")
    print(f"  Distribución: {np.bincount(labels)}")
    print(f"  Inercia: {kmeans.inertia_:.0f}")
    
    return labels, centroids, kmeans

# Encontrar K óptimo y aplicar K-Means
k_range = range(2, 10)
best_k, inertias, silhouettes = find_optimal_k(X_scaled, k_range)
labels_kmeans, centroids_kmeans, model_kmeans = apply_kmeans(X_scaled, best_k)


# --------------------------------------------
# 2.2 DBSCAN Clustering
# --------------------------------------------
print("\n--- 2.2 DBSCAN ---")

def find_optimal_epsilon(X: np.ndarray, min_samples: int = 5) -> float:
    """
    Find optimal epsilon using k-distance graph.
    """
    # Calcular k-distancias
    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    
    # Ordenar distancias al k-ésimo vecino
    k_distances = np.sort(distances[:, -1])
    
    # Visualizar
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(k_distances)), k_distances, 'b-', linewidth=2)
    plt.xlabel('Puntos (ordenados por distancia)', fontsize=12)
    plt.ylabel(f'{min_samples}-distancia', fontsize=12)
    plt.title(f'K-Distance Graph para Selección de Epsilon\n'
             f'(Buscar el "codo" en la curva)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Sugerir epsilon (aproximación del codo)
    # Usar derivada segunda para encontrar el codo
    gradient = np.gradient(k_distances)
    gradient2 = np.gradient(gradient)
    elbow_idx = np.argmax(gradient2)
    suggested_eps = k_distances[elbow_idx]
    
    plt.axhline(y=suggested_eps, color='r', linestyle='--', 
               label=f'ε sugerido ≈ {suggested_eps:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('k_distance_graph.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Gráfico guardado: k_distance_graph.png")
    print(f"✓ Epsilon sugerido: {suggested_eps:.2f}")
    
    return suggested_eps


def apply_dbscan(X: np.ndarray, eps: float, min_samples: int = 5) -> tuple:
    """
    Apply DBSCAN clustering.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    
    print(f"\nDBSCAN con eps={eps:.2f}, min_samples={min_samples}:")
    print(f"  Clusters encontrados: {n_clusters}")
    print(f"  Puntos de ruido: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    if n_clusters > 0:
        valid_labels = labels[labels >= 0]
        print(f"  Distribución (sin ruido): {np.bincount(valid_labels)}")
    
    return labels, n_clusters, n_noise

# Encontrar epsilon y aplicar DBSCAN
suggested_eps = find_optimal_epsilon(X_scaled, min_samples=5)
# Ajustar epsilon manualmente si es necesario
eps_final = max(0.8, suggested_eps)  # Mínimo 0.8 para datos estandarizados
labels_dbscan, n_clusters_dbscan, n_noise_dbscan = apply_dbscan(X_scaled, eps_final)


# --------------------------------------------
# 2.3 Clustering Jerárquico
# --------------------------------------------
print("\n--- 2.3 Clustering Jerárquico ---")

def create_dendrogram(X: np.ndarray, method: str = 'ward') -> np.ndarray:
    """
    Create dendrogram for hierarchical clustering.
    """
    Z = linkage(X, method=method)
    
    plt.figure(figsize=(14, 6))
    dendrogram(Z, truncate_mode='lastp', p=30, 
               leaf_rotation=90, leaf_font_size=8)
    plt.xlabel('Índice de Muestra (o tamaño de cluster)', fontsize=12)
    plt.ylabel('Distancia', fontsize=12)
    plt.title(f'Dendrograma - Método {method.capitalize()}', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Añadir línea de corte sugerida
    # Buscar el salto más grande en distancias
    distances = Z[:, 2]
    acceleration = np.diff(distances, 2)
    if len(acceleration) > 0:
        cut_idx = np.argmax(acceleration) + 2
        cut_distance = Z[cut_idx, 2]
        plt.axhline(y=cut_distance, color='r', linestyle='--',
                   label=f'Corte sugerido: {cut_distance:.1f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('dendrogram.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Gráfico guardado: dendrogram.png")
    
    return Z


def apply_hierarchical(X: np.ndarray, n_clusters: int, 
                       linkage_method: str = 'ward') -> np.ndarray:
    """
    Apply Agglomerative Clustering.
    """
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = agg.fit_predict(X)
    
    print(f"\nClustering Jerárquico ({linkage_method}) con K={n_clusters}:")
    print(f"  Distribución: {np.bincount(labels)}")
    
    return labels

# Crear dendrograma y aplicar clustering jerárquico
Z = create_dendrogram(X_scaled, method='ward')
labels_hierarchical = apply_hierarchical(X_scaled, n_clusters=best_k)


# ============================================
# FASE 3: EVALUACIÓN Y COMPARACIÓN
# ============================================
print("\n" + "=" * 70)
print("FASE 3: EVALUACIÓN Y COMPARACIÓN")
print("=" * 70)

# --------------------------------------------
# 3.1 Métricas de Evaluación
# --------------------------------------------
print("\n--- 3.1 Métricas de Evaluación ---")

def evaluate_clustering(X: np.ndarray, labels: np.ndarray, 
                        name: str) -> dict:
    """
    Evaluate clustering results with multiple metrics.
    """
    # Filtrar ruido si existe
    valid_mask = labels >= 0
    if valid_mask.sum() < 2 or len(np.unique(labels[valid_mask])) < 2:
        return {
            'Method': name,
            'Clusters': 0,
            'Noise': (labels == -1).sum(),
            'Silhouette': np.nan,
            'Davies-Bouldin': np.nan
        }
    
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    sil = silhouette_score(X_valid, labels_valid)
    dbi = davies_bouldin_score(X_valid, labels_valid)
    
    n_clusters = len(np.unique(labels_valid))
    n_noise = (labels == -1).sum()
    
    return {
        'Method': name,
        'Clusters': n_clusters,
        'Noise': n_noise,
        'Silhouette': sil,
        'Davies-Bouldin': dbi
    }

# Evaluar cada algoritmo
results = {
    'K-Means': evaluate_clustering(X_scaled, labels_kmeans, 'K-Means'),
    'DBSCAN': evaluate_clustering(X_scaled, labels_dbscan, 'DBSCAN'),
    'Hierarchical': evaluate_clustering(X_scaled, labels_hierarchical, 'Hierarchical')
}

# Mostrar resultados
print("\n📊 Comparación de Métricas:")
print("-" * 70)
print(f"{'Método':<15} {'Clusters':<10} {'Ruido':<8} {'Silhouette':<12} {'Davies-Bouldin':<15}")
print("-" * 70)
for name, metrics in results.items():
    sil = f"{metrics['Silhouette']:.3f}" if not np.isnan(metrics['Silhouette']) else "N/A"
    dbi = f"{metrics['Davies-Bouldin']:.3f}" if not np.isnan(metrics['Davies-Bouldin']) else "N/A"
    print(f"{metrics['Method']:<15} {metrics['Clusters']:<10} {metrics['Noise']:<8} {sil:<12} {dbi:<15}")


# --------------------------------------------
# 3.2 Comparación Visual
# --------------------------------------------
print("\n--- 3.2 Comparación Visual ---")

def compare_algorithms(X: np.ndarray, labels_dict: dict) -> None:
    """
    Compare clustering results visually using PCA.
    """
    # Reducir a 2D con PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"Varianza explicada por PCA: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    
    # Crear visualización
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (name, labels) in zip(axes, labels_dict.items()):
        # Colorear puntos
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=labels, cmap='viridis', alpha=0.6, s=50)
        
        # Marcar ruido si existe
        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1],
                      c='red', marker='x', s=50, label='Ruido')
            ax.legend()
        
        # Métricas
        metrics = results[name]
        sil = f"{metrics['Silhouette']:.3f}" if not np.isnan(metrics['Silhouette']) else "N/A"
        
        ax.set_title(f'{name}\nClusters: {metrics["Clusters"]}, Silhouette: {sil}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.colorbar(scatter, ax=ax, label='Cluster')
    
    plt.suptitle('Comparación de Algoritmos de Clustering (PCA 2D)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('clusters_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Gráfico guardado: clusters_comparison.png")

labels_dict = {
    'K-Means': labels_kmeans,
    'DBSCAN': labels_dbscan,
    'Hierarchical': labels_hierarchical
}
compare_algorithms(X_scaled, labels_dict)


# ============================================
# FASE 4: INTERPRETACIÓN Y RECOMENDACIONES
# ============================================
print("\n" + "=" * 70)
print("FASE 4: INTERPRETACIÓN Y RECOMENDACIONES")
print("=" * 70)

# Seleccionar el mejor modelo (K-Means en este caso)
best_labels = labels_kmeans
best_model_name = 'K-Means'
print(f"\n📌 Modelo seleccionado: {best_model_name}")

# --------------------------------------------
# 4.1 Análisis de Segmentos
# --------------------------------------------
print("\n--- 4.1 Análisis de Segmentos ---")

def analyze_segments(df: pd.DataFrame, labels: np.ndarray, 
                     feature_names: list) -> pd.DataFrame:
    """
    Analyze and characterize each customer segment.
    """
    df_analysis = df.copy()
    df_analysis['Cluster'] = labels
    
    # Calcular estadísticas por cluster
    segment_stats = df_analysis.groupby('Cluster').agg({
        col: ['mean', 'median', 'std', 'count'] for col in feature_names
    }).round(2)
    
    # Simplificar para mostrar
    segment_profiles = df_analysis.groupby('Cluster')[feature_names].mean().round(2)
    segment_sizes = df_analysis.groupby('Cluster').size()
    
    print("\n📊 Perfil de Cada Segmento (Medias):")
    print(segment_profiles)
    
    print("\n📊 Tamaño de Cada Segmento:")
    for cluster, size in segment_sizes.items():
        pct = size / len(df_analysis) * 100
        print(f"  Cluster {cluster}: {size} clientes ({pct:.1f}%)")
    
    return segment_profiles


def name_segments(segment_profiles: pd.DataFrame) -> dict:
    """
    Assign meaningful names to segments based on characteristics.
    """
    segment_names = {}
    
    for cluster in segment_profiles.index:
        profile = segment_profiles.loc[cluster]
        
        # Lógica de nombrado basada en características
        recency = profile['recency']
        frequency = profile['frequency']
        monetary = profile['monetary']
        tenure = profile['tenure']
        
        # Clasificar basado en RFM y tenure
        if frequency > 35 and monetary > 3000 and recency < 30:
            name = "🏆 Champions"
            description = "Compradores frecuentes, alto valor, muy recientes"
        elif frequency > 20 and monetary > 1500 and recency < 50:
            name = "💎 Loyal Customers"
            description = "Clientes fieles con buen valor"
        elif recency > 100 and frequency < 15:
            name = "⚠️ At Risk"
            description = "Compradores pasados que pueden perderse"
        elif tenure < 150 and frequency < 10:
            name = "🌱 New Customers"
            description = "Clientes recientes con potencial"
        elif recency > 150:
            name = "😴 Lost/Dormant"
            description = "Clientes inactivos por mucho tiempo"
        else:
            name = f"📊 Segment {cluster}"
            description = "Segmento mixto"
        
        segment_names[cluster] = {
            'name': name,
            'description': description
        }
    
    print("\n🏷️ Nombres de Segmentos:")
    for cluster, info in segment_names.items():
        print(f"\n  Cluster {cluster}: {info['name']}")
        print(f"    → {info['description']}")
    
    return segment_names

segment_profiles = analyze_segments(df_clean, best_labels, feature_names)
segment_names = name_segments(segment_profiles)


# --------------------------------------------
# 4.2 Visualización de Perfiles
# --------------------------------------------
print("\n--- 4.2 Visualización de Perfiles ---")

def visualize_segment_profiles(df: pd.DataFrame, labels: np.ndarray,
                               segment_names: dict, feature_names: list) -> None:
    """
    Create visualizations of segment profiles.
    """
    df_vis = df.copy()
    df_vis['Cluster'] = labels
    df_vis['Segment'] = df_vis['Cluster'].map(
        lambda x: segment_names[x]['name'] if x in segment_names else f'Cluster {x}'
    )
    
    # 1. Box plots por feature
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(feature_names):
        df_vis.boxplot(column=col, by='Segment', ax=axes[idx])
        axes[idx].set_title(col)
        axes[idx].set_xlabel('')
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Distribución de Variables por Segmento', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('segment_boxplots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Gráfico guardado: segment_boxplots.png")
    
    # 2. Radar chart de perfiles
    # Normalizar perfiles para radar
    profiles_norm = segment_profiles.copy()
    for col in profiles_norm.columns:
        min_val = profiles_norm[col].min()
        max_val = profiles_norm[col].max()
        if max_val > min_val:
            profiles_norm[col] = (profiles_norm[col] - min_val) / (max_val - min_val)
    
    # Crear radar chart
    categories = list(profiles_norm.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(profiles_norm)))
    
    for idx, (cluster, row) in enumerate(profiles_norm.iterrows()):
        values = row.values.tolist()
        values += values[:1]
        
        name = segment_names[cluster]['name'] if cluster in segment_names else f'Cluster {cluster}'
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title('Perfiles de Segmentos (Normalizado)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('segment_profiles.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Gráfico guardado: segment_profiles.png")
    
    # 3. Distribución de tamaños
    segment_sizes = df_vis['Segment'].value_counts()
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(segment_sizes)))
    bars = plt.bar(range(len(segment_sizes)), segment_sizes.values, color=colors)
    plt.xticks(range(len(segment_sizes)), segment_sizes.index, rotation=45, ha='right')
    plt.xlabel('Segmento', fontsize=12)
    plt.ylabel('Número de Clientes', fontsize=12)
    plt.title('Distribución de Clientes por Segmento', fontsize=14, fontweight='bold')
    
    # Añadir etiquetas de porcentaje
    for bar, size in zip(bars, segment_sizes.values):
        pct = size / len(df_vis) * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{pct:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('segment_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Gráfico guardado: segment_distribution.png")

visualize_segment_profiles(df_clean, best_labels, segment_names, feature_names)


# --------------------------------------------
# 4.3 Recomendaciones de Marketing
# --------------------------------------------
print("\n--- 4.3 Recomendaciones de Marketing ---")

def generate_recommendations(segment_profiles: pd.DataFrame,
                             segment_names: dict) -> dict:
    """
    Generate marketing recommendations for each segment.
    """
    recommendations = {}
    
    for cluster in segment_profiles.index:
        profile = segment_profiles.loc[cluster]
        info = segment_names.get(cluster, {'name': f'Cluster {cluster}', 'description': ''})
        
        # Generar recomendaciones basadas en el perfil
        if 'Champions' in info['name']:
            recs = {
                'strategy': 'Retención y Advocacy',
                'actions': [
                    '🎁 Programa VIP exclusivo con beneficios premium',
                    '🗣️ Programa de referidos con recompensas',
                    '📧 Acceso anticipado a nuevos productos',
                    '💌 Comunicación personalizada y agradecimiento',
                    '🎯 Cross-selling de productos complementarios'
                ],
                'priority': 'ALTA - Proteger y aprovechar'
            }
        elif 'Loyal' in info['name']:
            recs = {
                'strategy': 'Upselling y Engagement',
                'actions': [
                    '⬆️ Ofertas de upgrade a productos premium',
                    '🎯 Recomendaciones personalizadas',
                    '💳 Programa de lealtad con beneficios',
                    '📊 Encuestas de satisfacción',
                    '🎁 Descuentos exclusivos por antigüedad'
                ],
                'priority': 'ALTA - Incrementar valor'
            }
        elif 'At Risk' in info['name']:
            recs = {
                'strategy': 'Reactivación Urgente',
                'actions': [
                    '🚨 Campañas de win-back con ofertas especiales',
                    '📞 Contacto personal para entender necesidades',
                    '💰 Descuentos significativos en próxima compra',
                    '❓ Encuesta de satisfacción para identificar problemas',
                    '⏰ Crear urgencia con ofertas limitadas'
                ],
                'priority': 'CRÍTICA - Actuar inmediatamente'
            }
        elif 'New' in info['name']:
            recs = {
                'strategy': 'Onboarding y Conversión',
                'actions': [
                    '👋 Secuencia de bienvenida personalizada',
                    '📚 Contenido educativo sobre productos',
                    '💰 Descuento en segunda compra',
                    '🎯 Recomendaciones basadas en primera compra',
                    '📱 Incentivar descarga de app/registro'
                ],
                'priority': 'MEDIA - Construir relación'
            }
        elif 'Lost' in info['name'] or 'Dormant' in info['name']:
            recs = {
                'strategy': 'Reactivación o Depuración',
                'actions': [
                    '📧 Campaña de "te extrañamos" con oferta fuerte',
                    '❓ Encuesta para entender abandono',
                    '🔄 Mostrar novedades desde última visita',
                    '⚠️ Evaluar costo de reactivación vs adquisición',
                    '🗑️ Considerar limpieza de base de datos'
                ],
                'priority': 'BAJA - Evaluar ROI'
            }
        else:
            recs = {
                'strategy': 'Análisis Adicional',
                'actions': [
                    '🔍 Profundizar análisis de este segmento',
                    '📊 Identificar patrones específicos',
                    '🎯 Personalizar ofertas según características'
                ],
                'priority': 'MEDIA'
            }
        
        recommendations[cluster] = recs
    
    # Imprimir recomendaciones
    print("\n" + "=" * 70)
    print("📋 RECOMENDACIONES DE MARKETING POR SEGMENTO")
    print("=" * 70)
    
    for cluster, recs in recommendations.items():
        info = segment_names.get(cluster, {'name': f'Cluster {cluster}'})
        print(f"\n{'='*50}")
        print(f"🎯 {info['name']} (Cluster {cluster})")
        print(f"{'='*50}")
        print(f"📌 Estrategia: {recs['strategy']}")
        print(f"⚡ Prioridad: {recs['priority']}")
        print(f"\n📝 Acciones recomendadas:")
        for action in recs['actions']:
            print(f"   {action}")
    
    return recommendations

recommendations = generate_recommendations(segment_profiles, segment_names)


# ============================================
# RESUMEN FINAL
# ============================================
print("\n" + "=" * 70)
print("📊 RESUMEN EJECUTIVO")
print("=" * 70)

print(f"""
🔬 ANÁLISIS DE SEGMENTACIÓN DE CLIENTES

📌 Metodología:
   - Dataset: {len(df_clean)} clientes, {len(feature_names)} variables
   - Algoritmos probados: K-Means, DBSCAN, Jerárquico
   - Modelo final: {best_model_name} con K={best_k}
   - Silhouette Score: {results[best_model_name]['Silhouette']:.3f}

📊 Segmentos Identificados:
""")

for cluster in sorted(segment_names.keys()):
    info = segment_names[cluster]
    size = (best_labels == cluster).sum()
    pct = size / len(best_labels) * 100
    print(f"   {info['name']}: {size} clientes ({pct:.1f}%)")

print(f"""
📁 Archivos Generados:
   - eda_distributions.png
   - eda_correlation.png
   - elbow_silhouette.png
   - k_distance_graph.png
   - dendrogram.png
   - clusters_comparison.png
   - segment_boxplots.png
   - segment_profiles.png
   - segment_distribution.png

✅ Proyecto completado exitosamente!
""")
