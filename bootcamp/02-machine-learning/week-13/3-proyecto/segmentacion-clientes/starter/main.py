# ============================================
# PROYECTO: Segmentación de Clientes
# ============================================
# Aplicar K-Means, DBSCAN y Jerárquico para
# segmentar clientes de un e-commerce
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats

print("=" * 70)
print("PROYECTO: Segmentación de Clientes para Marketing")
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
    # TODO: Implementar generación de datos
    # Hint: Usar np.random.seed(random_state)
    # Hint: Crear diferentes segmentos con características distintas
    pass


# --------------------------------------------
# 1.2 Exploración de Datos (EDA)
# --------------------------------------------
print("\n--- 1.2 Exploración de Datos ---")

def exploratory_analysis(df: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis on customer data.
    
    Includes:
    - Basic statistics
    - Distribution plots
    - Correlation matrix
    
    Args:
        df: Customer DataFrame
    """
    # TODO: Implementar EDA
    # Hint: Usar df.describe(), histogramas, heatmap de correlación
    pass


# --------------------------------------------
# 1.3 Preprocesamiento
# --------------------------------------------
print("\n--- 1.3 Preprocesamiento ---")

def preprocess_data(df: pd.DataFrame, remove_outliers: bool = True,
                    outlier_threshold: float = 3.0) -> tuple:
    """
    Preprocess customer data for clustering.
    
    Steps:
    1. Remove outliers using z-score
    2. Standardize features
    
    Args:
        df: Customer DataFrame
        remove_outliers: Whether to remove outliers
        outlier_threshold: Z-score threshold for outliers
    
    Returns:
        X_scaled: Scaled feature matrix
        scaler: Fitted StandardScaler
        df_clean: Cleaned DataFrame
    """
    # TODO: Implementar preprocesamiento
    # Hint: scipy.stats.zscore para outliers
    # Hint: StandardScaler para normalización
    pass


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
    
    Args:
        X: Scaled feature matrix
        k_range: Range of K values to try
    
    Returns:
        best_k: Optimal number of clusters
        inertias: List of inertias for elbow plot
        silhouettes: List of silhouette scores
    """
    # TODO: Implementar búsqueda de K óptimo
    # Hint: Calcular inertia (elbow) y silhouette para cada K
    pass


def apply_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> tuple:
    """
    Apply K-Means clustering.
    
    Args:
        X: Scaled feature matrix
        n_clusters: Number of clusters
        random_state: Random seed
    
    Returns:
        labels: Cluster labels
        centroids: Cluster centroids
        model: Fitted KMeans model
    """
    # TODO: Implementar K-Means
    pass


# --------------------------------------------
# 2.2 DBSCAN Clustering
# --------------------------------------------
print("\n--- 2.2 DBSCAN ---")

def find_optimal_epsilon(X: np.ndarray, min_samples: int = 5) -> float:
    """
    Find optimal epsilon using k-distance graph.
    
    Args:
        X: Scaled feature matrix
        min_samples: Minimum samples parameter
    
    Returns:
        epsilon: Suggested epsilon value
    """
    # TODO: Implementar k-distance graph
    # Hint: NearestNeighbors para calcular distancias
    pass


def apply_dbscan(X: np.ndarray, eps: float, min_samples: int = 5) -> tuple:
    """
    Apply DBSCAN clustering.
    
    Args:
        X: Scaled feature matrix
        eps: Epsilon parameter
        min_samples: Minimum samples parameter
    
    Returns:
        labels: Cluster labels (-1 for noise)
        n_clusters: Number of clusters found
        n_noise: Number of noise points
    """
    # TODO: Implementar DBSCAN
    pass


# --------------------------------------------
# 2.3 Clustering Jerárquico
# --------------------------------------------
print("\n--- 2.3 Clustering Jerárquico ---")

def create_dendrogram(X: np.ndarray, method: str = 'ward') -> np.ndarray:
    """
    Create dendrogram for hierarchical clustering.
    
    Args:
        X: Scaled feature matrix
        method: Linkage method
    
    Returns:
        Z: Linkage matrix
    """
    # TODO: Implementar dendrograma
    # Hint: scipy.cluster.hierarchy.linkage y dendrogram
    pass


def apply_hierarchical(X: np.ndarray, n_clusters: int, 
                       linkage: str = 'ward') -> np.ndarray:
    """
    Apply Agglomerative Clustering.
    
    Args:
        X: Scaled feature matrix
        n_clusters: Number of clusters
        linkage: Linkage method
    
    Returns:
        labels: Cluster labels
    """
    # TODO: Implementar clustering jerárquico
    pass


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
    
    Args:
        X: Scaled feature matrix
        labels: Cluster labels
        name: Name of the algorithm
    
    Returns:
        Dictionary with metrics
    """
    # TODO: Implementar evaluación
    # Métricas: silhouette, davies_bouldin, n_clusters
    pass


# --------------------------------------------
# 3.2 Comparación de Algoritmos
# --------------------------------------------
print("\n--- 3.2 Comparación de Algoritmos ---")

def compare_algorithms(X: np.ndarray, results: dict) -> None:
    """
    Compare clustering results from all algorithms.
    
    Creates visualization comparing:
    - Cluster assignments
    - Metrics comparison
    
    Args:
        X: Scaled feature matrix
        results: Dictionary with results from each algorithm
    """
    # TODO: Implementar comparación visual
    # Hint: PCA para reducir a 2D y visualizar
    pass


# ============================================
# FASE 4: INTERPRETACIÓN Y RECOMENDACIONES
# ============================================
print("\n" + "=" * 70)
print("FASE 4: INTERPRETACIÓN Y RECOMENDACIONES")
print("=" * 70)

# --------------------------------------------
# 4.1 Análisis de Segmentos
# --------------------------------------------
print("\n--- 4.1 Análisis de Segmentos ---")

def analyze_segments(df: pd.DataFrame, labels: np.ndarray, 
                     feature_names: list) -> pd.DataFrame:
    """
    Analyze and characterize each customer segment.
    
    Args:
        df: Original customer DataFrame
        labels: Cluster labels
        feature_names: List of feature names
    
    Returns:
        DataFrame with segment profiles
    """
    # TODO: Implementar análisis de segmentos
    # Calcular media, mediana, std por segmento
    pass


def name_segments(segment_profiles: pd.DataFrame) -> dict:
    """
    Assign meaningful names to segments based on characteristics.
    
    Args:
        segment_profiles: DataFrame with segment statistics
    
    Returns:
        Dictionary mapping cluster ID to segment name
    """
    # TODO: Implementar nombrado de segmentos
    # Ejemplos: "Champions", "At Risk", "New Customers", etc.
    pass


# --------------------------------------------
# 4.2 Visualización de Perfiles
# --------------------------------------------
print("\n--- 4.2 Visualización de Perfiles ---")

def visualize_segment_profiles(df: pd.DataFrame, labels: np.ndarray,
                               segment_names: dict) -> None:
    """
    Create visualizations of segment profiles.
    
    Includes:
    - Radar charts
    - Box plots per feature
    - Segment size distribution
    
    Args:
        df: Customer DataFrame
        labels: Cluster labels
        segment_names: Dictionary of segment names
    """
    # TODO: Implementar visualizaciones de perfiles
    pass


# --------------------------------------------
# 4.3 Recomendaciones de Marketing
# --------------------------------------------
print("\n--- 4.3 Recomendaciones de Marketing ---")

def generate_recommendations(segment_profiles: pd.DataFrame,
                             segment_names: dict) -> dict:
    """
    Generate marketing recommendations for each segment.
    
    Args:
        segment_profiles: DataFrame with segment statistics
        segment_names: Dictionary of segment names
    
    Returns:
        Dictionary with recommendations per segment
    """
    # TODO: Implementar recomendaciones
    # Considerar: retención, upselling, reactivación, etc.
    pass


# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================
print("\n" + "=" * 70)
print("EJECUCIÓN DEL PROYECTO")
print("=" * 70)

def main():
    """
    Main function to run the complete customer segmentation pipeline.
    """
    print("\n🚀 Iniciando pipeline de segmentación de clientes...\n")
    
    # -----------------------------------------
    # Fase 1: Preparación
    # -----------------------------------------
    print("📊 Fase 1: Preparación de datos")
    
    # TODO: Llamar a generate_customer_data
    # TODO: Llamar a exploratory_analysis
    # TODO: Llamar a preprocess_data
    
    # -----------------------------------------
    # Fase 2: Clustering
    # -----------------------------------------
    print("\n🔬 Fase 2: Aplicación de algoritmos")
    
    # TODO: Aplicar K-Means
    # TODO: Aplicar DBSCAN
    # TODO: Aplicar Jerárquico
    
    # -----------------------------------------
    # Fase 3: Evaluación
    # -----------------------------------------
    print("\n📈 Fase 3: Evaluación y comparación")
    
    # TODO: Evaluar cada algoritmo
    # TODO: Comparar resultados
    
    # -----------------------------------------
    # Fase 4: Interpretación
    # -----------------------------------------
    print("\n💡 Fase 4: Interpretación y recomendaciones")
    
    # TODO: Analizar segmentos del mejor modelo
    # TODO: Nombrar segmentos
    # TODO: Visualizar perfiles
    # TODO: Generar recomendaciones
    
    print("\n✅ Pipeline completado!")

# Ejecutar
if __name__ == "__main__":
    main()

# ============================================
# RESUMEN DEL PROYECTO
# ============================================
print("\n" + "=" * 70)
print("INSTRUCCIONES PARA COMPLETAR EL PROYECTO")
print("=" * 70)
print("""
Este proyecto requiere que implementes las siguientes funciones:

FASE 1 - Preparación:
  □ generate_customer_data() - Generar datos sintéticos RFM
  □ exploratory_analysis() - EDA con visualizaciones
  □ preprocess_data() - Limpieza y normalización

FASE 2 - Clustering:
  □ find_optimal_k() - Método del codo y silhouette
  □ apply_kmeans() - Aplicar K-Means
  □ find_optimal_epsilon() - K-distance graph
  □ apply_dbscan() - Aplicar DBSCAN
  □ create_dendrogram() - Dendrograma
  □ apply_hierarchical() - Clustering jerárquico

FASE 3 - Evaluación:
  □ evaluate_clustering() - Métricas de evaluación
  □ compare_algorithms() - Comparación visual

FASE 4 - Interpretación:
  □ analyze_segments() - Perfiles de segmentos
  □ name_segments() - Nombrar segmentos
  □ visualize_segment_profiles() - Visualizaciones
  □ generate_recommendations() - Recomendaciones de marketing

Tiempo estimado: 2 horas

Entregables:
  - Código completo y funcional
  - Visualizaciones generadas
  - Informe de segmentos con recomendaciones
""")
