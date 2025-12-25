"""
Proyecto: Segmentación de Clientes
==================================
Aplica clustering para segmentar clientes de retail.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ============================================
# 1. CARGAR Y EXPLORAR DATOS
# ============================================

def create_sample_data(n_samples=200):
    """
    Crea datos simulados de clientes.
    
    Returns:
        DataFrame con datos de clientes
    """
    # TODO: Generar datos sintéticos de clientes con:
    # - CustomerID
    # - Age (18-70)
    # - Annual Income (15-150 K)
    # - Spending Score (1-100)
    # - Years as Customer (0-20)
    pass


def explore_data(df):
    """
    Análisis exploratorio de datos.
    
    Args:
        df: DataFrame con datos de clientes
    """
    # TODO: Mostrar estadísticas descriptivas
    # TODO: Visualizar distribuciones de cada variable
    # TODO: Matriz de correlación
    # TODO: Scatter plots de pares de variables
    pass


# ============================================
# 2. PREPROCESAMIENTO
# ============================================

def preprocess_data(df, features):
    """
    Preprocesa los datos para clustering.
    
    Args:
        df: DataFrame original
        features: Lista de features a usar
    
    Returns:
        X_scaled: Datos escalados
        scaler: Objeto StandardScaler
    """
    # TODO: Seleccionar features
    # TODO: Escalar datos con StandardScaler
    # TODO: Retornar datos escalados y scaler
    pass


# ============================================
# 3. DETERMINAR K ÓPTIMO
# ============================================

def find_optimal_k(X, max_k=10):
    """
    Encuentra el número óptimo de clusters.
    
    Args:
        X: Datos escalados
        max_k: Máximo K a probar
    
    Returns:
        dict: Resultados de métricas por K
    """
    # TODO: Calcular inercia, silhouette, davies-bouldin para cada K
    # TODO: Graficar método del codo
    # TODO: Graficar silhouette por K
    # TODO: Retornar diccionario con resultados
    pass


# ============================================
# 4. APLICAR ALGORITMOS DE CLUSTERING
# ============================================

def apply_kmeans(X, n_clusters):
    """
    Aplica K-Means clustering.
    """
    # TODO: Crear y ajustar KMeans
    # TODO: Retornar modelo y labels
    pass


def apply_dbscan(X, eps, min_samples):
    """
    Aplica DBSCAN clustering.
    """
    # TODO: Crear y ajustar DBSCAN
    # TODO: Retornar modelo y labels
    pass


def apply_hierarchical(X, n_clusters, linkage='ward'):
    """
    Aplica clustering jerárquico.
    """
    # TODO: Crear y ajustar AgglomerativeClustering
    # TODO: Retornar modelo y labels
    pass


# ============================================
# 5. EVALUAR CLUSTERING
# ============================================

def evaluate_clustering(X, labels, algorithm_name):
    """
    Evalúa la calidad del clustering.
    
    Args:
        X: Datos escalados
        labels: Etiquetas de cluster
        algorithm_name: Nombre del algoritmo
    
    Returns:
        dict: Métricas de evaluación
    """
    # TODO: Calcular silhouette score
    # TODO: Calcular davies-bouldin score
    # TODO: Contar clusters y noise (si aplica)
    # TODO: Retornar diccionario con métricas
    pass


# ============================================
# 6. CARACTERIZAR SEGMENTOS
# ============================================

def characterize_segments(df, labels, features):
    """
    Genera perfiles de cada segmento.
    
    Args:
        df: DataFrame original
        labels: Etiquetas de cluster
        features: Features usadas
    
    Returns:
        DataFrame: Estadísticas por segmento
    """
    # TODO: Añadir labels al DataFrame
    # TODO: Calcular media de cada feature por segmento
    # TODO: Calcular tamaño de cada segmento
    # TODO: Nombrar segmentos según características
    pass


def visualize_segments(X, labels, centers=None, title="Segmentos"):
    """
    Visualiza los segmentos en 2D.
    """
    # TODO: Scatter plot coloreado por segmento
    # TODO: Marcar centroides si están disponibles
    # TODO: Añadir leyenda y título
    pass


# ============================================
# 7. GENERAR RECOMENDACIONES
# ============================================

def generate_recommendations(segment_profiles):
    """
    Genera recomendaciones de marketing por segmento.
    
    Args:
        segment_profiles: DataFrame con perfiles de segmentos
    """
    # TODO: Para cada segmento, generar recomendaciones como:
    # - Tipo de productos a ofrecer
    # - Canales de comunicación
    # - Promociones específicas
    # - Estrategias de retención
    pass


# ============================================
# MAIN
# ============================================

def main():
    """
    Pipeline principal del proyecto.
    """
    print("="*60)
    print("PROYECTO: SEGMENTACIÓN DE CLIENTES")
    print("="*60)
    
    # 1. Crear/cargar datos
    print("\n1. Cargando datos...")
    # df = create_sample_data(200)
    # explore_data(df)
    
    # 2. Preprocesamiento
    print("\n2. Preprocesando datos...")
    # features = ['Age', 'Annual_Income', 'Spending_Score']
    # X_scaled, scaler = preprocess_data(df, features)
    
    # 3. Encontrar K óptimo
    print("\n3. Buscando K óptimo...")
    # results = find_optimal_k(X_scaled)
    
    # 4. Aplicar algoritmos
    print("\n4. Aplicando algoritmos...")
    # kmeans_model, labels_km = apply_kmeans(X_scaled, n_clusters=4)
    # dbscan_model, labels_db = apply_dbscan(X_scaled, eps=0.5, min_samples=5)
    # hier_model, labels_hier = apply_hierarchical(X_scaled, n_clusters=4)
    
    # 5. Evaluar
    print("\n5. Evaluando resultados...")
    # evaluate_clustering(X_scaled, labels_km, 'K-Means')
    # evaluate_clustering(X_scaled, labels_db, 'DBSCAN')
    # evaluate_clustering(X_scaled, labels_hier, 'Jerárquico')
    
    # 6. Caracterizar segmentos
    print("\n6. Caracterizando segmentos...")
    # profiles = characterize_segments(df, labels_km, features)
    # visualize_segments(X_scaled, labels_km, kmeans_model.cluster_centers_)
    
    # 7. Recomendaciones
    print("\n7. Generando recomendaciones...")
    # generate_recommendations(profiles)
    
    print("\n" + "="*60)
    print("PROYECTO COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    main()
