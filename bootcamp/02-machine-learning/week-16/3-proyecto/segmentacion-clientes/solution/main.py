"""
Proyecto: Segmentación de Clientes - SOLUCIÓN
=============================================
Implementación completa de segmentación de clientes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')


# ============================================
# 1. CARGAR Y EXPLORAR DATOS
# ============================================

def create_sample_data(n_samples=200):
    """Crea datos simulados de clientes."""
    np.random.seed(42)
    
    # Simular diferentes segmentos
    # Segmento 1: Jóvenes bajo ingreso alto gasto
    n1 = n_samples // 4
    age1 = np.random.normal(25, 5, n1).clip(18, 35)
    income1 = np.random.normal(30, 10, n1).clip(15, 50)
    spending1 = np.random.normal(75, 10, n1).clip(50, 100)
    
    # Segmento 2: Adultos alto ingreso alto gasto
    n2 = n_samples // 4
    age2 = np.random.normal(40, 8, n2).clip(30, 55)
    income2 = np.random.normal(100, 20, n2).clip(70, 150)
    spending2 = np.random.normal(80, 10, n2).clip(60, 100)
    
    # Segmento 3: Adultos medio ingreso bajo gasto
    n3 = n_samples // 4
    age3 = np.random.normal(45, 10, n3).clip(30, 60)
    income3 = np.random.normal(60, 15, n3).clip(40, 90)
    spending3 = np.random.normal(30, 10, n3).clip(10, 50)
    
    # Segmento 4: Seniors moderado
    n4 = n_samples - n1 - n2 - n3
    age4 = np.random.normal(55, 8, n4).clip(45, 70)
    income4 = np.random.normal(50, 15, n4).clip(30, 80)
    spending4 = np.random.normal(50, 15, n4).clip(30, 70)
    
    df = pd.DataFrame({
        'CustomerID': range(1, n_samples + 1),
        'Age': np.concatenate([age1, age2, age3, age4]).astype(int),
        'Annual_Income': np.concatenate([income1, income2, income3, income4]).round(1),
        'Spending_Score': np.concatenate([spending1, spending2, spending3, spending4]).astype(int),
        'Years_Customer': np.random.exponential(3, n_samples).clip(0, 20).round(1)
    })
    
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def explore_data(df):
    """Análisis exploratorio de datos."""
    print("\n=== ANÁLISIS EXPLORATORIO ===")
    print(f"\nShape: {df.shape}")
    print("\nPrimeras filas:")
    print(df.head())
    
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    # Visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Distribuciones
    features = ['Age', 'Annual_Income', 'Spending_Score', 'Years_Customer']
    for i, feat in enumerate(features):
        ax = axes[i//2, i%2]
        ax.hist(df[feat], bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel(feat)
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'Distribución de {feat}')
    
    plt.tight_layout()
    plt.savefig('distribucion_features.png', dpi=150)
    plt.close()
    
    # Scatter plot principal
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['Annual_Income'], df['Spending_Score'], 
                          c=df['Age'], cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Age')
    plt.xlabel('Annual Income (K)')
    plt.ylabel('Spending Score')
    plt.title('Clientes: Ingreso vs Gasto (color = edad)')
    plt.savefig('scatter_income_spending.png', dpi=150)
    plt.close()
    
    print("\n✓ Visualizaciones guardadas")


# ============================================
# 2. PREPROCESAMIENTO
# ============================================

def preprocess_data(df, features):
    """Preprocesa los datos para clustering."""
    X = df[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nFeatures seleccionadas: {features}")
    print(f"Shape de datos escalados: {X_scaled.shape}")
    
    return X_scaled, scaler


# ============================================
# 3. DETERMINAR K ÓPTIMO
# ============================================

def find_optimal_k(X, max_k=10):
    """Encuentra el número óptimo de clusters."""
    results = {'K': [], 'Inertia': [], 'Silhouette': [], 'Davies_Bouldin': []}
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        results['K'].append(k)
        results['Inertia'].append(kmeans.inertia_)
        results['Silhouette'].append(silhouette_score(X, labels))
        results['Davies_Bouldin'].append(davies_bouldin_score(X, labels))
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Elbow
    axes[0].plot(results['K'], results['Inertia'], 'bo-', linewidth=2)
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Inercia')
    axes[0].set_title('Método del Codo')
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette
    axes[1].plot(results['K'], results['Silhouette'], 'go-', linewidth=2)
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette (mayor es mejor)')
    axes[1].grid(True, alpha=0.3)
    best_k_sil = results['K'][np.argmax(results['Silhouette'])]
    axes[1].axvline(x=best_k_sil, color='r', linestyle='--', label=f'Mejor: K={best_k_sil}')
    axes[1].legend()
    
    # Davies-Bouldin
    axes[2].plot(results['K'], results['Davies_Bouldin'], 'ro-', linewidth=2)
    axes[2].set_xlabel('K')
    axes[2].set_ylabel('Davies-Bouldin Index')
    axes[2].set_title('Davies-Bouldin (menor es mejor)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('seleccion_k.png', dpi=150)
    plt.close()
    
    df_results = pd.DataFrame(results)
    print("\n=== MÉTRICAS POR K ===")
    print(df_results.to_string(index=False))
    
    return results


# ============================================
# 4. APLICAR ALGORITMOS DE CLUSTERING
# ============================================

def apply_kmeans(X, n_clusters):
    """Aplica K-Means clustering."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return model, labels


def apply_dbscan(X, eps, min_samples):
    """Aplica DBSCAN clustering."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return model, labels


def apply_hierarchical(X, n_clusters, linkage_method='ward'):
    """Aplica clustering jerárquico."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X)
    return model, labels


# ============================================
# 5. EVALUAR CLUSTERING
# ============================================

def evaluate_clustering(X, labels, algorithm_name):
    """Evalúa la calidad del clustering."""
    mask = labels >= 0
    n_clusters = len(set(labels[mask]))
    n_noise = (labels == -1).sum()
    
    results = {
        'Algorithm': algorithm_name,
        'N_Clusters': n_clusters,
        'N_Noise': n_noise
    }
    
    if n_clusters > 1 and mask.sum() > n_clusters:
        results['Silhouette'] = silhouette_score(X[mask], labels[mask])
        results['Davies_Bouldin'] = davies_bouldin_score(X[mask], labels[mask])
    else:
        results['Silhouette'] = None
        results['Davies_Bouldin'] = None
    
    print(f"\n{algorithm_name}:")
    print(f"  Clusters: {n_clusters}, Noise: {n_noise}")
    if results['Silhouette']:
        print(f"  Silhouette: {results['Silhouette']:.4f}")
        print(f"  Davies-Bouldin: {results['Davies_Bouldin']:.4f}")
    
    return results


# ============================================
# 6. CARACTERIZAR SEGMENTOS
# ============================================

def characterize_segments(df, labels, features):
    """Genera perfiles de cada segmento."""
    df_with_labels = df.copy()
    df_with_labels['Segment'] = labels
    
    # Excluir noise
    df_clusters = df_with_labels[df_with_labels['Segment'] >= 0]
    
    # Estadísticas por segmento
    profiles = df_clusters.groupby('Segment')[features].agg(['mean', 'std', 'count'])
    
    # Simplificar
    summary = df_clusters.groupby('Segment').agg({
        'Age': 'mean',
        'Annual_Income': 'mean',
        'Spending_Score': 'mean',
        'Years_Customer': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Count'})
    
    # Nombrar segmentos
    segment_names = []
    for idx, row in summary.iterrows():
        if row['Age'] < 35 and row['Spending_Score'] > 60:
            name = "Jóvenes Alto Gasto"
        elif row['Annual_Income'] > 80 and row['Spending_Score'] > 60:
            name = "Premium Alto Valor"
        elif row['Spending_Score'] < 40:
            name = "Conservadores"
        elif row['Age'] > 50:
            name = "Seniors Moderados"
        else:
            name = f"Segmento {idx}"
        segment_names.append(name)
    
    summary['Segment_Name'] = segment_names
    
    print("\n=== PERFILES DE SEGMENTOS ===")
    print(summary.round(1).to_string())
    
    return summary


def visualize_segments(X, labels, df, features, centers=None, title="Segmentos de Clientes"):
    """Visualiza los segmentos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 2D: Income vs Spending
    mask = labels >= 0
    scatter = axes[0].scatter(df.loc[mask, 'Annual_Income'], df.loc[mask, 'Spending_Score'],
                               c=labels[mask], cmap='viridis', alpha=0.6, s=50)
    if centers is not None:
        # Denormalizar centroides si es necesario
        axes[0].scatter(centers[:, 1] * df['Annual_Income'].std() + df['Annual_Income'].mean(),
                        centers[:, 2] * df['Spending_Score'].std() + df['Spending_Score'].mean(),
                        c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    axes[0].set_xlabel('Annual Income (K)')
    axes[0].set_ylabel('Spending Score')
    axes[0].set_title('Ingreso vs Gasto')
    plt.colorbar(scatter, ax=axes[0], label='Segmento')
    
    # 2D escalado
    axes[1].scatter(X[mask, 0], X[mask, 1], c=labels[mask], cmap='viridis', alpha=0.6, s=50)
    if centers is not None:
        axes[1].scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200,
                        edgecolors='black', linewidths=2)
    axes[1].set_xlabel('Feature 1 (scaled)')
    axes[1].set_ylabel('Feature 2 (scaled)')
    axes[1].set_title('Vista Escalada')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('segmentos_visualizacion.png', dpi=150)
    plt.close()
    
    print("\n✓ Visualización guardada: segmentos_visualizacion.png")


# ============================================
# 7. GENERAR RECOMENDACIONES
# ============================================

def generate_recommendations(segment_profiles):
    """Genera recomendaciones de marketing por segmento."""
    print("\n" + "="*60)
    print("RECOMENDACIONES DE MARKETING")
    print("="*60)
    
    for idx, row in segment_profiles.iterrows():
        name = row['Segment_Name']
        print(f"\n📊 {name} (n={int(row['Count'])})")
        print("-" * 40)
        
        if "Jóvenes Alto Gasto" in name:
            print("• Productos: Tech, moda, experiencias")
            print("• Canales: Redes sociales, apps móviles")
            print("• Promociones: Descuentos flash, exclusivas")
            print("• Estrategia: Gamificación, programas de referidos")
            
        elif "Premium" in name:
            print("• Productos: Líneas premium, servicios exclusivos")
            print("• Canales: Email personalizado, atención VIP")
            print("• Promociones: Acceso anticipado, eventos privados")
            print("• Estrategia: Programa de lealtad premium")
            
        elif "Conservadores" in name:
            print("• Productos: Básicos, ofertas de valor")
            print("• Canales: Email, tienda física")
            print("• Promociones: Cupones, descuentos por volumen")
            print("• Estrategia: Educación sobre beneficios")
            
        elif "Seniors" in name:
            print("• Productos: Calidad, durabilidad")
            print("• Canales: Email, teléfono, tienda física")
            print("• Promociones: Descuentos senior, bundles")
            print("• Estrategia: Servicio al cliente personalizado")
        
        else:
            print("• Analizar características específicas")
            print("• Desarrollar estrategia personalizada")


# ============================================
# MAIN
# ============================================

def main():
    """Pipeline principal del proyecto."""
    print("="*60)
    print("PROYECTO: SEGMENTACIÓN DE CLIENTES")
    print("="*60)
    
    # 1. Crear datos
    print("\n1. Cargando datos...")
    df = create_sample_data(200)
    explore_data(df)
    
    # 2. Preprocesamiento
    print("\n2. Preprocesando datos...")
    features = ['Age', 'Annual_Income', 'Spending_Score', 'Years_Customer']
    X_scaled, scaler = preprocess_data(df, features)
    
    # 3. Encontrar K óptimo
    print("\n3. Buscando K óptimo...")
    results = find_optimal_k(X_scaled)
    optimal_k = 4  # Basado en análisis
    print(f"\n→ K óptimo seleccionado: {optimal_k}")
    
    # 4. Aplicar algoritmos
    print("\n4. Aplicando algoritmos de clustering...")
    kmeans_model, labels_km = apply_kmeans(X_scaled, n_clusters=optimal_k)
    dbscan_model, labels_db = apply_dbscan(X_scaled, eps=0.8, min_samples=5)
    hier_model, labels_hier = apply_hierarchical(X_scaled, n_clusters=optimal_k)
    
    # 5. Evaluar
    print("\n5. Evaluando resultados...")
    evaluate_clustering(X_scaled, labels_km, 'K-Means')
    evaluate_clustering(X_scaled, labels_db, 'DBSCAN')
    evaluate_clustering(X_scaled, labels_hier, 'Jerárquico')
    
    # Seleccionar mejor (K-Means en este caso)
    best_labels = labels_km
    best_model = kmeans_model
    
    # 6. Caracterizar segmentos
    print("\n6. Caracterizando segmentos...")
    profiles = characterize_segments(df, best_labels, features)
    visualize_segments(X_scaled, best_labels, df, features, 
                       best_model.cluster_centers_, "Segmentación Final (K-Means)")
    
    # 7. Recomendaciones
    generate_recommendations(profiles)
    
    print("\n" + "="*60)
    print("ARCHIVOS GENERADOS:")
    print("  • distribucion_features.png")
    print("  • scatter_income_spending.png")
    print("  • seleccion_k.png")
    print("  • segmentos_visualizacion.png")
    print("="*60)
    print("PROYECTO COMPLETADO EXITOSAMENTE")
    print("="*60)


if __name__ == "__main__":
    main()
