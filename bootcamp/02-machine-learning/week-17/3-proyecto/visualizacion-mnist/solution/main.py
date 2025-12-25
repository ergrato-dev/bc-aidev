"""
Proyecto: Visualización y Clasificación de MNIST
=================================================
SOLUCIÓN COMPLETA
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print('UMAP no disponible. Instalar con: pip install umap-learn')


# ============================================
# CARGA DE DATOS
# ============================================

def load_data():
    """Carga y prepara el dataset de dígitos."""
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f'Dataset: {X.shape[0]} muestras, {X.shape[1]} features')
    print(f'Train: {X_train.shape[0]}, Test: {X_test.shape[0]}')
    
    return X_scaled, y, X_train, X_test, y_train, y_test


# ============================================
# FUNCIÓN 1: Visualización con PCA
# ============================================

def visualize_pca(X, y, n_components=2):
    """Aplica PCA y visualiza los datos."""
    pca = PCA(n_components=n_components)
    
    start = time.time()
    X_pca = pca.fit_transform(X)
    elapsed = time.time() - start
    
    var_explained = pca.explained_variance_ratio_.sum() * 100
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', 
                          alpha=0.7, s=30)
    plt.colorbar(scatter, label='Dígito')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title(f'PCA ({var_explained:.1f}% varianza, {elapsed:.3f}s)')
    plt.tight_layout()
    
    return X_pca, pca, elapsed


# ============================================
# FUNCIÓN 2: Visualización con t-SNE
# ============================================

def visualize_tsne(X, y, perplexity=30):
    """Aplica t-SNE y visualiza los datos."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    
    start = time.time()
    X_tsne = tsne.fit_transform(X)
    elapsed = time.time() - start
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', 
                          alpha=0.7, s=30)
    plt.colorbar(scatter, label='Dígito')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f't-SNE (perplexity={perplexity}, KL={tsne.kl_divergence_:.3f}, {elapsed:.2f}s)')
    plt.tight_layout()
    
    return X_tsne, elapsed


# ============================================
# FUNCIÓN 3: Visualización con UMAP
# ============================================

def visualize_umap(X, y, n_neighbors=15, min_dist=0.1):
    """Aplica UMAP y visualiza los datos."""
    if not UMAP_AVAILABLE:
        print('UMAP no disponible')
        return None, None, 0
    
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, 
                        min_dist=min_dist, random_state=42)
    
    start = time.time()
    X_umap = reducer.fit_transform(X)
    elapsed = time.time() - start
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', 
                          alpha=0.7, s=30)
    plt.colorbar(scatter, label='Dígito')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(f'UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, {elapsed:.2f}s)')
    plt.tight_layout()
    
    return X_umap, reducer, elapsed


# ============================================
# FUNCIÓN 4: Comparación de Técnicas
# ============================================

def compare_techniques(X, y):
    """Compara PCA, t-SNE y UMAP lado a lado."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    results = {}
    
    # PCA
    pca = PCA(n_components=2)
    start = time.time()
    X_pca = pca.fit_transform(X)
    time_pca = time.time() - start
    trust_pca = trustworthiness(X, X_pca, n_neighbors=15)
    
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6, s=15)
    axes[0].set_title(f'PCA\nTime: {time_pca:.2f}s, Trust: {trust_pca:.3f}')
    results['PCA'] = {'time': time_pca, 'trust': trust_pca}
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    start = time.time()
    X_tsne = tsne.fit_transform(X)
    time_tsne = time.time() - start
    trust_tsne = trustworthiness(X, X_tsne, n_neighbors=15)
    
    axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6, s=15)
    axes[1].set_title(f't-SNE\nTime: {time_tsne:.2f}s, Trust: {trust_tsne:.3f}')
    results['t-SNE'] = {'time': time_tsne, 'trust': trust_tsne}
    
    # UMAP
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42)
        start = time.time()
        X_umap = reducer.fit_transform(X)
        time_umap = time.time() - start
        trust_umap = trustworthiness(X, X_umap, n_neighbors=15)
        
        axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.6, s=15)
        axes[2].set_title(f'UMAP\nTime: {time_umap:.2f}s, Trust: {trust_umap:.3f}')
        results['UMAP'] = {'time': time_umap, 'trust': trust_umap}
    else:
        axes[2].text(0.5, 0.5, 'UMAP not available', ha='center', va='center')
        axes[2].set_title('UMAP\nNo disponible')
    
    plt.suptitle('Comparación: PCA vs t-SNE vs UMAP', fontsize=14)
    plt.tight_layout()
    
    return results


# ============================================
# FUNCIÓN 5: Análisis de Hiperparámetros t-SNE
# ============================================

def analyze_tsne_perplexity(X, y):
    """Analiza el efecto del parámetro perplexity en t-SNE."""
    perplexities = [5, 15, 30, 50]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for ax, perp in zip(axes, perplexities):
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        X_embedded = tsne.fit_transform(X)
        
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='tab10', alpha=0.6, s=15)
        ax.set_title(f'Perplexity = {perp}\nKL = {tsne.kl_divergence_:.3f}')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('t-SNE: Efecto de Perplexity', fontsize=14)
    plt.tight_layout()


# ============================================
# FUNCIÓN 6: Análisis de Hiperparámetros UMAP
# ============================================

def analyze_umap_params(X, y):
    """Analiza el efecto de n_neighbors y min_dist en UMAP."""
    if not UMAP_AVAILABLE:
        print('UMAP no disponible')
        return
    
    n_neighbors_list = [5, 15, 50]
    min_dist_list = [0.0, 0.25, 0.5]
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    for i, nn in enumerate(n_neighbors_list):
        for j, md in enumerate(min_dist_list):
            reducer = umap.UMAP(n_neighbors=nn, min_dist=md, random_state=42)
            X_embedded = reducer.fit_transform(X)
            
            axes[i, j].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, 
                               cmap='tab10', alpha=0.5, s=10)
            axes[i, j].set_title(f'nn={nn}, md={md}')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    
    plt.suptitle('UMAP: Efecto de n_neighbors y min_dist', fontsize=14)
    plt.tight_layout()


# ============================================
# FUNCIÓN 7: Pipeline de Clasificación
# ============================================

def classification_pipeline(X_train, X_test, y_train, y_test):
    """Compara clasificación con y sin reducción dimensional."""
    results = {}
    
    # Sin reducción
    pipe_no_red = Pipeline([('clf', SVC(kernel='rbf', random_state=42))])
    start = time.time()
    pipe_no_red.fit(X_train, y_train)
    acc_no_red = accuracy_score(y_test, pipe_no_red.predict(X_test))
    time_no_red = time.time() - start
    results['Sin reducción (64D)'] = {'accuracy': acc_no_red, 'time': time_no_red}
    
    # Con PCA diferentes componentes
    for n_comp in [10, 20, 30]:
        pipe = Pipeline([
            ('pca', PCA(n_components=n_comp)),
            ('clf', SVC(kernel='rbf', random_state=42))
        ])
        start = time.time()
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        elapsed = time.time() - start
        results[f'PCA ({n_comp}D)'] = {'accuracy': acc, 'time': elapsed}
    
    print('\n=== Resultados de Clasificación ===')
    for name, res in results.items():
        print(f'{name}: Accuracy={res["accuracy"]:.4f}, Time={res["time"]:.3f}s')
    
    return results


# ============================================
# FUNCIÓN 8: Encontrar Componentes Óptimos
# ============================================

def find_optimal_components(X_train, X_test, y_train, y_test):
    """Encuentra el número óptimo de componentes PCA para clasificación."""
    n_components_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    accuracies = []
    
    for n_comp in n_components_range:
        pipe = Pipeline([
            ('pca', PCA(n_components=n_comp)),
            ('clf', SVC(kernel='rbf', random_state=42))
        ])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        accuracies.append(acc)
    
    # Sin reducción
    pipe_full = Pipeline([('clf', SVC(kernel='rbf', random_state=42))])
    pipe_full.fit(X_train, y_train)
    acc_full = accuracy_score(y_test, pipe_full.predict(X_test))
    
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=acc_full, color='r', linestyle='--', label=f'Sin reducción ({acc_full:.3f})')
    plt.xlabel('Número de Componentes PCA')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Número de Componentes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    optimal_n = n_components_range[np.argmax(accuracies)]
    print(f'\nÓptimo: {optimal_n} componentes con accuracy {max(accuracies):.4f}')
    
    return optimal_n


# ============================================
# FUNCIÓN 9: Dashboard de Resultados
# ============================================

def create_dashboard(X, y, X_train, X_test, y_train, y_test):
    """Crea un dashboard visual con todos los resultados."""
    fig = plt.figure(figsize=(20, 15))
    
    # Row 1: Visualizaciones
    ax1 = fig.add_subplot(2, 3, 1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6, s=10)
    ax1.set_title('PCA')
    
    ax2 = fig.add_subplot(2, 3, 2)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)
    ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6, s=10)
    ax2.set_title('t-SNE')
    
    ax3 = fig.add_subplot(2, 3, 3)
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X)
        ax3.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.6, s=10)
    ax3.set_title('UMAP')
    
    # Row 2: Métricas y Scree Plot
    ax4 = fig.add_subplot(2, 3, 4)
    pca_full = PCA()
    pca_full.fit(X)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_) * 100
    ax4.plot(range(1, len(cumsum)+1), cumsum, 'g-', linewidth=2)
    ax4.axhline(y=95, color='r', linestyle='--')
    ax4.set_xlabel('Componentes')
    ax4.set_ylabel('Varianza Acumulada (%)')
    ax4.set_title('Scree Plot - PCA')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(2, 3, 5)
    n_comp_range = range(5, 55, 5)
    accs = []
    for n in n_comp_range:
        pipe = Pipeline([('pca', PCA(n_components=n)), ('clf', SVC(random_state=42))])
        pipe.fit(X_train, y_train)
        accs.append(accuracy_score(y_test, pipe.predict(X_test)))
    ax5.plot(list(n_comp_range), accs, 'bo-')
    ax5.set_xlabel('Componentes PCA')
    ax5.set_ylabel('Accuracy')
    ax5.set_title('Accuracy vs Componentes')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(2, 3, 6)
    methods = ['PCA', 't-SNE', 'UMAP']
    trusts = [
        trustworthiness(X, X_pca, n_neighbors=15),
        trustworthiness(X, X_tsne, n_neighbors=15),
        trustworthiness(X, X_umap, n_neighbors=15) if UMAP_AVAILABLE else 0
    ]
    ax6.bar(methods, trusts, color=['steelblue', 'orange', 'green'])
    ax6.set_ylabel('Trustworthiness')
    ax6.set_title('Métricas de Calidad')
    ax6.set_ylim(0.9, 1.0)
    
    plt.suptitle('Dashboard: Reducción Dimensional en MNIST', fontsize=16)
    plt.tight_layout()
    plt.savefig('dashboard_mnist.png', dpi=150, bbox_inches='tight')
    print('Dashboard guardado como dashboard_mnist.png')


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def main():
    """Función principal del proyecto."""
    print('=' * 60)
    print('PROYECTO: Visualización y Clasificación de MNIST')
    print('=' * 60)
    
    # Cargar datos
    X, y, X_train, X_test, y_train, y_test = load_data()
    
    # Parte 1: Visualizaciones individuales
    print('\n--- Parte 1: Visualizaciones Individuales ---')
    X_pca, pca, time_pca = visualize_pca(X, y)
    plt.show()
    
    X_tsne, time_tsne = visualize_tsne(X, y)
    plt.show()
    
    X_umap, reducer_umap, time_umap = visualize_umap(X, y)
    plt.show()
    
    # Parte 2: Comparación de técnicas
    print('\n--- Parte 2: Comparación de Técnicas ---')
    results = compare_techniques(X, y)
    plt.show()
    
    print('\n=== Métricas Comparativas ===')
    for name, res in results.items():
        print(f'{name}: Time={res["time"]:.2f}s, Trust={res["trust"]:.4f}')
    
    # Parte 3: Análisis de hiperparámetros
    print('\n--- Parte 3: Análisis de Hiperparámetros ---')
    analyze_tsne_perplexity(X, y)
    plt.show()
    
    analyze_umap_params(X, y)
    plt.show()
    
    # Parte 4: Pipeline de clasificación
    print('\n--- Parte 4: Pipeline de Clasificación ---')
    class_results = classification_pipeline(X_train, X_test, y_train, y_test)
    
    # Parte 5: Encontrar componentes óptimos
    print('\n--- Parte 5: Componentes Óptimos ---')
    optimal_n = find_optimal_components(X_train, X_test, y_train, y_test)
    plt.show()
    
    # Parte 6: Dashboard final
    print('\n--- Parte 6: Dashboard Final ---')
    create_dashboard(X, y, X_train, X_test, y_train, y_test)
    plt.show()
    
    # Conclusiones
    print('\n' + '=' * 60)
    print('CONCLUSIONES')
    print('=' * 60)
    print("""
    1. SEPARACIÓN DE CLASES:
       - t-SNE y UMAP producen mejor separación visual que PCA
       - UMAP es ligeramente mejor en trustworthiness
    
    2. VELOCIDAD:
       - PCA es el más rápido (milisegundos)
       - UMAP es más rápido que t-SNE
       - t-SNE es el más lento (segundos)
    
    3. CLASIFICACIÓN:
       - PCA con 20-30 componentes mantiene accuracy similar al original
       - Reducción de 64D a 30D con mínima pérdida de accuracy
       - Ventaja: menor tiempo de entrenamiento y predicción
    
    4. RECOMENDACIONES:
       - Exploración visual: t-SNE o UMAP
       - Pipeline de ML: PCA (puede transformar nuevos datos rápido)
       - Datos nuevos en producción: UMAP (tiene transform())
       - Interpretabilidad: PCA (componentes tienen significado)
    """)


if __name__ == '__main__':
    main()
