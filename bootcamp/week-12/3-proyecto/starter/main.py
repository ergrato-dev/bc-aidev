"""
Proyecto Semana 12: Clasificador de Spam
========================================
Compara KNN, SVM y Naive Bayes para clasificación de spam.

Objetivo: Accuracy >= 0.90
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import seaborn as sns

# ============================================
# CARGAR DATOS
# ============================================

def load_spam_data():
    """
    Carga el dataset de spam.
    Usa fetch_20newsgroups como alternativa si no tienes SMS Spam.
    """
    from sklearn.datasets import fetch_20newsgroups
    
    # Simulamos spam/ham con 2 categorías de newsgroups
    categories = ['rec.sport.hockey', 'talk.politics.misc']
    
    data = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    # 0 = ham (hockey), 1 = spam (politics)
    return data.data, data.target


# ============================================
# FUNCIÓN 1: Preprocesar Datos
# ============================================

def preprocess_data(texts, labels, test_size=0.2, random_state=42):
    """
    Divide los datos y aplica TF-IDF.
    
    Args:
        texts: Lista de textos
        labels: Lista de etiquetas (0=ham, 1=spam)
        test_size: Proporción de test
        random_state: Semilla
    
    Returns:
        X_train, X_test, y_train, y_test: Datos divididos (texto sin vectorizar)
        X_train_tfidf, X_test_tfidf: Datos vectorizados
        vectorizer: TfidfVectorizer ajustado
    """
    # TODO: Dividir datos en train/test con stratify
    X_train, X_test, y_train, y_test = None, None, None, None
    
    # TODO: Crear TfidfVectorizer con stop_words='english' y max_features=5000
    vectorizer = None
    
    # TODO: Ajustar vectorizer con X_train y transformar X_train y X_test
    X_train_tfidf, X_test_tfidf = None, None
    
    return X_train, X_test, y_train, y_test, X_train_tfidf, X_test_tfidf, vectorizer


# ============================================
# FUNCIÓN 2: Entrenar KNN
# ============================================

def train_knn(X_train, y_train, k_range=range(1, 21)):
    """
    Entrena KNN encontrando el k óptimo.
    
    Args:
        X_train: Features de entrenamiento (TF-IDF)
        y_train: Labels
        k_range: Rango de k a probar
    
    Returns:
        best_knn: Modelo KNN con mejor k
        best_k: Valor óptimo de k
        k_scores: Lista de scores por cada k
    """
    # TODO: Probar diferentes valores de k con cross_val_score
    k_scores = []
    
    # TODO: Encontrar el mejor k
    best_k = None
    
    # TODO: Crear y entrenar KNN con el mejor k
    best_knn = None
    
    return best_knn, best_k, k_scores


# ============================================
# FUNCIÓN 3: Entrenar SVM
# ============================================

def train_svm(X_train, y_train):
    """
    Entrena SVM con GridSearch para encontrar mejores parámetros.
    
    Args:
        X_train: Features de entrenamiento (TF-IDF)
        y_train: Labels
    
    Returns:
        best_svm: Mejor modelo SVM
        best_params: Mejores parámetros encontrados
    """
    # TODO: Definir param_grid con C=[0.1, 1, 10] y kernel=['linear', 'rbf']
    param_grid = {}
    
    # TODO: Crear GridSearchCV con SVC y cv=5
    grid_search = None
    
    # TODO: Ajustar grid_search
    
    # TODO: Obtener mejor modelo y parámetros
    best_svm = None
    best_params = None
    
    return best_svm, best_params


# ============================================
# FUNCIÓN 4: Entrenar Naive Bayes
# ============================================

def train_naive_bayes(X_train, y_train, alphas=[0.01, 0.1, 0.5, 1.0]):
    """
    Entrena MultinomialNB encontrando el mejor alpha.
    
    Args:
        X_train: Features de entrenamiento (TF-IDF)
        y_train: Labels
        alphas: Lista de valores de alpha a probar
    
    Returns:
        best_nb: Mejor modelo Naive Bayes
        best_alpha: Mejor valor de alpha
    """
    # TODO: Probar diferentes valores de alpha con cross_val_score
    
    # TODO: Encontrar el mejor alpha
    best_alpha = None
    
    # TODO: Crear y entrenar MultinomialNB con el mejor alpha
    best_nb = None
    
    return best_nb, best_alpha


# ============================================
# FUNCIÓN 5: Evaluar Modelo
# ============================================

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evalúa un modelo y retorna métricas.
    
    Args:
        model: Modelo entrenado
        X_test: Features de test
        y_test: Labels de test
        model_name: Nombre del modelo
    
    Returns:
        metrics: Dict con accuracy, precision, recall, f1
    """
    # TODO: Predecir con el modelo
    y_pred = None
    
    # TODO: Calcular métricas
    metrics = {
        'name': model_name,
        'accuracy': None,
        'precision': None,
        'recall': None,
        'f1': None
    }
    
    return metrics, y_pred


# ============================================
# FUNCIÓN 6: Comparar Modelos
# ============================================

def compare_models(results):
    """
    Crea visualización comparativa de modelos.
    
    Args:
        results: Lista de dicts con métricas de cada modelo
    
    Returns:
        fig: Figura de matplotlib
    """
    # TODO: Crear DataFrame con resultados
    df = None
    
    # TODO: Crear gráfico de barras comparando métricas
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # TODO: Guardar figura como 'model_comparison.png'
    
    return fig


# ============================================
# FUNCIÓN 7: Matriz de Confusión
# ============================================

def plot_confusion_matrix(y_true, y_pred, model_name, filename='best_model_cm.png'):
    """
    Genera y guarda matriz de confusión.
    
    Args:
        y_true: Labels reales
        y_pred: Labels predichos
        model_name: Nombre del modelo
        filename: Nombre del archivo a guardar
    """
    # TODO: Calcular matriz de confusión
    cm = None
    
    # TODO: Crear heatmap con seaborn
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # TODO: Guardar figura
    pass


# ============================================
# MAIN
# ============================================

def main():
    """Función principal del proyecto."""
    
    print("=" * 60)
    print("PROYECTO: Clasificador de Spam")
    print("=" * 60)
    
    # 1. Cargar datos
    print("\n1. Cargando datos...")
    texts, labels = load_spam_data()
    print(f"   Total muestras: {len(texts)}")
    print(f"   Distribución: Ham={sum(labels==0)}, Spam={sum(labels==1)}")
    
    # 2. Preprocesar
    print("\n2. Preprocesando datos...")
    X_train, X_test, y_train, y_test, X_train_tfidf, X_test_tfidf, vectorizer = \
        preprocess_data(texts, labels)
    print(f"   Train: {X_train_tfidf.shape}")
    print(f"   Test: {X_test_tfidf.shape}")
    
    # 3. Entrenar modelos
    print("\n3. Entrenando modelos...")
    
    print("   - KNN...")
    knn, best_k, k_scores = train_knn(X_train_tfidf, y_train)
    print(f"     Mejor k: {best_k}")
    
    print("   - SVM...")
    svm, svm_params = train_svm(X_train_tfidf, y_train)
    print(f"     Mejores params: {svm_params}")
    
    print("   - Naive Bayes...")
    nb, best_alpha = train_naive_bayes(X_train_tfidf, y_train)
    print(f"     Mejor alpha: {best_alpha}")
    
    # 4. Evaluar modelos
    print("\n4. Evaluando modelos...")
    results = []
    
    knn_metrics, knn_pred = evaluate_model(knn, X_test_tfidf, y_test, "KNN")
    results.append(knn_metrics)
    print(f"   KNN Accuracy: {knn_metrics['accuracy']:.4f}")
    
    svm_metrics, svm_pred = evaluate_model(svm, X_test_tfidf, y_test, "SVM")
    results.append(svm_metrics)
    print(f"   SVM Accuracy: {svm_metrics['accuracy']:.4f}")
    
    nb_metrics, nb_pred = evaluate_model(nb, X_test_tfidf, y_test, "Naive Bayes")
    results.append(nb_metrics)
    print(f"   NB Accuracy: {nb_metrics['accuracy']:.4f}")
    
    # 5. Comparar modelos
    print("\n5. Generando comparación...")
    compare_models(results)
    print("   Guardado: model_comparison.png")
    
    # 6. Mejor modelo
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\n🏆 MEJOR MODELO: {best_result['name']}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")
    print(f"   Precision: {best_result['precision']:.4f}")
    print(f"   Recall: {best_result['recall']:.4f}")
    print(f"   F1: {best_result['f1']:.4f}")
    
    # 7. Matriz de confusión del mejor
    if best_result['name'] == 'KNN':
        best_pred = knn_pred
    elif best_result['name'] == 'SVM':
        best_pred = svm_pred
    else:
        best_pred = nb_pred
    
    plot_confusion_matrix(y_test, best_pred, best_result['name'])
    print("\n   Guardado: best_model_cm.png")
    
    # Verificar objetivo
    print("\n" + "=" * 60)
    if best_result['accuracy'] >= 0.90:
        print("✅ OBJETIVO CUMPLIDO: Accuracy >= 0.90")
    else:
        print("❌ OBJETIVO NO CUMPLIDO: Accuracy < 0.90")
    print("=" * 60)


if __name__ == "__main__":
    main()
