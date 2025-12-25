"""
Proyecto: Evaluación Completa de Modelo
=======================================
Implementa una evaluación rigurosa de modelos de clasificación.

Dataset: Breast Cancer Wisconsin
Objetivo: Clasificar tumores como malignos o benignos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_validate,
    GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Métricas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, 
    precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay
)

import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. CARGAR Y EXPLORAR DATOS
# ============================================

def load_and_explore_data():
    """
    Carga el dataset y muestra información básica.
    
    Returns:
        X, y: Features y target
        feature_names: Nombres de las features
    """
    # TODO: Cargar breast cancer dataset
    # TODO: Mostrar información: muestras, features, distribución de clases
    # TODO: Retornar X, y, feature_names
    pass


# ============================================
# 2. DEFINIR MODELOS Y GRIDS
# ============================================

def get_models_and_grids():
    """
    Define los modelos a comparar y sus grids de hiperparámetros.
    
    Returns:
        dict: Diccionario con modelos y sus param_grids
    """
    # TODO: Definir al menos 3 modelos diferentes
    # TODO: Cada modelo debe tener un grid de hiperparámetros
    # TODO: Usar pipelines con StandardScaler
    
    models_config = {
        # 'LogisticRegression': {
        #     'pipeline': Pipeline([...]),
        #     'param_grid': {...}
        # },
        # 'RandomForest': {...},
        # 'GradientBoosting': {...}
    }
    
    return models_config


# ============================================
# 3. NESTED CROSS-VALIDATION
# ============================================

def nested_cv_evaluation(model_config, X, y, outer_cv=5, inner_cv=3):
    """
    Realiza Nested CV para evaluación honesta.
    
    Args:
        model_config: dict con 'pipeline' y 'param_grid'
        X, y: Datos
        outer_cv, inner_cv: Número de folds
    
    Returns:
        dict: Métricas con mean y std
    """
    # TODO: Implementar nested CV
    # TODO: CV interno: GridSearchCV para selección de hiperparámetros
    # TODO: CV externo: cross_val_score para evaluación
    # TODO: Calcular múltiples métricas: accuracy, f1, roc_auc
    # TODO: Retornar dict con resultados
    pass


# ============================================
# 4. COMPARAR MODELOS
# ============================================

def compare_models(models_config, X, y):
    """
    Compara todos los modelos usando Nested CV.
    
    Args:
        models_config: Diccionario de configuraciones
        X, y: Datos
    
    Returns:
        DataFrame con resultados comparativos
    """
    # TODO: Evaluar cada modelo con nested_cv_evaluation
    # TODO: Crear DataFrame con resultados
    # TODO: Ordenar por mejor métrica
    # TODO: Mostrar tabla comparativa
    pass


# ============================================
# 5. ENTRENAR MEJOR MODELO
# ============================================

def train_best_model(best_model_config, X_train, y_train):
    """
    Entrena el mejor modelo con GridSearchCV.
    
    Args:
        best_model_config: Configuración del mejor modelo
        X_train, y_train: Datos de entrenamiento
    
    Returns:
        Modelo entrenado (GridSearchCV)
    """
    # TODO: Crear GridSearchCV con la configuración del mejor modelo
    # TODO: Entrenar en X_train
    # TODO: Mostrar mejores hiperparámetros
    # TODO: Retornar modelo entrenado
    pass


# ============================================
# 6. EVALUACIÓN FINAL EN TEST
# ============================================

def final_evaluation(model, X_test, y_test, class_names):
    """
    Evaluación completa en el conjunto de test.
    
    Args:
        model: Modelo entrenado
        X_test, y_test: Datos de test
        class_names: Nombres de las clases
    """
    # TODO: Predecir en test
    # TODO: Calcular todas las métricas
    # TODO: Mostrar classification_report
    pass


# ============================================
# 7. VISUALIZACIONES
# ============================================

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """
    Genera y guarda la matriz de confusión.
    """
    # TODO: Crear matriz de confusión
    # TODO: Visualizar con ConfusionMatrixDisplay
    # TODO: Guardar figura
    pass


def plot_roc_curve(y_true, y_proba, save_path='roc_curve.png'):
    """
    Genera y guarda la curva ROC.
    """
    # TODO: Calcular FPR, TPR, AUC
    # TODO: Graficar curva ROC
    # TODO: Añadir línea diagonal de referencia
    # TODO: Guardar figura
    pass


def plot_pr_curve(y_true, y_proba, save_path='pr_curve.png'):
    """
    Genera y guarda la curva Precision-Recall.
    """
    # TODO: Calcular precision, recall, AP
    # TODO: Graficar curva PR
    # TODO: Guardar figura
    pass


def plot_feature_importance(model, feature_names, top_n=15, save_path='feature_importance.png'):
    """
    Grafica importancia de features (si el modelo lo soporta).
    """
    # TODO: Extraer importancia de features del modelo
    # TODO: Ordenar por importancia
    # TODO: Graficar top N features
    # TODO: Guardar figura
    pass


# ============================================
# 8. REPORTE FINAL
# ============================================

def generate_report(results_df, best_model_name, test_metrics):
    """
    Genera un reporte final con todos los resultados.
    """
    # TODO: Crear reporte con:
    # - Resumen de modelos comparados
    # - Mejor modelo y sus hiperparámetros
    # - Métricas finales en test
    # - Conclusiones y recomendaciones
    pass


# ============================================
# MAIN
# ============================================

def main():
    """
    Pipeline principal del proyecto.
    """
    print("="*60)
    print("PROYECTO: EVALUACIÓN COMPLETA DE MODELO")
    print("="*60)
    
    # 1. Cargar datos
    print("\n1. Cargando datos...")
    # X, y, feature_names = load_and_explore_data()
    
    # 2. Split train/test (test se reserva para evaluación final)
    print("\n2. Dividiendo datos...")
    # X_train, X_test, y_train, y_test = train_test_split(...)
    
    # 3. Definir modelos
    print("\n3. Definiendo modelos...")
    # models_config = get_models_and_grids()
    
    # 4. Comparar modelos con Nested CV
    print("\n4. Comparando modelos (Nested CV)...")
    # results_df = compare_models(models_config, X_train, y_train)
    
    # 5. Entrenar mejor modelo
    print("\n5. Entrenando mejor modelo...")
    # best_model = train_best_model(best_config, X_train, y_train)
    
    # 6. Evaluación final en test
    print("\n6. Evaluación final en test...")
    # final_evaluation(best_model, X_test, y_test, class_names)
    
    # 7. Visualizaciones
    print("\n7. Generando visualizaciones...")
    # plot_confusion_matrix(...)
    # plot_roc_curve(...)
    # plot_pr_curve(...)
    # plot_feature_importance(...)
    
    # 8. Reporte final
    print("\n8. Generando reporte...")
    # generate_report(...)
    
    print("\n" + "="*60)
    print("PROYECTO COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    main()
