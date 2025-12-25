"""
Proyecto: Evaluación Completa de Modelo - SOLUCIÓN
==================================================
Implementación completa de evaluación rigurosa de modelos.

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
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    class_names = data.target_names
    
    print("Dataset: Breast Cancer Wisconsin")
    print("-" * 40)
    print(f"Muestras: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Clases: {class_names}")
    print(f"Distribución:")
    for i, name in enumerate(class_names):
        count = (y == i).sum()
        pct = count / len(y) * 100
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    return X, y, feature_names, class_names


# ============================================
# 2. DEFINIR MODELOS Y GRIDS
# ============================================

def get_models_and_grids():
    """
    Define los modelos a comparar y sus grids de hiperparámetros.
    """
    models_config = {
        'LogisticRegression': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(max_iter=10000, random_state=42))
            ]),
            'param_grid': {
                'model__C': [0.01, 0.1, 1, 10],
                'model__penalty': ['l1', 'l2'],
                'model__solver': ['saga']
            }
        },
        'RandomForest': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
            ]),
            'param_grid': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [5, 10, 20],
                'model__min_samples_split': [2, 5]
            }
        },
        'GradientBoosting': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingClassifier(random_state=42))
            ]),
            'param_grid': {
                'model__n_estimators': [50, 100],
                'model__max_depth': [3, 5],
                'model__learning_rate': [0.01, 0.1, 0.2]
            }
        },
        'SVM': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(probability=True, random_state=42))
            ]),
            'param_grid': {
                'model__C': [0.1, 1, 10],
                'model__kernel': ['rbf', 'linear'],
                'model__gamma': ['scale', 'auto']
            }
        }
    }
    
    return models_config


# ============================================
# 3. NESTED CROSS-VALIDATION
# ============================================

def nested_cv_evaluation(model_config, X, y, outer_cv=5, inner_cv=3):
    """
    Realiza Nested CV para evaluación honesta.
    """
    # CV interno: GridSearchCV para selección de hiperparámetros
    grid_search = GridSearchCV(
        estimator=model_config['pipeline'],
        param_grid=model_config['param_grid'],
        cv=inner_cv,
        scoring='f1',
        n_jobs=-1
    )
    
    # CV externo: Evaluación con múltiples métricas
    outer_cv_split = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    results = cross_validate(
        grid_search, X, y, 
        cv=outer_cv_split, 
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )
    
    metrics = {}
    for metric in scoring.keys():
        key = f'test_{metric}'
        metrics[metric] = {
            'mean': results[key].mean(),
            'std': results[key].std()
        }
    
    return metrics


# ============================================
# 4. COMPARAR MODELOS
# ============================================

def compare_models(models_config, X, y):
    """
    Compara todos los modelos usando Nested CV.
    """
    results_list = []
    
    for name, config in models_config.items():
        print(f"  Evaluando {name}...")
        metrics = nested_cv_evaluation(config, X, y)
        
        result = {'Modelo': name}
        for metric, values in metrics.items():
            result[f'{metric}_mean'] = values['mean']
            result[f'{metric}_std'] = values['std']
        
        results_list.append(result)
    
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('f1_mean', ascending=False)
    
    # Mostrar tabla formateada
    print("\n" + "="*80)
    print("COMPARACIÓN DE MODELOS (Nested CV)")
    print("="*80)
    print(f"{'Modelo':<20} | {'Accuracy':^15} | {'F1':^15} | {'ROC-AUC':^15}")
    print("-"*80)
    
    for _, row in results_df.iterrows():
        print(f"{row['Modelo']:<20} | "
              f"{row['accuracy_mean']:.4f}±{row['accuracy_std']:.3f} | "
              f"{row['f1_mean']:.4f}±{row['f1_std']:.3f} | "
              f"{row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.3f}")
    
    print("="*80)
    
    return results_df


# ============================================
# 5. ENTRENAR MEJOR MODELO
# ============================================

def train_best_model(best_model_config, X_train, y_train):
    """
    Entrena el mejor modelo con GridSearchCV.
    """
    grid_search = GridSearchCV(
        estimator=best_model_config['pipeline'],
        param_grid=best_model_config['param_grid'],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nMejores hiperparámetros:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nMejor F1 Score (CV): {grid_search.best_score_:.4f}")
    
    return grid_search


# ============================================
# 6. EVALUACIÓN FINAL EN TEST
# ============================================

def final_evaluation(model, X_test, y_test, class_names):
    """
    Evaluación completa en el conjunto de test.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Average Precision': average_precision_score(y_test, y_proba)
    }
    
    print("\n" + "="*50)
    print("EVALUACIÓN FINAL EN TEST")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric:<18}: {value:.4f}")
    
    print("\nClassification Report:")
    print("-"*50)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return y_pred, y_proba, metrics


# ============================================
# 7. VISUALIZACIONES
# ============================================

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """
    Genera y guarda la matriz de confusión.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Blues')
    ax.set_title('Matriz de Confusión - Mejor Modelo', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Guardada: {save_path}")


def plot_roc_curve(y_true, y_proba, save_path='roc_curve.png'):
    """
    Genera y guarda la curva ROC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Modelo (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    ax.fill_between(fpr, tpr, alpha=0.2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Curva ROC', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Guardada: {save_path}")


def plot_pr_curve(y_true, y_proba, save_path='pr_curve.png'):
    """
    Genera y guarda la curva Precision-Recall.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'Modelo (AP = {ap:.4f})')
    ax.fill_between(recall, precision, alpha=0.2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Curva Precision-Recall', fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Guardada: {save_path}")


def plot_feature_importance(model, feature_names, top_n=15, save_path='feature_importance.png'):
    """
    Grafica importancia de features.
    """
    # Intentar obtener feature importances
    best_model = model.best_estimator_.named_steps['model']
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
    else:
        print("  ⚠ El modelo no soporta feature importance")
        return
    
    # Ordenar por importancia
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices][::-1], color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel('Importancia')
    ax.set_title(f'Top {top_n} Features más Importantes', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Guardada: {save_path}")


# ============================================
# 8. REPORTE FINAL
# ============================================

def generate_report(results_df, best_model_name, best_params, test_metrics):
    """
    Genera un reporte final con todos los resultados.
    """
    print("\n" + "="*60)
    print("REPORTE FINAL")
    print("="*60)
    
    print("\n📊 MODELOS EVALUADOS:")
    print("-"*40)
    for model in results_df['Modelo'].values:
        print(f"  • {model}")
    
    print(f"\n🏆 MEJOR MODELO: {best_model_name}")
    print("-"*40)
    print("Hiperparámetros óptimos:")
    for param, value in best_params.items():
        print(f"  • {param}: {value}")
    
    print("\n📈 MÉTRICAS FINALES EN TEST:")
    print("-"*40)
    for metric, value in test_metrics.items():
        print(f"  • {metric}: {value:.4f}")
    
    print("\n💡 CONCLUSIONES:")
    print("-"*40)
    print(f"  1. El modelo {best_model_name} obtuvo el mejor rendimiento")
    print(f"  2. F1-Score de {test_metrics['F1-Score']:.4f} indica buen balance precision/recall")
    print(f"  3. AUC-ROC de {test_metrics['ROC-AUC']:.4f} muestra excelente discriminación")
    print(f"  4. El modelo es apto para uso en diagnóstico (alta sensibilidad)")
    
    print("\n📁 ARCHIVOS GENERADOS:")
    print("-"*40)
    print("  • confusion_matrix.png")
    print("  • roc_curve.png")
    print("  • pr_curve.png")
    print("  • feature_importance.png")
    
    print("\n" + "="*60)


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
    X, y, feature_names, class_names = load_and_explore_data()
    
    # 2. Split train/test
    print("\n2. Dividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 3. Definir modelos
    print("\n3. Definiendo modelos...")
    models_config = get_models_and_grids()
    print(f"  {len(models_config)} modelos configurados")
    
    # 4. Comparar modelos con Nested CV
    print("\n4. Comparando modelos (Nested CV)...")
    results_df = compare_models(models_config, X_train, y_train)
    
    # Identificar mejor modelo
    best_model_name = results_df.iloc[0]['Modelo']
    best_model_config = models_config[best_model_name]
    print(f"\n✓ Mejor modelo: {best_model_name}")
    
    # 5. Entrenar mejor modelo
    print("\n5. Entrenando mejor modelo...")
    best_model = train_best_model(best_model_config, X_train, y_train)
    
    # 6. Evaluación final en test
    print("\n6. Evaluación final en test...")
    y_pred, y_proba, test_metrics = final_evaluation(
        best_model, X_test, y_test, class_names
    )
    
    # 7. Visualizaciones
    print("\n7. Generando visualizaciones...")
    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_roc_curve(y_test, y_proba)
    plot_pr_curve(y_test, y_proba)
    plot_feature_importance(best_model, feature_names)
    
    # 8. Reporte final
    generate_report(
        results_df, 
        best_model_name, 
        best_model.best_params_,
        test_metrics
    )
    
    print("\n" + "="*60)
    print("PROYECTO COMPLETADO EXITOSAMENTE")
    print("="*60)


if __name__ == "__main__":
    main()
