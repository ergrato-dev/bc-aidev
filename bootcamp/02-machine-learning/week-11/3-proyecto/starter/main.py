"""
Proyecto: Clasificación de Vinos con Random Forest
===================================================

Objetivo: Alcanzar accuracy >= 0.92 en el dataset Wine

Dataset:
- 178 muestras
- 13 features químicas
- 3 clases de vino

Instrucciones:
- Completa cada función marcada con TODO
- Ejecuta el código para verificar resultados
- Asegúrate de alcanzar accuracy >= 0.92
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Configuración
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_ACCURACY = 0.92


# ============================================
# 1. CARGA Y EXPLORACIÓN DE DATOS
# ============================================

def load_and_explore_data():
    """
    Carga el dataset Wine y muestra información básica.

    Returns:
        X: Features (numpy array)
        y: Target (numpy array)
        feature_names: Lista de nombres de features
        target_names: Lista de nombres de clases
    """
    # TODO: Cargar el dataset Wine usando load_wine()
    # wine = ...
    # X, y = ...

    # TODO: Mostrar información del dataset
    # print(f"Shape X: ...")
    # print(f"Shape y: ...")
    # print(f"Features: ...")
    # print(f"Clases: ...")

    # TODO: Mostrar distribución de clases
    # unique, counts = np.unique(y, return_counts=True)
    # print("\nDistribución de clases:")
    # for cls, count in zip(unique, counts):
    #     print(f"  Clase {cls}: {count} muestras ({count/len(y)*100:.1f}%)")

    # TODO: Retornar X, y, feature_names, target_names
    pass


# ============================================
# 2. PREPROCESAMIENTO
# ============================================

def preprocess_data(X, y):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.

    Args:
        X: Features
        y: Target

    Returns:
        X_train, X_test, y_train, y_test
    """
    # TODO: Dividir datos usando train_test_split
    # - test_size=TEST_SIZE (0.2)
    # - random_state=RANDOM_STATE (42)
    # - stratify=y para mantener proporciones

    # X_train, X_test, y_train, y_test = train_test_split(...)

    # TODO: Mostrar tamaños de conjuntos
    # print(f"Train: {X_train.shape[0]} muestras")
    # print(f"Test: {X_test.shape[0]} muestras")

    # TODO: Retornar los conjuntos
    pass


# ============================================
# 3. MODELO BASELINE (Decision Tree)
# ============================================

def train_baseline(X_train, X_test, y_train, y_test):
    """
    Entrena un Decision Tree como modelo baseline.

    Args:
        X_train, X_test, y_train, y_test: Datos de entrenamiento y prueba

    Returns:
        tree: Modelo entrenado
        baseline_accuracy: Accuracy en test
    """
    print("\n" + "="*50)
    print("BASELINE: Decision Tree")
    print("="*50)

    # TODO: Crear DecisionTreeClassifier
    # tree = DecisionTreeClassifier(random_state=RANDOM_STATE)

    # TODO: Entrenar el modelo
    # tree.fit(...)

    # TODO: Calcular accuracy en train y test
    # train_acc = tree.score(...)
    # test_acc = tree.score(...)

    # TODO: Mostrar resultados
    # print(f"Train Accuracy: {train_acc:.4f}")
    # print(f"Test Accuracy: {test_acc:.4f}")

    # TODO: Retornar modelo y accuracy
    pass


# ============================================
# 4. RANDOM FOREST
# ============================================

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Entrena un Random Forest para alcanzar accuracy >= 0.92.

    Args:
        X_train, X_test, y_train, y_test: Datos de entrenamiento y prueba

    Returns:
        rf: Modelo entrenado
        rf_accuracy: Accuracy en test
    """
    print("\n" + "="*50)
    print("MODELO: Random Forest")
    print("="*50)

    # TODO: Crear RandomForestClassifier con hiperparámetros adecuados
    # Sugerencia: n_estimators=100, oob_score=True
    # rf = RandomForestClassifier(
    #     n_estimators=...,
    #     max_depth=...,
    #     oob_score=True,
    #     random_state=RANDOM_STATE,
    #     n_jobs=-1
    # )

    # TODO: Entrenar el modelo
    # rf.fit(...)

    # TODO: Calcular accuracy en train y test
    # train_acc = rf.score(...)
    # test_acc = rf.score(...)

    # TODO: Mostrar resultados
    # print(f"Train Accuracy: {train_acc:.4f}")
    # print(f"Test Accuracy: {test_acc:.4f}")
    # print(f"OOB Score: {rf.oob_score_:.4f}")

    # TODO: Verificar si se alcanzó el objetivo
    # if test_acc >= TARGET_ACCURACY:
    #     print(f"\n✅ ¡OBJETIVO ALCANZADO! Accuracy >= {TARGET_ACCURACY}")
    # else:
    #     print(f"\n❌ Accuracy {test_acc:.4f} < {TARGET_ACCURACY}. Ajusta hiperparámetros.")

    # TODO: Retornar modelo y accuracy
    pass


# ============================================
# 5. EVALUACIÓN COMPLETA
# ============================================

def evaluate_model(model, X_train, X_test, y_train, y_test, target_names):
    """
    Realiza evaluación completa del modelo.

    Args:
        model: Modelo entrenado
        X_train, X_test, y_train, y_test: Datos
        target_names: Nombres de las clases
    """
    print("\n" + "="*50)
    print("EVALUACIÓN COMPLETA")
    print("="*50)

    # TODO: Obtener predicciones
    # y_pred = model.predict(X_test)

    # TODO: Mostrar classification report
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred, target_names=target_names))

    # TODO: Crear y mostrar confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix:")
    # print(cm)

    # TODO: Visualizar confusion matrix
    # fig, ax = plt.subplots(figsize=(8, 6))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    # disp.plot(ax=ax, cmap='Blues')
    # plt.title('Confusion Matrix - Random Forest')
    # plt.tight_layout()
    # plt.savefig('confusion_matrix.png', dpi=150)
    # plt.show()

    pass


def cross_validate_model(model, X, y):
    """
    Realiza cross-validation del modelo.

    Args:
        model: Modelo a evaluar
        X: Features completas
        y: Target completo
    """
    print("\n" + "="*50)
    print("CROSS-VALIDATION")
    print("="*50)

    # TODO: Realizar 5-fold cross-validation
    # cv_scores = cross_val_score(model, X, y, cv=5)

    # TODO: Mostrar resultados
    # print(f"CV Scores: {cv_scores}")
    # print(f"CV Mean: {cv_scores.mean():.4f}")
    # print(f"CV Std: {cv_scores.std():.4f}")
    # print(f"CV Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")

    pass


# ============================================
# 6. FEATURE IMPORTANCE
# ============================================

def analyze_feature_importance(model, feature_names):
    """
    Analiza y visualiza la importancia de features.

    Args:
        model: Random Forest entrenado
        feature_names: Nombres de las features
    """
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)

    # TODO: Obtener importancia de features
    # importance = model.feature_importances_

    # TODO: Crear DataFrame ordenado
    # importance_df = pd.DataFrame({
    #     'feature': feature_names,
    #     'importance': importance
    # }).sort_values('importance', ascending=False)

    # TODO: Mostrar ranking
    # print("\nRanking de Features:")
    # for i, row in importance_df.iterrows():
    #     print(f"  {row['feature']}: {row['importance']:.4f}")

    # TODO: Identificar top 3
    # print("\nTop 3 Features más importantes:")
    # top3 = importance_df.head(3)
    # for i, (_, row) in enumerate(top3.iterrows(), 1):
    #     print(f"  {i}. {row['feature']} ({row['importance']:.4f})")

    # TODO: Visualizar
    # plt.figure(figsize=(10, 8))
    # plt.barh(
    #     importance_df['feature'][::-1],
    #     importance_df['importance'][::-1],
    #     color='steelblue'
    # )
    # plt.xlabel('Importancia')
    # plt.title('Feature Importance - Wine Classification')
    # plt.tight_layout()
    # plt.savefig('feature_importance.png', dpi=150)
    # plt.show()

    pass


# ============================================
# MAIN
# ============================================

def main():
    """
    Función principal que ejecuta todo el pipeline.
    """
    print("="*60)
    print("PROYECTO: CLASIFICACIÓN DE VINOS CON RANDOM FOREST")
    print("="*60)
    print(f"Objetivo: Accuracy >= {TARGET_ACCURACY}")

    # 1. Cargar datos
    print("\n[1/6] Cargando datos...")
    result = load_and_explore_data()
    if result is None:
        print("⚠️  Completa la función load_and_explore_data()")
        return
    X, y, feature_names, target_names = result

    # 2. Preprocesar
    print("\n[2/6] Preprocesando datos...")
    result = preprocess_data(X, y)
    if result is None:
        print("⚠️  Completa la función preprocess_data()")
        return
    X_train, X_test, y_train, y_test = result

    # 3. Baseline
    print("\n[3/6] Entrenando baseline...")
    result = train_baseline(X_train, X_test, y_train, y_test)
    if result is None:
        print("⚠️  Completa la función train_baseline()")
        return
    tree, baseline_acc = result

    # 4. Random Forest
    print("\n[4/6] Entrenando Random Forest...")
    result = train_random_forest(X_train, X_test, y_train, y_test)
    if result is None:
        print("⚠️  Completa la función train_random_forest()")
        return
    rf, rf_acc = result

    # 5. Evaluación
    print("\n[5/6] Evaluando modelo...")
    evaluate_model(rf, X_train, X_test, y_train, y_test, target_names)
    cross_validate_model(rf, X, y)

    # 6. Feature Importance
    print("\n[6/6] Analizando features...")
    analyze_feature_importance(rf, feature_names)

    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"Baseline (Decision Tree): {baseline_acc:.4f}")
    print(f"Random Forest: {rf_acc:.4f}")
    print(f"Mejora: {(rf_acc - baseline_acc)*100:.2f}%")
    print(f"Objetivo ({TARGET_ACCURACY}): {'✅ ALCANZADO' if rf_acc >= TARGET_ACCURACY else '❌ NO ALCANZADO'}")


if __name__ == "__main__":
    main()
