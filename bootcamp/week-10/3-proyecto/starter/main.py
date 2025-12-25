"""
🏠 Proyecto: Predicción de Precios de Casas
===========================================
Dataset: California Housing (sklearn)
Objetivo: R² ≥ 0.60 en test

Instrucciones:
- Completa cada función marcada con TODO
- Ejecuta el script completo al finalizar
- Genera los gráficos requeridos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================
# PARTE 1: CARGA Y EXPLORACIÓN (EDA)
# ============================================

def load_data():
    """
    Carga el dataset California Housing.
    
    Returns:
        df: DataFrame con features y target
        feature_names: Lista de nombres de features
    """
    # TODO: Cargar dataset con fetch_california_housing()
    # TODO: Crear DataFrame con features y columna 'Price' para target
    # TODO: Retornar df y feature_names
    
    # Hint:
    # housing = fetch_california_housing()
    # df = pd.DataFrame(housing.data, columns=housing.feature_names)
    # df['Price'] = housing.target
    
    pass


def explore_data(df):
    """
    Realiza análisis exploratorio del dataset.
    
    Args:
        df: DataFrame con los datos
    """
    # TODO: Imprimir shape del dataset
    # TODO: Imprimir estadísticas descriptivas (describe())
    # TODO: Imprimir información de tipos de datos (info())
    # TODO: Verificar valores nulos
    
    pass


def plot_distributions(df, feature_names):
    """
    Visualiza distribución de cada feature.
    
    Args:
        df: DataFrame con los datos
        feature_names: Lista de nombres de features
    
    Guarda: 'distribucion_features.png'
    """
    # TODO: Crear figura con subplots (3 filas, 3 columnas)
    # TODO: Histograma para cada feature + Price
    # TODO: Guardar figura como 'distribucion_features.png'
    
    pass


def plot_correlation_matrix(df):
    """
    Visualiza matriz de correlación.
    
    Args:
        df: DataFrame con los datos
    
    Guarda: 'correlacion_matrix.png'
    """
    # TODO: Calcular matriz de correlación
    # TODO: Crear heatmap con seaborn
    # TODO: Guardar figura como 'correlacion_matrix.png'
    
    pass


# ============================================
# PARTE 2: PREPROCESAMIENTO
# ============================================

def prepare_data(df):
    """
    Prepara los datos para modelado.
    
    Args:
        df: DataFrame con los datos
    
    Returns:
        X_train, X_test, y_train, y_test: Datos divididos
        X_train_scaled, X_test_scaled: Datos escalados
        scaler: Objeto StandardScaler ajustado
    """
    # TODO: Separar features (X) y target (y)
    # TODO: Dividir en train/test (80/20, random_state=42)
    # TODO: Crear y ajustar StandardScaler en train
    # TODO: Transformar train y test
    # TODO: Retornar todos los elementos
    
    pass


def check_multicollinearity(df, feature_names):
    """
    Analiza multicolinealidad entre features.
    
    Args:
        df: DataFrame con los datos
        feature_names: Lista de nombres de features
    
    Imprime pares de features con correlación > 0.7
    """
    # TODO: Calcular matriz de correlación solo de features
    # TODO: Encontrar pares con |correlación| > 0.7
    # TODO: Imprimir pares problemáticos
    
    pass


# ============================================
# PARTE 3: MODELADO
# ============================================

def train_linear_regression(X_train, y_train):
    """
    Entrena modelo de regresión lineal básico.
    
    Args:
        X_train: Features de entrenamiento (escaladas)
        y_train: Target de entrenamiento
    
    Returns:
        model: Modelo entrenado
    """
    # TODO: Crear y entrenar LinearRegression
    # TODO: Retornar modelo
    
    pass


def train_ridge_cv(X_train, y_train, alphas=None):
    """
    Entrena Ridge con cross-validation para encontrar mejor alpha.
    
    Args:
        X_train: Features de entrenamiento (escaladas)
        y_train: Target de entrenamiento
        alphas: Lista de alphas a probar
    
    Returns:
        model: Modelo entrenado con mejor alpha
    """
    # TODO: Definir alphas por defecto si es None
    # TODO: Crear y entrenar RidgeCV con cv=5
    # TODO: Imprimir mejor alpha encontrado
    # TODO: Retornar modelo
    
    pass


def train_lasso_cv(X_train, y_train, alphas=None):
    """
    Entrena Lasso con cross-validation para encontrar mejor alpha.
    
    Args:
        X_train: Features de entrenamiento (escaladas)
        y_train: Target de entrenamiento
        alphas: Lista de alphas a probar
    
    Returns:
        model: Modelo entrenado con mejor alpha
    """
    # TODO: Definir alphas por defecto si es None
    # TODO: Crear y entrenar LassoCV con cv=5
    # TODO: Imprimir mejor alpha encontrado
    # TODO: Retornar modelo
    
    pass


# ============================================
# PARTE 4: EVALUACIÓN
# ============================================

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evalúa un modelo y retorna métricas.
    
    Args:
        model: Modelo entrenado
        X_test: Features de test (escaladas)
        y_test: Target de test
        model_name: Nombre del modelo para imprimir
    
    Returns:
        dict: Diccionario con R², MAE, RMSE
    """
    # TODO: Hacer predicciones
    # TODO: Calcular R², MAE, RMSE
    # TODO: Imprimir resultados
    # TODO: Retornar diccionario con métricas
    
    pass


def compare_models(results):
    """
    Compara resultados de múltiples modelos.
    
    Args:
        results: Dict {nombre_modelo: {métricas}}
    
    Imprime tabla comparativa y mejor modelo.
    """
    # TODO: Crear DataFrame con resultados
    # TODO: Imprimir tabla formateada
    # TODO: Identificar mejor modelo por R²
    
    pass


def plot_predictions(y_test, predictions_dict):
    """
    Visualiza predicciones vs valores reales.
    
    Args:
        y_test: Valores reales
        predictions_dict: Dict {nombre_modelo: predicciones}
    
    Guarda: 'predicciones_vs_real.png'
    """
    # TODO: Crear figura con subplots (1 por modelo)
    # TODO: Scatter plot de predicciones vs real
    # TODO: Añadir línea diagonal (predicción perfecta)
    # TODO: Guardar figura
    
    pass


def analyze_coefficients(models_dict, feature_names):
    """
    Analiza e imprime coeficientes de cada modelo.
    
    Args:
        models_dict: Dict {nombre_modelo: modelo}
        feature_names: Lista de nombres de features
    """
    # TODO: Para cada modelo, imprimir coeficientes ordenados por magnitud
    # TODO: Identificar features más importantes
    # TODO: Comparar qué features Lasso pone a 0
    
    pass


def plot_feature_importance(models_dict, feature_names):
    """
    Visualiza importancia de features por modelo.
    
    Args:
        models_dict: Dict {nombre_modelo: modelo}
        feature_names: Lista de nombres de features
    
    Guarda: 'importancia_features.png'
    """
    # TODO: Crear gráfico de barras comparando |coeficientes|
    # TODO: Guardar figura
    
    pass


# ============================================
# PARTE 5: CONCLUSIONES
# ============================================

def print_conclusions(results, models_dict, feature_names):
    """
    Imprime conclusiones del análisis.
    
    Args:
        results: Resultados de evaluación
        models_dict: Modelos entrenados
        feature_names: Nombres de features
    """
    # TODO: Responder las siguientes preguntas:
    
    print("\n" + "="*60)
    print("CONCLUSIONES")
    print("="*60)
    
    # 1. ¿Qué modelo funciona mejor y por qué?
    # TODO: Imprimir respuesta basada en métricas
    
    # 2. ¿Qué features son más importantes?
    # TODO: Listar top 3 features por importancia
    
    # 3. ¿Hay evidencia de multicolinealidad?
    # TODO: Comentar basado en diferencias Ridge vs Lasso
    
    # 4. ¿Cómo mejorarías el modelo?
    # TODO: Sugerir al menos 2 mejoras posibles
    
    pass


# ============================================
# MAIN
# ============================================

def main():
    """Ejecuta el pipeline completo del proyecto."""
    
    print("="*60)
    print("🏠 PROYECTO: PREDICCIÓN DE PRECIOS DE CASAS")
    print("="*60)
    
    # --- PARTE 1: EDA ---
    print("\n📊 PARTE 1: Exploración de Datos")
    print("-"*40)
    
    df = load_data()
    if df is None:
        print("❌ Implementa load_data() primero")
        return
    
    feature_names = [c for c in df.columns if c != 'Price']
    explore_data(df)
    plot_distributions(df, feature_names)
    plot_correlation_matrix(df)
    
    # --- PARTE 2: Preprocesamiento ---
    print("\n🔧 PARTE 2: Preprocesamiento")
    print("-"*40)
    
    result = prepare_data(df)
    if result is None:
        print("❌ Implementa prepare_data() primero")
        return
    
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = result
    check_multicollinearity(df, feature_names)
    
    # --- PARTE 3: Modelado ---
    print("\n🤖 PARTE 3: Entrenamiento de Modelos")
    print("-"*40)
    
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    
    lr_model = train_linear_regression(X_train_scaled, y_train)
    ridge_model = train_ridge_cv(X_train_scaled, y_train, alphas)
    lasso_model = train_lasso_cv(X_train_scaled, y_train, alphas)
    
    models_dict = {
        'LinearRegression': lr_model,
        'Ridge': ridge_model,
        'Lasso': lasso_model
    }
    
    # --- PARTE 4: Evaluación ---
    print("\n📈 PARTE 4: Evaluación")
    print("-"*40)
    
    results = {}
    predictions = {}
    
    for name, model in models_dict.items():
        if model is not None:
            results[name] = evaluate_model(model, X_test_scaled, y_test, name)
            predictions[name] = model.predict(X_test_scaled)
    
    if results:
        compare_models(results)
        plot_predictions(y_test, predictions)
        analyze_coefficients(models_dict, feature_names)
        plot_feature_importance(models_dict, feature_names)
    
    # --- PARTE 5: Conclusiones ---
    print_conclusions(results, models_dict, feature_names)
    
    print("\n" + "="*60)
    print("✅ Proyecto completado")
    print("="*60)


if __name__ == "__main__":
    main()
