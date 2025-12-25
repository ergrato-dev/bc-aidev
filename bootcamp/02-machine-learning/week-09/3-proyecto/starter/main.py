"""
Proyecto Semana 09: Predicción de Supervivencia en el Titanic
=============================================================

Objetivo: Aplicar el flujo completo de ML para clasificación binaria.

Instrucciones:
1. Completa cada función siguiendo los TODOs
2. Ejecuta el script para verificar tus resultados
3. El modelo debe alcanzar al menos 75% de accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Configuración
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')


# ============================================
# PARTE 1: CARGA Y EXPLORACIÓN DE DATOS
# ============================================

def cargar_datos():
    """
    Carga el dataset Titanic desde seaborn.
    
    Returns:
        pd.DataFrame: Dataset Titanic
    """
    # TODO: Cargar el dataset titanic usando seaborn
    # Hint: sns.load_dataset('titanic')
    df = None  # TODO: Implementar
    
    return df


def explorar_datos(df: pd.DataFrame) -> None:
    """
    Realiza exploración inicial del dataset.
    
    Args:
        df: DataFrame con los datos
    """
    print('=' * 60)
    print('EXPLORACIÓN DE DATOS')
    print('=' * 60)
    
    # TODO: Mostrar las primeras 5 filas
    # Hint: df.head()
    print('\nPrimeras filas:')
    # TODO: Implementar
    
    # TODO: Mostrar shape del dataset
    print('\nShape del dataset:')
    # TODO: Implementar
    
    # TODO: Mostrar tipos de datos
    print('\nTipos de datos:')
    # TODO: Implementar
    
    # TODO: Mostrar estadísticas descriptivas
    print('\nEstadísticas descriptivas:')
    # TODO: Implementar
    
    # TODO: Mostrar cantidad de valores nulos por columna
    print('\nValores nulos por columna:')
    # TODO: Implementar


def analizar_target(df: pd.DataFrame) -> None:
    """
    Analiza la distribución de la variable target (survived).
    
    Args:
        df: DataFrame con los datos
    """
    print('\n' + '=' * 60)
    print('ANÁLISIS DEL TARGET')
    print('=' * 60)
    
    # TODO: Mostrar distribución de survived (value_counts)
    print('\nDistribución de supervivencia:')
    # TODO: Implementar
    
    # TODO: Calcular porcentaje de supervivencia
    print('\nPorcentaje de supervivencia:')
    # TODO: Implementar
    
    # TODO: Verificar si está balanceado
    # Hint: Si una clase tiene más del 60%, está desbalanceado


# ============================================
# PARTE 2: PREPARACIÓN DE DATOS
# ============================================

def preparar_datos(df: pd.DataFrame) -> tuple:
    """
    Prepara los datos para el modelo.
    
    - Maneja valores nulos
    - Codifica variables categóricas
    - Selecciona features
    - Divide en train/test
    
    Args:
        df: DataFrame original
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print('\n' + '=' * 60)
    print('PREPARACIÓN DE DATOS')
    print('=' * 60)
    
    # Crear copia para no modificar original
    data = df.copy()
    
    # --- MANEJO DE VALORES NULOS ---
    
    # TODO: Imputar valores nulos en 'age' con la mediana
    # Hint: data['age'].fillna(data['age'].median(), inplace=True)
    print('\nImputando valores nulos en age...')
    # TODO: Implementar
    
    # TODO: Eliminar filas con valores nulos en 'embarked'
    # Hint: data.dropna(subset=['embarked'], inplace=True)
    print('Eliminando filas con embarked nulo...')
    # TODO: Implementar
    
    # --- CODIFICACIÓN DE VARIABLES CATEGÓRICAS ---
    
    # TODO: Codificar 'sex' (male=0, female=1)
    # Hint: Puedes usar map({'male': 0, 'female': 1}) o LabelEncoder
    print('\nCodificando variable sex...')
    # TODO: Implementar
    
    # TODO: Codificar 'embarked' usando LabelEncoder
    print('Codificando variable embarked...')
    # TODO: Implementar
    
    # --- SELECCIÓN DE FEATURES ---
    
    # TODO: Seleccionar las features para el modelo
    # Features sugeridas: pclass, sex, age, sibsp, parch, fare, embarked
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    
    # TODO: Crear X (features) e y (target)
    X = None  # TODO: data[features]
    y = None  # TODO: data['survived']
    
    print(f'\nFeatures seleccionadas: {features}')
    print(f'Shape de X: {X.shape if X is not None else "No definido"}')
    
    # --- DIVISIÓN TRAIN/TEST ---
    
    # TODO: Dividir datos en train (80%) y test (20%)
    # Hint: usar stratify=y para mantener proporciones
    X_train, X_test, y_train, y_test = None, None, None, None  # TODO: Implementar
    
    print(f'\nTrain set: {len(X_train) if X_train is not None else 0} samples')
    print(f'Test set: {len(X_test) if X_test is not None else 0} samples')
    
    return X_train, X_test, y_train, y_test


# ============================================
# PARTE 3: MODELADO
# ============================================

def entrenar_modelo(X_train, y_train):
    """
    Entrena un modelo de clasificación.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Labels de entrenamiento
        
    Returns:
        modelo: Modelo entrenado
    """
    print('\n' + '=' * 60)
    print('ENTRENAMIENTO DEL MODELO')
    print('=' * 60)
    
    # TODO: Crear el modelo KNeighborsClassifier
    # Hint: Puedes probar diferentes valores de n_neighbors (3, 5, 7)
    modelo = None  # TODO: Implementar
    
    # TODO: Entrenar el modelo con fit()
    # TODO: Implementar
    
    print(f'Modelo: {type(modelo).__name__ if modelo else "No definido"}')
    print('✅ Modelo entrenado')
    
    return modelo


def hacer_predicciones(modelo, X_test):
    """
    Realiza predicciones con el modelo entrenado.
    
    Args:
        modelo: Modelo entrenado
        X_test: Features de test
        
    Returns:
        np.array: Predicciones
    """
    # TODO: Usar el modelo para predecir
    # Hint: modelo.predict(X_test)
    y_pred = None  # TODO: Implementar
    
    return y_pred


# ============================================
# PARTE 4: EVALUACIÓN
# ============================================

def evaluar_modelo(y_test, y_pred) -> dict:
    """
    Evalúa el modelo con múltiples métricas.
    
    Args:
        y_test: Labels reales
        y_pred: Labels predichos
        
    Returns:
        dict: Diccionario con métricas
    """
    print('\n' + '=' * 60)
    print('EVALUACIÓN DEL MODELO')
    print('=' * 60)
    
    # TODO: Calcular accuracy
    accuracy = None  # TODO: accuracy_score(y_test, y_pred)
    
    print(f'\nAccuracy: {accuracy:.4f if accuracy else "No calculado"}')
    
    # TODO: Generar y mostrar matriz de confusión
    print('\nMatriz de Confusión:')
    # TODO: Implementar
    
    # TODO: Mostrar classification report
    print('\nClassification Report:')
    # TODO: Implementar classification_report(y_test, y_pred)
    
    # Verificar si cumple el criterio mínimo
    if accuracy and accuracy >= 0.75:
        print('\n🎉 ¡El modelo cumple el criterio mínimo de 75% accuracy!')
    elif accuracy:
        print(f'\n⚠️  El modelo no alcanza el 75% requerido. Intenta mejorar.')
    
    return {
        'accuracy': accuracy,
    }


def visualizar_resultados(y_test, y_pred) -> None:
    """
    Crea visualizaciones de los resultados.
    
    Args:
        y_test: Labels reales
        y_pred: Labels predichos
    """
    # TODO: Crear visualización de la matriz de confusión
    # Hint: ConfusionMatrixDisplay
    
    # Descomentar cuando implementes:
    # fig, ax = plt.subplots(figsize=(8, 6))
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Sobrevivió', 'Sobrevivió'])
    # disp.plot(ax=ax, cmap='Blues')
    # plt.title('Matriz de Confusión - Predicción Titanic')
    # plt.tight_layout()
    # plt.savefig('confusion_matrix_titanic.png', dpi=150)
    # print('\nGráfico guardado: confusion_matrix_titanic.png')
    # plt.show()
    
    pass


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def main():
    """Ejecuta el pipeline completo de ML."""
    
    print('\n' + '=' * 60)
    print('🚢 PROYECTO: PREDICCIÓN DE SUPERVIVENCIA EN EL TITANIC')
    print('=' * 60)
    
    # Paso 1: Cargar datos
    df = cargar_datos()
    
    if df is None:
        print('\n❌ Error: No se pudo cargar el dataset.')
        print('   Implementa la función cargar_datos()')
        return
    
    # Paso 2: Explorar datos
    explorar_datos(df)
    analizar_target(df)
    
    # Paso 3: Preparar datos
    X_train, X_test, y_train, y_test = preparar_datos(df)
    
    if X_train is None:
        print('\n❌ Error: No se prepararon los datos correctamente.')
        print('   Implementa la función preparar_datos()')
        return
    
    # Paso 4: Entrenar modelo
    modelo = entrenar_modelo(X_train, y_train)
    
    if modelo is None:
        print('\n❌ Error: No se entrenó el modelo.')
        print('   Implementa la función entrenar_modelo()')
        return
    
    # Paso 5: Hacer predicciones
    y_pred = hacer_predicciones(modelo, X_test)
    
    if y_pred is None:
        print('\n❌ Error: No se generaron predicciones.')
        print('   Implementa la función hacer_predicciones()')
        return
    
    # Paso 6: Evaluar modelo
    metricas = evaluar_modelo(y_test, y_pred)
    
    # Paso 7: Visualizar (opcional)
    # visualizar_resultados(y_test, y_pred)
    
    print('\n' + '=' * 60)
    print('✅ PROYECTO COMPLETADO')
    print('=' * 60)


if __name__ == '__main__':
    main()
