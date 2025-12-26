"""
Ejercicio 03: Formato de Submission
===================================
Generar submissions válidas para Kaggle.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ============================================
# PASO 1: Cargar Train y Test
# ============================================
print('--- Paso 1: Cargar Datos ---')

# Descomenta las siguientes líneas:
# # URLs de los datasets
# url_train = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
# 
# # Para test, normalmente se descarga de Kaggle
# # Aquí simulamos con un split del train
# train_full = pd.read_csv(url_train)
# 
# # Simular train/test split como en Kaggle
# # En realidad, debes descargar test.csv de Kaggle
# from sklearn.model_selection import train_test_split
# train, test = train_test_split(train_full, test_size=0.3, random_state=42)
# 
# # Guardar PassengerId del test (importante!)
# test_ids = test['PassengerId'].copy()
# 
# # En test real de Kaggle no hay columna 'Survived'
# y_test_real = test['Survived'].copy()  # Solo para verificar (no disponible en Kaggle)
# test = test.drop('Survived', axis=1)  # Simular que no tenemos el target
# 
# print(f'Train shape: {train.shape}')
# print(f'Test shape: {test.shape}')
# print(f'Test IDs: {len(test_ids)}')

print()


# ============================================
# PASO 2: Preprocesamiento CONSISTENTE
# ============================================
print('--- Paso 2: Preprocesamiento ---')

# Descomenta las siguientes líneas:
# def preprocess(df, fit_encoders=None):
#     """
#     Preprocesa los datos de manera consistente.
#     
#     Args:
#         df: DataFrame a procesar
#         fit_encoders: dict con encoders ya fiteados (para test)
#     
#     Returns:
#         X: Features procesadas
#         encoders: dict con encoders (para usar en test)
#     """
#     X = df.copy()
#     encoders = fit_encoders or {}
#     
#     # Seleccionar features
#     features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
#     X = X[features]
#     
#     # Imputar Age con mediana
#     if 'age_median' not in encoders:
#         encoders['age_median'] = X['Age'].median()
#     X['Age'] = X['Age'].fillna(encoders['age_median'])
#     
#     # Imputar Fare con mediana
#     if 'fare_median' not in encoders:
#         encoders['fare_median'] = X['Fare'].median()
#     X['Fare'] = X['Fare'].fillna(encoders['fare_median'])
#     
#     # Encoding Sex
#     if 'sex_encoder' not in encoders:
#         encoders['sex_encoder'] = LabelEncoder()
#         X['Sex'] = encoders['sex_encoder'].fit_transform(X['Sex'])
#     else:
#         X['Sex'] = encoders['sex_encoder'].transform(X['Sex'])
#     
#     return X, encoders
# 
# # Procesar train (fit + transform)
# X_train, encoders = preprocess(train)
# y_train = train['Survived']
# 
# # Procesar test (solo transform con mismos encoders!)
# X_test, _ = preprocess(test, fit_encoders=encoders)
# 
# print(f'X_train shape: {X_train.shape}')
# print(f'X_test shape: {X_test.shape}')
# print(f'Encoders: {list(encoders.keys())}')

print()


# ============================================
# PASO 3: Entrenar Modelo
# ============================================
print('--- Paso 3: Entrenar Modelo ---')

# Descomenta las siguientes líneas:
# # Cross-validation en train
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
# print(f'CV Score: {scores.mean():.4f} ± {scores.std():.4f}')
# 
# # Entrenar en TODO el train
# model.fit(X_train, y_train)
# print('Modelo entrenado en todo el train set')

print()


# ============================================
# PASO 4: Generar Predicciones
# ============================================
print('--- Paso 4: Generar Predicciones ---')

# Descomenta las siguientes líneas:
# # Predecir en test
# predictions = model.predict(X_test)
# 
# print(f'Predicciones shape: {predictions.shape}')
# print(f'Valores únicos: {np.unique(predictions, return_counts=True)}')
# print(f'Distribución: {pd.Series(predictions).value_counts(normalize=True).to_dict()}')

print()


# ============================================
# PASO 5: Crear Submission
# ============================================
print('--- Paso 5: Crear Submission ---')

# Descomenta las siguientes líneas:
# # Formato requerido por Kaggle Titanic:
# # PassengerId,Survived
# # 892,0
# # 893,1
# # ...
# 
# submission = pd.DataFrame({
#     'PassengerId': test_ids,
#     'Survived': predictions
# })
# 
# print('=== Submission Preview ===')
# print(submission.head(10))
# print(f'\nShape: {submission.shape}')

print()


# ============================================
# PASO 6: Validar Submission
# ============================================
print('--- Paso 6: Validar Submission ---')

# Descomenta las siguientes líneas:
# def validate_submission(sub_df, expected_rows):
#     """Valida que la submission tenga el formato correcto."""
#     errors = []
#     
#     # Verificar columnas
#     expected_cols = ['PassengerId', 'Survived']
#     if list(sub_df.columns) != expected_cols:
#         errors.append(f'Columnas incorrectas. Esperado: {expected_cols}, Actual: {list(sub_df.columns)}')
#     
#     # Verificar número de filas
#     if len(sub_df) != expected_rows:
#         errors.append(f'Número de filas incorrecto. Esperado: {expected_rows}, Actual: {len(sub_df)}')
#     
#     # Verificar tipos de datos
#     if sub_df['PassengerId'].dtype not in ['int64', 'int32']:
#         errors.append(f'PassengerId debe ser entero, es: {sub_df["PassengerId"].dtype}')
#     
#     if sub_df['Survived'].dtype not in ['int64', 'int32']:
#         errors.append(f'Survived debe ser entero, es: {sub_df["Survived"].dtype}')
#     
#     # Verificar valores de Survived
#     if not set(sub_df['Survived'].unique()).issubset({0, 1}):
#         errors.append(f'Survived debe ser 0 o 1, valores encontrados: {sub_df["Survived"].unique()}')
#     
#     # Verificar missing values
#     if sub_df.isnull().any().any():
#         errors.append('Hay valores nulos en la submission')
#     
#     # Verificar duplicados en PassengerId
#     if sub_df['PassengerId'].duplicated().any():
#         errors.append('Hay PassengerId duplicados')
#     
#     if errors:
#         print('❌ Errores encontrados:')
#         for e in errors:
#             print(f'  - {e}')
#         return False
#     else:
#         print('✅ Submission válida!')
#         return True
# 
# # Validar
# is_valid = validate_submission(submission, len(test_ids))

print()


# ============================================
# PASO 7: Guardar Submission
# ============================================
print('--- Paso 7: Guardar Submission ---')

# Descomenta las siguientes líneas:
# # Guardar CSV
# submission.to_csv('submission.csv', index=False)
# 
# # Verificar archivo guardado
# saved_sub = pd.read_csv('submission.csv')
# print('=== Archivo guardado: submission.csv ===')
# print(saved_sub.head())
# print(f'\nFormato CSV:')
# with open('submission.csv', 'r') as f:
#     for i, line in enumerate(f):
#         if i < 5:
#             print(line.strip())
#         else:
#             break

print()


# ============================================
# PASO 8: Verificar Score (solo simulación)
# ============================================
print('--- Paso 8: Verificar Score (simulación) ---')

# Descomenta las siguientes líneas:
# # En Kaggle no tienes las etiquetas reales del test
# # Aquí podemos verificar porque simulamos el split
# from sklearn.metrics import accuracy_score
# 
# accuracy = accuracy_score(y_test_real, predictions)
# print(f'Accuracy en test: {accuracy:.4f}')
# print('(En Kaggle real, solo verás tu score después de hacer submit)')

print()


# ============================================
# PASO 9: Errores Comunes
# ============================================
print('--- Paso 9: Errores Comunes ---')

# Descomenta las siguientes líneas:
# print('''
# ⚠️ ERRORES COMUNES A EVITAR:
# 
# 1. Data Leakage
#    ❌ MAL: Fitear encoder/scaler en train+test juntos
#    ✅ BIEN: Fitear SOLO en train, transformar test con mismos parámetros
# 
# 2. Preprocesamiento Inconsistente
#    ❌ MAL: Usar mediana de test para imputar test
#    ✅ BIEN: Usar mediana de train para imputar ambos
# 
# 3. Columnas Faltantes
#    ❌ MAL: Submission sin PassengerId
#    ✅ BIEN: Siempre incluir el ID requerido
# 
# 4. Tipos de Datos
#    ❌ MAL: Survived como float (0.0, 1.0)
#    ✅ BIEN: Survived como int (0, 1)
# 
# 5. Valores Incorrectos
#    ❌ MAL: Survived con valores 1, 2 o probabilidades
#    ✅ BIEN: Survived solo 0 o 1
# 
# 6. Filas Faltantes/Extra
#    ❌ MAL: Diferente número de filas que test
#    ✅ BIEN: Exactamente mismo número de filas
# ''')

print()
print('=== Ejercicio completado ===')
