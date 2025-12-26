"""
🏆 Proyecto: Titanic Competition
================================
Pipeline completo de ML para predecir supervivencia en el Titanic.

Instrucciones:
- Completa los TODOs en cada sección
- Ejecuta el código para verificar que funciona
- Genera la submission final

Autor: [Tu nombre]
Fecha: [Fecha]
"""

# ============================================
# IMPORTS
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.max_columns', None)
RANDOM_STATE = 42

print('=== Titanic Competition ===\n')


# ============================================
# SECCIÓN 1: CARGAR DATOS
# ============================================
print('--- 1. CARGAR DATOS ---')

# URLs del dataset
URL_TRAIN = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# TODO: Cargar datos de entrenamiento
# train = ...
# print(f'Train shape: {train.shape}')

# Para simular test de Kaggle (sin labels)
# TODO: Crear split para simular escenario real
# train, test = train_test_split(..., test_size=0.3, random_state=RANDOM_STATE)
# test_ids = test['PassengerId'].copy()
# y_test_real = test['Survived'].copy()  # Solo para verificar
# test = test.drop('Survived', axis=1)

print()


# ============================================
# SECCIÓN 2: EDA (Exploratory Data Analysis)
# ============================================
print('--- 2. EDA ---')

# TODO: Análisis de missing values
# missing = train.isnull().sum()
# print('Missing values:')
# print(missing[missing > 0])

# TODO: Distribución del target
# print(f'\nBalance de clases:')
# print(train['Survived'].value_counts(normalize=True))

# TODO: Tasa de supervivencia por sexo
# print('\nSupervivencia por sexo:')
# print(train.groupby('Sex')['Survived'].mean())

# TODO: Tasa de supervivencia por clase
# print('\nSupervivencia por clase:')
# print(train.groupby('Pclass')['Survived'].mean())

# TODO: Visualizaciones (opcional pero recomendado)
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# ...
# plt.tight_layout()
# plt.show()

print()


# ============================================
# SECCIÓN 3: FEATURE ENGINEERING
# ============================================
print('--- 3. FEATURE ENGINEERING ---')

# TODO: Definir función para crear nuevas features
def create_features(df):
    """
    Crea nuevas features a partir del dataframe original.
    
    Args:
        df: DataFrame con datos del Titanic
    
    Returns:
        DataFrame con nuevas features
    """
    data = df.copy()
    
    # TODO: FamilySize = SibSp + Parch + 1
    # data['FamilySize'] = ...
    
    # TODO: IsAlone = 1 si FamilySize == 1, 0 si no
    # data['IsAlone'] = ...
    
    # TODO: Extraer Title del nombre
    # data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    # Agrupar títulos raros
    # rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 
    #                'Rev', 'Sir', 'Jonkheer', 'Dona']
    # data['Title'] = data['Title'].replace(rare_titles, 'Rare')
    # data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
    # data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
    # TODO: AgeGroup (bins)
    # data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 60, 100],
    #                          labels=['Child', 'Teen', 'Young', 'Middle', 'Senior'])
    
    # TODO: HasCabin (tiene información de cabina)
    # data['HasCabin'] = data['Cabin'].notna().astype(int)
    
    return data


# TODO: Aplicar feature engineering al train
# train_fe = create_features(train)
# print(f'Features después de FE: {train_fe.columns.tolist()}')

print()


# ============================================
# SECCIÓN 4: PREPROCESAMIENTO
# ============================================
print('--- 4. PREPROCESAMIENTO ---')

# TODO: Definir columnas numéricas y categóricas
# numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
# categorical_features = ['Sex', 'Pclass', 'Embarked', 'Title']

# TODO: Crear transformador para numéricas
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# TODO: Crear transformador para categóricas
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
# ])

# TODO: Combinar con ColumnTransformer
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

print()


# ============================================
# SECCIÓN 5: BASELINE
# ============================================
print('--- 5. BASELINE ---')

# TODO: Preparar X e y
# feature_cols = numeric_features + categorical_features
# X_train = train_fe[feature_cols]
# y_train = train_fe['Survived']

# TODO: Baseline con DummyClassifier
# dummy = DummyClassifier(strategy='most_frequent')
# dummy_scores = cross_val_score(dummy, X_train, y_train, cv=5, scoring='accuracy')
# print(f'Baseline (Dummy): {dummy_scores.mean():.4f} ± {dummy_scores.std():.4f}')

print()


# ============================================
# SECCIÓN 6: COMPARACIÓN DE MODELOS
# ============================================
print('--- 6. COMPARACIÓN DE MODELOS ---')

# TODO: Definir modelos a comparar
# models = {
#     'Logistic Regression': LogisticRegression(max_iter=1000),
#     'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
#     'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
#     'SVM': SVC(random_state=RANDOM_STATE)
# }

# TODO: Crear pipelines y evaluar
# results = {}
# for name, model in models.items():
#     pipe = Pipeline([
#         ('preprocessor', preprocessor),
#         ('classifier', model)
#     ])
#     scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
#     results[name] = {'mean': scores.mean(), 'std': scores.std()}
#     print(f'{name}: {scores.mean():.4f} ± {scores.std():.4f}')

# TODO: Seleccionar mejor modelo
# best_model_name = max(results, key=lambda x: results[x]['mean'])
# print(f'\nMejor modelo: {best_model_name}')

print()


# ============================================
# SECCIÓN 7: OPTIMIZACIÓN DE HIPERPARÁMETROS
# ============================================
print('--- 7. OPTIMIZACIÓN ---')

# TODO: GridSearchCV para el mejor modelo
# param_grid = {
#     'classifier__n_estimators': [100, 200],
#     'classifier__max_depth': [5, 10, None],
#     'classifier__min_samples_split': [2, 5]
# }

# best_pipe = Pipeline([
#     ('preprocessor', preprocessor),
#     ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
# ])

# grid_search = GridSearchCV(
#     best_pipe,
#     param_grid,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1
# )

# grid_search.fit(X_train, y_train)

# print(f'Mejores parámetros: {grid_search.best_params_}')
# print(f'Mejor score CV: {grid_search.best_score_:.4f}')

print()


# ============================================
# SECCIÓN 8: MODELO FINAL
# ============================================
print('--- 8. MODELO FINAL ---')

# TODO: Entrenar modelo final con mejores parámetros
# final_model = grid_search.best_estimator_
# # O crear uno nuevo con los mejores parámetros
# final_model.fit(X_train, y_train)

# TODO: Evaluar en datos de validación (si tienes split adicional)
# train_pred = final_model.predict(X_train)
# print(f'Accuracy en train: {accuracy_score(y_train, train_pred):.4f}')

print()


# ============================================
# SECCIÓN 9: PREPARAR TEST Y PREDECIR
# ============================================
print('--- 9. PREDICCIONES EN TEST ---')

# TODO: Aplicar mismas transformaciones al test
# test_fe = create_features(test)
# X_test = test_fe[feature_cols]

# TODO: Generar predicciones
# predictions = final_model.predict(X_test)

# print(f'Predicciones shape: {predictions.shape}')
# print(f'Distribución: {pd.Series(predictions).value_counts().to_dict()}')

print()


# ============================================
# SECCIÓN 10: CREAR SUBMISSION
# ============================================
print('--- 10. CREAR SUBMISSION ---')

# TODO: Crear DataFrame de submission
# submission = pd.DataFrame({
#     'PassengerId': test_ids,
#     'Survived': predictions
# })

# TODO: Verificar formato
# print('Submission preview:')
# print(submission.head())

# TODO: Guardar CSV
# submission.to_csv('../submissions/submission.csv', index=False)
# print('\n✅ Submission guardada en submissions/submission.csv')

# TODO: Verificar score (solo porque tenemos y_test_real)
# print(f'\nScore en test: {accuracy_score(y_test_real, predictions):.4f}')

print()


# ============================================
# RESUMEN FINAL
# ============================================
print('--- RESUMEN FINAL ---')

# TODO: Completar el resumen con tus resultados
print('''
===========================================
RESUMEN DEL PROYECTO
===========================================

📊 EDA:
- Missing values: Age (~20%), Cabin (~77%), Embarked (<1%)
- Desbalance de clases: 62% no sobrevivió, 38% sobrevivió
- Factores clave: Sexo, Clase, Edad

🔧 Feature Engineering:
- Features creadas: FamilySize, IsAlone, Title, HasCabin, AgeGroup
- Encoding: OneHotEncoder para categóricas
- Escalado: StandardScaler para numéricas

🤖 Modelado:
- Baseline: ~0.62
- Mejor modelo: [Completar]
- Score CV final: [Completar]

📈 Mejoras aplicadas:
- [Listar técnicas usadas]

💡 Lecciones aprendidas:
- [Tus observaciones]

===========================================
''')

print('=== Proyecto completado ===')
