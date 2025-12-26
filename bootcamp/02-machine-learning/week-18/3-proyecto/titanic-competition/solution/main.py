"""
🏆 Proyecto: Titanic Competition - SOLUCIÓN
===========================================
Pipeline completo de ML para predecir supervivencia en el Titanic.
"""

# ============================================
# IMPORTS
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.max_columns', None)
RANDOM_STATE = 42

print('=== Titanic Competition - SOLUCIÓN ===\n')


# ============================================
# SECCIÓN 1: CARGAR DATOS
# ============================================
print('--- 1. CARGAR DATOS ---')

URL_TRAIN = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
train_full = pd.read_csv(URL_TRAIN)
print(f'Dataset completo shape: {train_full.shape}')

# Simular escenario Kaggle: split en train/test
train, test = train_test_split(train_full, test_size=0.3, random_state=RANDOM_STATE)
test_ids = test['PassengerId'].copy()
y_test_real = test['Survived'].copy()
test = test.drop('Survived', axis=1)

print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')
print()


# ============================================
# SECCIÓN 2: EDA
# ============================================
print('--- 2. EDA ---')

# Missing values
missing = train.isnull().sum()
missing_pct = (missing / len(train)) * 100
print('Missing values (%):')
print(missing_pct[missing_pct > 0].sort_values(ascending=False))

# Balance de clases
print(f'\nBalance de clases:')
print(train['Survived'].value_counts(normalize=True))

# Supervivencia por sexo
print('\nSupervivencia por sexo:')
print(train.groupby('Sex')['Survived'].mean())

# Supervivencia por clase
print('\nSupervivencia por clase:')
print(train.groupby('Pclass')['Survived'].mean())
print()


# ============================================
# SECCIÓN 3: FEATURE ENGINEERING
# ============================================
print('--- 3. FEATURE ENGINEERING ---')

def create_features(df):
    """Crea nuevas features."""
    data = df.copy()
    
    # FamilySize
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    # IsAlone
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    # Title
    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 
                   'Rev', 'Sir', 'Jonkheer', 'Dona']
    data['Title'] = data['Title'].replace(rare_titles, 'Rare')
    data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
    # HasCabin
    data['HasCabin'] = data['Cabin'].notna().astype(int)
    
    return data

train_fe = create_features(train)
print(f'Nuevas features: FamilySize, IsAlone, Title, HasCabin')
print(f'Titles encontrados: {train_fe["Title"].value_counts().to_dict()}')
print()


# ============================================
# SECCIÓN 4: PREPROCESAMIENTO
# ============================================
print('--- 4. PREPROCESAMIENTO ---')

numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
categorical_features = ['Sex', 'Pclass', 'Embarked', 'Title', 'IsAlone', 'HasCabin']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print(f'Numéricas: {numeric_features}')
print(f'Categóricas: {categorical_features}')
print()


# ============================================
# SECCIÓN 5: BASELINE
# ============================================
print('--- 5. BASELINE ---')

feature_cols = numeric_features + categorical_features
X_train = train_fe[feature_cols]
y_train = train_fe['Survived']

dummy = DummyClassifier(strategy='most_frequent')
dummy_scores = cross_val_score(dummy, X_train, y_train, cv=5, scoring='accuracy')
baseline_score = dummy_scores.mean()
print(f'Baseline (Dummy): {baseline_score:.4f} ± {dummy_scores.std():.4f}')
print()


# ============================================
# SECCIÓN 6: COMPARACIÓN DE MODELOS
# ============================================
print('--- 6. COMPARACIÓN DE MODELOS ---')

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    'SVM': SVC(random_state=RANDOM_STATE)
}

results = {}
for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = {'mean': scores.mean(), 'std': scores.std()}
    improvement = ((scores.mean() - baseline_score) / baseline_score) * 100
    print(f'{name}: {scores.mean():.4f} ± {scores.std():.4f} | Mejora: {improvement:+.1f}%')

best_model_name = max(results, key=lambda x: results[x]['mean'])
print(f'\n🏆 Mejor modelo: {best_model_name}')
print()


# ============================================
# SECCIÓN 7: OPTIMIZACIÓN DE HIPERPARÁMETROS
# ============================================
print('--- 7. OPTIMIZACIÓN ---')

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [5, 10, 15, None],
    'classifier__min_samples_split': [2, 5, 10]
}

best_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=RANDOM_STATE))
])

grid_search = GridSearchCV(
    best_pipe,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

print('Ejecutando GridSearchCV...')
grid_search.fit(X_train, y_train)

print(f'Mejores parámetros: {grid_search.best_params_}')
print(f'Mejor score CV: {grid_search.best_score_:.4f}')
print()


# ============================================
# SECCIÓN 8: MODELO FINAL
# ============================================
print('--- 8. MODELO FINAL ---')

final_model = grid_search.best_estimator_

train_pred = final_model.predict(X_train)
print(f'Accuracy en train: {accuracy_score(y_train, train_pred):.4f}')
print()


# ============================================
# SECCIÓN 9: PREDICCIONES EN TEST
# ============================================
print('--- 9. PREDICCIONES EN TEST ---')

test_fe = create_features(test)
X_test = test_fe[feature_cols]

predictions = final_model.predict(X_test)

print(f'Predicciones shape: {predictions.shape}')
print(f'Distribución: {pd.Series(predictions).value_counts(normalize=True).to_dict()}')
print()


# ============================================
# SECCIÓN 10: CREAR SUBMISSION
# ============================================
print('--- 10. CREAR SUBMISSION ---')

submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': predictions.astype(int)
})

print('Submission preview:')
print(submission.head())

submission.to_csv('../submissions/submission.csv', index=False)
print('\n✅ Submission guardada en submissions/submission.csv')

# Verificar score (tenemos y_test_real para verificar)
test_accuracy = accuracy_score(y_test_real, predictions)
print(f'\n📊 Score en test: {test_accuracy:.4f}')
print()


# ============================================
# ANÁLISIS DE ERRORES
# ============================================
print('--- ANÁLISIS DE ERRORES ---')

print('\nClassification Report:')
print(classification_report(y_test_real, predictions, 
                           target_names=['No Survived', 'Survived']))

print('Confusion Matrix:')
print(confusion_matrix(y_test_real, predictions))


# ============================================
# FEATURE IMPORTANCE
# ============================================
print('\n--- FEATURE IMPORTANCE ---')

# Obtener nombres de features después del preprocessing
feature_names = (
    numeric_features + 
    list(final_model.named_steps['preprocessor']
         .transformers_[1][1]
         .named_steps['encoder']
         .get_feature_names_out(categorical_features))
)

# Obtener importancias
clf = final_model.named_steps['classifier']
importances = clf.feature_importances_

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print('Top 10 features:')
print(importance_df.head(10).to_string(index=False))


# ============================================
# RESUMEN FINAL
# ============================================
print('\n--- RESUMEN FINAL ---')

print(f'''
===========================================
RESUMEN DEL PROYECTO
===========================================

📊 EDA:
- Missing values: Age (~20%), Cabin (~77%), Embarked (<1%)
- Desbalance de clases: 62% no sobrevivió, 38% sobrevivió
- Factores clave: Sexo, Clase, Edad

🔧 Feature Engineering:
- Features creadas: FamilySize, IsAlone, Title, HasCabin
- Encoding: OneHotEncoder para categóricas
- Escalado: StandardScaler para numéricas

🤖 Modelado:
- Baseline: {baseline_score:.4f}
- Mejor modelo: Gradient Boosting
- Score CV final: {grid_search.best_score_:.4f}
- Score test: {test_accuracy:.4f}

📈 Mejoras sobre baseline: {((grid_search.best_score_ - baseline_score) / baseline_score) * 100:.1f}%

💡 Lecciones aprendidas:
- El sexo es el factor más predictivo
- Title captura información útil adicional
- Pipelines evitan data leakage

===========================================
''')

print('=== Proyecto completado ===')
