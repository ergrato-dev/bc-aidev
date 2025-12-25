# ============================================
# EJERCICIO 04: SELECCIÓN DE CARACTERÍSTICAS
# ============================================
# Objetivo: Practicar SelectKBest, RFE y 
# SelectFromModel
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================
# PASO 1: Crear Dataset con Features Irrelevantes
# ============================================
print('--- Paso 1: Crear Dataset ---')

# Descomenta las siguientes líneas:
# from sklearn.datasets import make_classification

# # Crear dataset con features informativas y ruido
# X, y = make_classification(
#     n_samples=1000,
#     n_features=20,          # 20 features totales
#     n_informative=5,        # Solo 5 son informativas
#     n_redundant=3,          # 3 son combinaciones lineales
#     n_repeated=2,           # 2 son duplicadas
#     n_classes=2,
#     random_state=42
# )

# # Crear DataFrame con nombres descriptivos
# feature_names = [f'feature_{i}' for i in range(20)]
# df = pd.DataFrame(X, columns=feature_names)
# df['target'] = y

# print(f"Dataset shape: {X.shape}")
# print(f"Features informativas: 5")
# print(f"Features redundantes: 3")
# print(f"Features repetidas: 2")
# print(f"Features ruido: 10")

# # Añadir feature constante (varianza 0)
# df['constant_feature'] = 1.0
# X = df.drop(columns=['target']).values
# feature_names = df.drop(columns=['target']).columns.tolist()

print()

# ============================================
# PASO 2: Variance Threshold
# ============================================
print('--- Paso 2: Variance Threshold ---')

# Elimina features con varianza menor al umbral
# Útil para eliminar features constantes o casi constantes

# Descomenta las siguientes líneas:
# from sklearn.feature_selection import VarianceThreshold

# # Threshold = 0 elimina features constantes
# selector_var = VarianceThreshold(threshold=0.0)
# X_var = selector_var.fit_transform(X)

# # Ver cuáles se eliminaron
# selected_mask = selector_var.get_support()
# eliminated = [name for name, sel in zip(feature_names, selected_mask) if not sel]

# print(f"Features originales: {X.shape[1]}")
# print(f"Features después de VarianceThreshold: {X_var.shape[1]}")
# print(f"Eliminadas: {eliminated}")

# # Actualizar X y feature_names
# X = X_var
# feature_names = [name for name, sel in zip(feature_names, selected_mask) if sel]

print()

# ============================================
# PASO 3: SelectKBest (Método Filter)
# ============================================
print('--- Paso 3: SelectKBest (Filter) ---')

# Descomenta las siguientes líneas:
# from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# # Seleccionar las 10 mejores features con ANOVA F-value
# selector_kbest = SelectKBest(score_func=f_classif, k=10)
# X_kbest = selector_kbest.fit_transform(X, y)

# # Ver scores
# scores_df = pd.DataFrame({
#     'feature': feature_names,
#     'f_score': selector_kbest.scores_,
#     'selected': selector_kbest.get_support()
# }).sort_values('f_score', ascending=False)

# print("Top 10 features por F-score:")
# print(scores_df.head(10))

# # Visualizar
# plt.figure(figsize=(12, 6))
# colors = ['steelblue' if sel else 'lightgray' for sel in scores_df['selected']]
# plt.barh(scores_df['feature'], scores_df['f_score'], color=colors)
# plt.xlabel('F-Score')
# plt.title('SelectKBest: F-Scores por Feature')
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig('selectkbest_scores.png', dpi=150)
# plt.show()

# # También probar con Mutual Information
# selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
# selector_mi.fit(X, y)
# print("\nTop 5 por Mutual Information:")
# mi_scores = pd.Series(selector_mi.scores_, index=feature_names).sort_values(ascending=False)
# print(mi_scores.head())

print()

# ============================================
# PASO 4: RFE - Recursive Feature Elimination
# ============================================
print('--- Paso 4: RFE (Wrapper) ---')

# Descomenta las siguientes líneas:
# from sklearn.feature_selection import RFE, RFECV
# from sklearn.linear_model import LogisticRegression

# # RFE con Logistic Regression como estimador
# estimator = LogisticRegression(max_iter=1000, random_state=42)
# rfe = RFE(estimator=estimator, n_features_to_select=10, step=1)
# X_rfe = rfe.fit_transform(X, y)

# # Ver ranking
# rfe_df = pd.DataFrame({
#     'feature': feature_names,
#     'ranking': rfe.ranking_,
#     'selected': rfe.support_
# }).sort_values('ranking')

# print("Ranking RFE (1 = seleccionada):")
# print(rfe_df)

# # RFECV - encuentra el número óptimo automáticamente
# from sklearn.model_selection import StratifiedKFold

# rfecv = RFECV(
#     estimator=estimator,
#     step=1,
#     cv=StratifiedKFold(5),
#     scoring='accuracy',
#     min_features_to_select=3
# )
# rfecv.fit(X, y)

# print(f"\nNúmero óptimo de features (RFECV): {rfecv.n_features_}")

# # Visualizar curva
# plt.figure(figsize=(10, 5))
# plt.plot(range(3, len(rfecv.cv_results_['mean_test_score']) + 3), 
#          rfecv.cv_results_['mean_test_score'])
# plt.fill_between(
#     range(3, len(rfecv.cv_results_['mean_test_score']) + 3),
#     rfecv.cv_results_['mean_test_score'] - rfecv.cv_results_['std_test_score'],
#     rfecv.cv_results_['mean_test_score'] + rfecv.cv_results_['std_test_score'],
#     alpha=0.2
# )
# plt.xlabel('Número de Features')
# plt.ylabel('CV Accuracy')
# plt.title('RFECV: Accuracy vs Número de Features')
# plt.axvline(rfecv.n_features_, color='red', linestyle='--', label=f'Óptimo: {rfecv.n_features_}')
# plt.legend()
# plt.tight_layout()
# plt.savefig('rfecv_curve.png', dpi=150)
# plt.show()

print()

# ============================================
# PASO 5: SelectFromModel (Método Embedded)
# ============================================
print('--- Paso 5: SelectFromModel (Embedded) ---')

# Descomenta las siguientes líneas:
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import RandomForestClassifier

# # Entrenar Random Forest y usar importancias
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X, y)

# # Ver importancias
# importances_df = pd.DataFrame({
#     'feature': feature_names,
#     'importance': rf.feature_importances_
# }).sort_values('importance', ascending=False)

# print("Feature Importances (Random Forest):")
# print(importances_df)

# # Visualizar
# plt.figure(figsize=(12, 6))
# plt.barh(importances_df['feature'], importances_df['importance'], color='steelblue')
# plt.xlabel('Importance')
# plt.title('Random Forest Feature Importances')
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig('rf_importances.png', dpi=150)
# plt.show()

# # SelectFromModel con threshold
# selector_model = SelectFromModel(rf, threshold='median')
# X_model = selector_model.fit_transform(X, y)

# print(f"\nFeatures seleccionadas (threshold='median'): {X_model.shape[1]}")
# print(f"Features: {[f for f, s in zip(feature_names, selector_model.get_support()) if s]}")

print()

# ============================================
# PASO 6: Comparar Métodos
# ============================================
print('--- Paso 6: Comparar Métodos ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

# # Función para evaluar
# def evaluate_selection(X, y, selector, name):
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('selector', selector),
#         ('classifier', LogisticRegression(max_iter=1000))
#     ])
#     scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
#     return {
#         'Method': name,
#         'CV Accuracy': f"{scores.mean():.4f} ± {scores.std():.4f}",
#         'n_features': selector.fit(X, y).transform(X).shape[1] if hasattr(selector, 'fit') else 'N/A'
#     }

# # Evaluar diferentes métodos
# results = []

# # Sin selección (baseline)
# pipeline_all = Pipeline([
#     ('scaler', StandardScaler()),
#     ('classifier', LogisticRegression(max_iter=1000))
# ])
# scores_all = cross_val_score(pipeline_all, X, y, cv=5)
# results.append({
#     'Method': 'All Features',
#     'CV Accuracy': f"{scores_all.mean():.4f} ± {scores_all.std():.4f}",
#     'n_features': X.shape[1]
# })

# # SelectKBest
# results.append(evaluate_selection(
#     X, y, 
#     SelectKBest(f_classif, k=10),
#     'SelectKBest (k=10)'
# ))

# # RFE
# results.append(evaluate_selection(
#     X, y,
#     RFE(LogisticRegression(max_iter=1000), n_features_to_select=10),
#     'RFE (n=10)'
# ))

# # SelectFromModel
# results.append(evaluate_selection(
#     X, y,
#     SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), threshold='median'),
#     'SelectFromModel (RF)'
# ))

# # Mostrar resultados
# results_df = pd.DataFrame(results)
# print("\nComparación de Métodos de Selección:")
# print(results_df.to_string(index=False))

print()

# ============================================
# RESUMEN
# ============================================
print('=== RESUMEN ===')
print("""
Métodos de Selección de Features:

1. FILTER (Rápido, independiente del modelo):
   - VarianceThreshold: elimina varianza baja
   - SelectKBest: top K por métrica estadística
   - SelectPercentile: top % por métrica

2. WRAPPER (Preciso, costoso):
   - RFE: elimina recursivamente
   - RFECV: RFE + cross-validation para n óptimo
   - SequentialFeatureSelector: forward/backward

3. EMBEDDED (Balance velocidad/precisión):
   - SelectFromModel: basado en importancias
   - Lasso (L1): coeficientes a cero
   - Tree importances: RF, XGBoost

Cuándo usar cada uno:
- Muchas features + poco tiempo → Filter
- Pocas features + máxima precisión → Wrapper
- Balance general → Embedded

RECUERDA:
- Siempre usar cross-validation
- Comparar con baseline (todas las features)
- La selección se hace DENTRO del CV para evitar leakage
""")
