"""
Ejercicio 03: Random Forest Classifier
======================================

Aprenderás a:
- Implementar RandomForestClassifier
- Comparar con árbol individual
- Usar OOB Score para validación
- Analizar impacto de n_estimators

Instrucciones:
- Lee el README.md para entender cada paso
- Descomenta cada sección progresivamente
- Ejecuta y observa los resultados
"""

# ============================================
# PASO 1: Importar Librerías
# ============================================
print('--- Paso 1: Importar Librerías ---')

# Descomenta las siguientes líneas:
# from sklearn.datasets import load_breast_cancer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score, classification_report
# import numpy as np
# import matplotlib.pyplot as plt

print('Librerías importadas correctamente')
print()

# ============================================
# PASO 2: Cargar Breast Cancer Dataset
# ============================================
print('--- Paso 2: Cargar Dataset ---')

# Descomenta las siguientes líneas:
# cancer = load_breast_cancer()
# X, y = cancer.data, cancer.target
#
# print(f"Features: {len(cancer.feature_names)}")
# print(f"Clases: {cancer.target_names}")
# print(f"Shape X: {X.shape}")
# print(f"Distribución: Benign={sum(y==1)}, Malignant={sum(y==0)}")

print()

# ============================================
# PASO 3: Dividir los Datos
# ============================================
print('--- Paso 3: Dividir Datos ---')

# Descomenta las siguientes líneas:
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )
#
# print(f"Train: {X_train.shape[0]} muestras")
# print(f"Test: {X_test.shape[0]} muestras")

print()

# ============================================
# PASO 4: Entrenar Árbol Individual (baseline)
# ============================================
print('--- Paso 4: Árbol Individual (baseline) ---')

# Descomenta las siguientes líneas:
# tree = DecisionTreeClassifier(max_depth=5, random_state=42)
# tree.fit(X_train, y_train)
#
# tree_train_acc = tree.score(X_train, y_train)
# tree_test_acc = tree.score(X_test, y_test)
#
# print(f"\n--- Decision Tree (baseline) ---")
# print(f"Train Accuracy: {tree_train_acc:.4f}")
# print(f"Test Accuracy: {tree_test_acc:.4f}")

print()

# ============================================
# PASO 5: Entrenar Random Forest
# ============================================
print('--- Paso 5: Entrenar Random Forest ---')

# Descomenta las siguientes líneas:
# rf = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=5,
#     max_features='sqrt',
#     oob_score=True,
#     random_state=42,
#     n_jobs=-1
# )
#
# rf.fit(X_train, y_train)
# print("Random Forest entrenado con 100 árboles")

print()

# ============================================
# PASO 6: Evaluar y Comparar
# ============================================
print('--- Paso 6: Evaluar y Comparar ---')

# Descomenta las siguientes líneas:
# rf_train_acc = rf.score(X_train, y_train)
# rf_test_acc = rf.score(X_test, y_test)
#
# print(f"\n--- Random Forest ---")
# print(f"Train Accuracy: {rf_train_acc:.4f}")
# print(f"Test Accuracy: {rf_test_acc:.4f}")
# print(f"OOB Score: {rf.oob_score_:.4f}")
#
# print(f"\n--- Comparación ---")
# print(f"Decision Tree Test: {tree_test_acc:.4f}")
# print(f"Random Forest Test: {rf_test_acc:.4f}")
# print(f"Mejora: {(rf_test_acc - tree_test_acc)*100:.2f}%")

print()

# ============================================
# PASO 7: Impacto de n_estimators
# ============================================
print('--- Paso 7: Impacto de n_estimators ---')

# Descomenta las siguientes líneas:
# n_trees_list = [1, 5, 10, 25, 50, 100, 200]
# oob_scores = []
# test_scores = []
#
# for n_trees in n_trees_list:
#     rf_exp = RandomForestClassifier(
#         n_estimators=n_trees,
#         max_depth=5,
#         oob_score=True,
#         random_state=42,
#         n_jobs=-1
#     )
#     rf_exp.fit(X_train, y_train)
#     oob_scores.append(rf_exp.oob_score_)
#     test_scores.append(rf_exp.score(X_test, y_test))
#
# print("\n--- Impacto de n_estimators ---")
# for n, oob, test in zip(n_trees_list, oob_scores, test_scores):
#     print(f"n_estimators={n:3d}: OOB={oob:.4f}, Test={test:.4f}")

print()

# ============================================
# PASO 8: Visualizar n_estimators
# ============================================
print('--- Paso 8: Visualizar n_estimators ---')

# Descomenta las siguientes líneas:
# plt.figure(figsize=(10, 6))
# plt.plot(n_trees_list, oob_scores, 'g-o', label='OOB Score', linewidth=2)
# plt.plot(n_trees_list, test_scores, 'b-o', label='Test Score', linewidth=2)
# plt.xlabel('Número de Árboles (n_estimators)')
# plt.ylabel('Accuracy')
# plt.title('Impacto de n_estimators en Random Forest')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.savefig('n_estimators_impact.png', dpi=150)
# plt.show()

print()

# ============================================
# PASO 9: Feature Importance
# ============================================
print('--- Paso 9: Feature Importance ---')

# Descomenta las siguientes líneas:
# importance = rf.feature_importances_
# indices = np.argsort(importance)[::-1][:10]
#
# print("\n--- Top 10 Features ---")
# for i, idx in enumerate(indices):
#     print(f"{i+1}. {cancer.feature_names[idx]}: {importance[idx]:.4f}")
#
# plt.figure(figsize=(12, 6))
# plt.bar(range(10), importance[indices], color='steelblue')
# plt.xticks(range(10), [cancer.feature_names[i] for i in indices], rotation=45, ha='right')
# plt.xlabel('Feature')
# plt.ylabel('Importancia')
# plt.title('Top 10 Feature Importance - Random Forest')
# plt.tight_layout()
# plt.savefig('rf_feature_importance.png', dpi=150)
# plt.show()

print()

# ============================================
# PASO 10: Cross-Validation
# ============================================
print('--- Paso 10: Cross-Validation ---')

# Descomenta las siguientes líneas:
# tree_cv = cross_val_score(
#     DecisionTreeClassifier(max_depth=5, random_state=42),
#     X, y, cv=5
# )
#
# rf_cv = cross_val_score(
#     RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
#     X, y, cv=5
# )
#
# print("\n--- Cross-Validation (5-fold) ---")
# print(f"Decision Tree: {tree_cv.mean():.4f} ± {tree_cv.std():.4f}")
# print(f"Random Forest: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

print()
print('=== Ejercicio completado ===')
