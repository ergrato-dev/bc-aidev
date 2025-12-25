"""
Ejercicio 02: Support Vector Machines
=====================================
Implementa SVM con diferentes kernels para Breast Cancer.
"""

# ============================================
# PASO 1: Cargar Dataset
# ============================================
print('--- Paso 1: Cargar Dataset ---')

# Descomenta las siguientes líneas:
# from sklearn.datasets import load_breast_cancer
# import pandas as pd
#
# cancer = load_breast_cancer()
# print(f"Dataset: Breast Cancer")
# print(f"Samples: {cancer.data.shape[0]}")
# print(f"Features: {cancer.data.shape[1]}")
# print(f"Clases: {cancer.target_names}")
# print(f"\nPrimeras 5 features: {list(cancer.feature_names[:5])}")

print()

# ============================================
# PASO 2: Preparar Datos
# ============================================
print('--- Paso 2: Preparar Datos ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target,
#     test_size=0.2,
#     random_state=42,
#     stratify=cancer.target
# )
#
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# print(f"Train shape: {X_train_scaled.shape}")
# print(f"Test shape: {X_test_scaled.shape}")

print()

# ============================================
# PASO 3: SVM con Kernel Lineal
# ============================================
print('--- Paso 3: SVM con Kernel Lineal ---')

# Descomenta las siguientes líneas:
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
#
# svm_linear = SVC(kernel='linear', C=1.0)
# svm_linear.fit(X_train_scaled, y_train)
# y_pred = svm_linear.predict(X_test_scaled)
#
# print(f"Linear SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print(f"Vectores de soporte por clase: {svm_linear.n_support_}")
# print(f"Total vectores de soporte: {sum(svm_linear.n_support_)}")

print()

# ============================================
# PASO 4: Comparar Kernels
# ============================================
print('--- Paso 4: Comparar Kernels ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import cross_val_score
#
# kernels = ['linear', 'rbf', 'poly', 'sigmoid']
#
# print("Comparación de kernels (CV=5):")
# kernel_scores = {}
# for kernel in kernels:
#     svm = SVC(kernel=kernel)
#     scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
#     kernel_scores[kernel] = scores.mean()
#     print(f"  {kernel:8s}: {scores.mean():.4f} ± {scores.std():.4f}")
#
# best_kernel = max(kernel_scores, key=kernel_scores.get)
# print(f"\nMejor kernel: {best_kernel}")

print()

# ============================================
# PASO 5: GridSearch para RBF
# ============================================
print('--- Paso 5: GridSearch para RBF ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import GridSearchCV
#
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': ['scale', 'auto', 0.01, 0.1]
# }
#
# print("Ejecutando GridSearch...")
# grid = GridSearchCV(
#     SVC(kernel='rbf'),
#     param_grid,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1
# )
# grid.fit(X_train_scaled, y_train)
#
# print(f"Mejores parámetros: {grid.best_params_}")
# print(f"Mejor CV score: {grid.best_score_:.4f}")

print()

# ============================================
# PASO 6: Evaluación Final
# ============================================
print('--- Paso 6: Evaluación Final ---')

# Descomenta las siguientes líneas:
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# y_pred = grid.predict(X_test_scaled)
# test_acc = grid.score(X_test_scaled, y_test)
#
# print(f"Test Accuracy: {test_acc:.4f}")
# print(f"\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=cancer.target_names))
#
# # Matriz de confusión
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=cancer.target_names,
#             yticklabels=cancer.target_names)
# plt.xlabel('Predicho')
# plt.ylabel('Real')
# plt.title(f'Matriz de Confusión - SVM RBF (Acc: {test_acc:.4f})')
# plt.tight_layout()
# plt.savefig('svm_confusion_matrix.png', dpi=100, bbox_inches='tight')
# plt.show()
#
# print("\nMatriz guardada en: svm_confusion_matrix.png")

print()
print('=' * 50)
print('Ejercicio completado!')
