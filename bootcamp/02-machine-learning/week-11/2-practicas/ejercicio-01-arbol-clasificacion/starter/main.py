"""
Ejercicio 01: Árbol de Decisión para Clasificación
===================================================

Aprenderás a:
- Crear un DecisionTreeClassifier
- Entrenar y evaluar el modelo
- Visualizar el árbol de decisión
- Predecir con probabilidades

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
# from sklearn.datasets import load_iris
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# import matplotlib.pyplot as plt

print('Librerías importadas correctamente')
print()

# ============================================
# PASO 2: Cargar el Dataset Iris
# ============================================
print('--- Paso 2: Cargar Dataset ---')

# Descomenta las siguientes líneas:
# iris = load_iris()
# X, y = iris.data, iris.target
#
# print(f"Features: {iris.feature_names}")
# print(f"Clases: {iris.target_names}")
# print(f"Shape X: {X.shape}, Shape y: {y.shape}")

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
# PASO 4: Crear y Entrenar el Árbol
# ============================================
print('--- Paso 4: Entrenar Árbol ---')

# Descomenta las siguientes líneas:
# tree = DecisionTreeClassifier(
#     max_depth=3,
#     criterion='gini',
#     random_state=42
# )
#
# tree.fit(X_train, y_train)
# print("Árbol entrenado correctamente")

print()

# ============================================
# PASO 5: Evaluar el Modelo
# ============================================
print('--- Paso 5: Evaluar Modelo ---')

# Descomenta las siguientes líneas:
# y_pred = tree.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# print(f"\nAccuracy: {accuracy:.4f}")
#
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=iris.target_names))

print()

# ============================================
# PASO 6: Visualizar el Árbol
# ============================================
print('--- Paso 6: Visualizar Árbol ---')

# Descomenta las siguientes líneas:
# plt.figure(figsize=(20, 10))
# plot_tree(
#     tree,
#     feature_names=iris.feature_names,
#     class_names=iris.target_names,
#     filled=True,
#     rounded=True,
#     fontsize=10
# )
# plt.title('Árbol de Decisión - Iris Dataset')
# plt.tight_layout()
# plt.savefig('arbol_iris.png', dpi=150)
# plt.show()
# print("Árbol guardado como 'arbol_iris.png'")

print()

# ============================================
# PASO 7: Predecir con Probabilidades
# ============================================
print('--- Paso 7: Predicciones con Probabilidades ---')

# Descomenta las siguientes líneas:
# sample = X_test[:3]
# predictions = tree.predict(sample)
# probabilities = tree.predict_proba(sample)
#
# for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
#     print(f"\nMuestra {i+1}:")
#     print(f"  Predicción: {iris.target_names[pred]}")
#     print(f"  Probabilidades: {dict(zip(iris.target_names, proba.round(3)))}")

print()

# ============================================
# PASO 8: Experimentar con max_depth
# ============================================
print('--- Paso 8: Impacto de max_depth ---')

# Descomenta las siguientes líneas:
# for depth in [1, 2, 3, 5, 10, None]:
#     tree_exp = DecisionTreeClassifier(max_depth=depth, random_state=42)
#     tree_exp.fit(X_train, y_train)
#
#     train_acc = tree_exp.score(X_train, y_train)
#     test_acc = tree_exp.score(X_test, y_test)
#
#     depth_str = str(depth) if depth else "None"
#     print(f"max_depth={depth_str:4s}: Train={train_acc:.4f}, Test={test_acc:.4f}")

print()
print('=== Ejercicio completado ===')
