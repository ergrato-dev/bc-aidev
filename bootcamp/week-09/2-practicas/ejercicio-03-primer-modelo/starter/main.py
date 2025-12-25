"""
Ejercicio 03: Tu Primer Modelo de ML
====================================

Objetivo: Entrenar tu primer modelo de clasificación con scikit-learn.

Instrucciones:
1. Lee cada sección
2. Descomenta el código indicado
3. Ejecuta y observa los resultados
"""

# ============================================
# PASO 1: Preparar los Datos
# ============================================
print('--- Paso 1: Preparar los Datos ---')

# Cargar y dividir el dataset
# Descomenta las siguientes líneas:

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# import numpy as np
# 
# # Cargar datos
# iris = load_iris()
# X = iris.data
# y = iris.target
# 
# # Dividir datos
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
# 
# print(f'Datos de entrenamiento: {X_train.shape}')
# print(f'Datos de prueba: {X_test.shape}')
# print(f'Clases a predecir: {iris.target_names}')

print()

# ============================================
# PASO 2: Crear el Modelo
# ============================================
print('--- Paso 2: Crear el Modelo ---')

# KNN clasifica basándose en los K vecinos más cercanos
# Descomenta las siguientes líneas:

# from sklearn.neighbors import KNeighborsClassifier
# 
# # Crear modelo KNN con 3 vecinos
# modelo = KNeighborsClassifier(n_neighbors=3)
# 
# print(f'Modelo creado: {modelo}')
# print(f'Tipo: {type(modelo).__name__}')
# print(f'Parámetros: n_neighbors={modelo.n_neighbors}')

print()

# ============================================
# PASO 3: Entrenar el Modelo (fit)
# ============================================
print('--- Paso 3: Entrenar el Modelo ---')

# fit() ajusta el modelo a los datos de entrenamiento
# Descomenta las siguientes líneas:

# # Entrenar el modelo
# modelo.fit(X_train, y_train)
# 
# print('✅ Modelo entrenado exitosamente')
# print(f'Número de samples de entrenamiento: {len(X_train)}')
# print(f'Número de features: {X_train.shape[1]}')

print()

# ============================================
# PASO 4: Hacer Predicciones (predict)
# ============================================
print('--- Paso 4: Hacer Predicciones ---')

# predict() usa el modelo entrenado para predecir nuevos datos
# Descomenta las siguientes líneas:

# # Predecir en datos de test
# y_pred = modelo.predict(X_test)
# 
# print('Predicciones vs Valores Reales:')
# print('-' * 40)
# 
# for i in range(min(10, len(y_test))):
#     pred_name = iris.target_names[y_pred[i]]
#     real_name = iris.target_names[y_test[i]]
#     status = '✅' if y_pred[i] == y_test[i] else '❌'
#     print(f'{status} Predicción: {pred_name:12} | Real: {real_name}')
# 
# print(f'\n... (mostrando primeros 10 de {len(y_test)})')

print()

# ============================================
# PASO 5: Evaluar el Modelo (score)
# ============================================
print('--- Paso 5: Evaluar el Modelo ---')

# score() calcula la precisión (accuracy) del modelo
# Descomenta las siguientes líneas:

# # Accuracy en datos de entrenamiento
# train_accuracy = modelo.score(X_train, y_train)
# 
# # Accuracy en datos de test
# test_accuracy = modelo.score(X_test, y_test)
# 
# print(f'Accuracy en Train: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)')
# print(f'Accuracy en Test:  {test_accuracy:.4f} ({test_accuracy*100:.1f}%)')
# 
# # Interpretar resultados
# if test_accuracy >= 0.9:
#     print('\n🎉 Excelente! El modelo tiene muy buen rendimiento')
# elif test_accuracy >= 0.7:
#     print('\n👍 Bien! El modelo tiene rendimiento aceptable')
# else:
#     print('\n⚠️  El modelo necesita mejoras')
# 
# # Verificar overfitting
# gap = train_accuracy - test_accuracy
# if gap > 0.1:
#     print(f'⚠️  Posible overfitting (gap={gap:.2f})')
# else:
#     print(f'✅ No hay señales de overfitting (gap={gap:.2f})')

print()

# ============================================
# PASO 6: Predecir Nuevos Datos
# ============================================
print('--- Paso 6: Predecir Nuevos Datos ---')

# Usar el modelo para clasificar flores nuevas
# Descomenta las siguientes líneas:

# # Crear una "flor nueva" (valores inventados)
# nueva_flor = np.array([[5.0, 3.5, 1.5, 0.3]])  # sepal length, sepal width, petal length, petal width
# 
# # Predecir la especie
# prediccion = modelo.predict(nueva_flor)
# nombre_especie = iris.target_names[prediccion[0]]
# 
# print('Nueva flor a clasificar:')
# print(f'  Sepal length: {nueva_flor[0][0]} cm')
# print(f'  Sepal width: {nueva_flor[0][1]} cm')
# print(f'  Petal length: {nueva_flor[0][2]} cm')
# print(f'  Petal width: {nueva_flor[0][3]} cm')
# print(f'\n🌸 Predicción: {nombre_especie}')
# 
# # Predecir varias flores
# nuevas_flores = np.array([
#     [5.1, 3.5, 1.4, 0.2],  # Típica setosa
#     [6.0, 2.7, 4.5, 1.5],  # Típica versicolor
#     [7.2, 3.0, 6.0, 2.0],  # Típica virginica
# ])
# 
# print('\nPredicciones múltiples:')
# predicciones_mult = modelo.predict(nuevas_flores)
# for i, pred in enumerate(predicciones_mult):
#     print(f'  Flor {i+1}: {iris.target_names[pred]}')

print()

# ============================================
# PASO 7: Experimentar con Diferentes K
# ============================================
print('--- Paso 7: Experimentar con K ---')

# El parámetro n_neighbors afecta el rendimiento
# Descomenta las siguientes líneas:

# print('Comparación de diferentes valores de K:')
# print('-' * 40)
# 
# for k in [1, 3, 5, 7, 9, 11]:
#     modelo_k = KNeighborsClassifier(n_neighbors=k)
#     modelo_k.fit(X_train, y_train)
#     acc_train = modelo_k.score(X_train, y_train)
#     acc_test = modelo_k.score(X_test, y_test)
#     print(f'K={k:2d}: Train={acc_train:.3f}, Test={acc_test:.3f}')
# 
# print('-' * 40)
# print('💡 K muy pequeño puede causar overfitting')
# print('💡 K muy grande puede causar underfitting')

print()

# ============================================
# RESUMEN
# ============================================
print('--- Resumen ---')

# Descomenta para ver el resumen:

# print('''
# Flujo de Machine Learning con Scikit-learn:
# ===========================================
# 
# 1. PREPARAR DATOS
#    X_train, X_test, y_train, y_test = train_test_split(...)
# 
# 2. CREAR MODELO
#    modelo = KNeighborsClassifier(n_neighbors=3)
# 
# 3. ENTRENAR (fit)
#    modelo.fit(X_train, y_train)
# 
# 4. PREDECIR (predict)
#    y_pred = modelo.predict(X_test)
# 
# 5. EVALUAR (score)
#    accuracy = modelo.score(X_test, y_test)
# 
# Este patrón es IGUAL para casi todos los modelos de sklearn!
# ''')

print('🎉 ¡Felicidades! Has entrenado tu primer modelo de ML')
