"""
Ejercicio 04: Evaluación Básica de Modelos
==========================================

Objetivo: Aprender a evaluar modelos con diferentes métricas.

Instrucciones:
1. Lee cada sección
2. Descomenta el código indicado
3. Ejecuta y observa los resultados
"""

# ============================================
# PASO 1: Preparar y Entrenar un Modelo
# ============================================
print('--- Paso 1: Preparar y Entrenar ---')

# Preparamos un modelo para poder evaluarlo
# Descomenta las siguientes líneas:

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# import numpy as np
# 
# # Cargar y dividir datos
# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(
#     iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
# )
# 
# # Entrenar modelo
# modelo = KNeighborsClassifier(n_neighbors=3)
# modelo.fit(X_train, y_train)
# 
# # Hacer predicciones
# y_pred = modelo.predict(X_test)
# 
# print(f'Modelo entrenado: {type(modelo).__name__}')
# print(f'Samples de test: {len(y_test)}')
# print(f'Clases: {iris.target_names}')

print()

# ============================================
# PASO 2: Accuracy (Exactitud)
# ============================================
print('--- Paso 2: Accuracy ---')

# Accuracy = (predicciones correctas) / (total de predicciones)
# Descomenta las siguientes líneas:

# from sklearn.metrics import accuracy_score
# 
# # Método 1: usando accuracy_score
# accuracy = accuracy_score(y_test, y_pred)
# 
# # Método 2: usando score() del modelo
# accuracy_modelo = modelo.score(X_test, y_test)
# 
# # Método 3: cálculo manual
# correctos = (y_test == y_pred).sum()
# accuracy_manual = correctos / len(y_test)
# 
# print(f'Accuracy (accuracy_score): {accuracy:.4f}')
# print(f'Accuracy (modelo.score):   {accuracy_modelo:.4f}')
# print(f'Accuracy (manual):         {accuracy_manual:.4f}')
# print(f'\n✅ Accuracy = {correctos}/{len(y_test)} predicciones correctas')

print()

# ============================================
# PASO 3: Matriz de Confusión
# ============================================
print('--- Paso 3: Matriz de Confusión ---')

# La matriz muestra cómo se distribuyen las predicciones
# Descomenta las siguientes líneas:

# from sklearn.metrics import confusion_matrix
# 
# cm = confusion_matrix(y_test, y_pred)
# 
# print('Matriz de Confusión:')
# print(cm)
# 
# print('\nInterpretación (filas=real, columnas=predicción):')
# print('                Pred:')
# print('                setosa  versicolor  virginica')
# print(f'Real: setosa      {cm[0,0]:3d}       {cm[0,1]:3d}        {cm[0,2]:3d}')
# print(f'      versicolor  {cm[1,0]:3d}       {cm[1,1]:3d}        {cm[1,2]:3d}')
# print(f'      virginica   {cm[2,0]:3d}       {cm[2,1]:3d}        {cm[2,2]:3d}')
# 
# # Diagonal = predicciones correctas
# print(f'\n✅ Predicciones correctas (diagonal): {cm.diagonal().sum()}')
# print(f'❌ Predicciones incorrectas: {cm.sum() - cm.diagonal().sum()}')

print()

# ============================================
# PASO 4: Precision, Recall y F1-Score
# ============================================
print('--- Paso 4: Precision, Recall, F1 ---')

# Métricas más detalladas para clasificación
# Descomenta las siguientes líneas:

# from sklearn.metrics import precision_score, recall_score, f1_score
# 
# # Para multiclase, necesitamos especificar average
# precision = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')
# f1 = f1_score(y_test, y_pred, average='weighted')
# 
# print('Métricas (weighted average):')
# print(f'  Precision: {precision:.4f}')
# print(f'  Recall:    {recall:.4f}')
# print(f'  F1-Score:  {f1:.4f}')
# 
# print('\nSignificado:')
# print('  Precision: De los que predije como clase X, ¿qué % eran realmente X?')
# print('  Recall:    De los que realmente son clase X, ¿qué % encontré?')
# print('  F1-Score:  Balance entre precision y recall')

print()

# ============================================
# PASO 5: Classification Report
# ============================================
print('--- Paso 5: Classification Report ---')

# Reporte completo con todas las métricas por clase
# Descomenta las siguientes líneas:

# from sklearn.metrics import classification_report
# 
# print('Classification Report:')
# print('=' * 60)
# print(classification_report(y_test, y_pred, target_names=iris.target_names))
# print('=' * 60)
# 
# print('Interpretación:')
# print('  - Support: número de samples reales de cada clase')
# print('  - Macro avg: promedio simple de las métricas')
# print('  - Weighted avg: promedio ponderado por support')

print()

# ============================================
# PASO 6: Visualizar Matriz de Confusión
# ============================================
print('--- Paso 6: Visualizar Matriz ---')

# Crear visualización de la matriz de confusión
# Descomenta las siguientes líneas:

# import matplotlib.pyplot as plt
# from sklearn.metrics import ConfusionMatrixDisplay
# 
# fig, ax = plt.subplots(figsize=(8, 6))
# 
# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=iris.target_names
# )
# disp.plot(ax=ax, cmap='Blues', values_format='d')
# 
# plt.title('Matriz de Confusión - KNN Classifier')
# plt.tight_layout()
# plt.savefig('matriz_confusion.png', dpi=150)
# print('Gráfico guardado como: matriz_confusion.png')
# plt.show()

print()

# ============================================
# PASO 7: Comparar Métricas por Clase
# ============================================
print('--- Paso 7: Métricas por Clase ---')

# Ver métricas detalladas para cada clase
# Descomenta las siguientes líneas:

# print('Métricas detalladas por clase:')
# print('-' * 50)
# 
# precision_per_class = precision_score(y_test, y_pred, average=None)
# recall_per_class = recall_score(y_test, y_pred, average=None)
# f1_per_class = f1_score(y_test, y_pred, average=None)
# 
# for i, name in enumerate(iris.target_names):
#     print(f'{name:12s}: Precision={precision_per_class[i]:.3f}, '
#           f'Recall={recall_per_class[i]:.3f}, F1={f1_per_class[i]:.3f}')
# 
# print('-' * 50)
# 
# # Identificar clase con peor rendimiento
# worst_class_idx = f1_per_class.argmin()
# print(f'\n⚠️  Clase con peor F1: {iris.target_names[worst_class_idx]}')
# print('    (Esta clase necesita más atención)')

print()

# ============================================
# PASO 8: Cuándo usar cada métrica
# ============================================
print('--- Paso 8: Guía de Métricas ---')

# Descomenta para ver la guía:

# print('''
# ¿Cuándo usar cada métrica?
# ==========================
# 
# ACCURACY
#   ✅ Usar cuando: Clases balanceadas
#   ❌ Evitar cuando: Clases muy desbalanceadas
#   Ejemplo: Clasificación general con clases similares
# 
# PRECISION
#   ✅ Usar cuando: El costo de falsos positivos es alto
#   Ejemplo: Filtro de spam (no quieres mover emails buenos a spam)
#   Pregunta: "De los que marqué como spam, ¿cuántos eran realmente spam?"
# 
# RECALL
#   ✅ Usar cuando: El costo de falsos negativos es alto
#   Ejemplo: Detección de cáncer (no quieres miss ningún caso)
#   Pregunta: "De todos los casos de cáncer, ¿cuántos detecté?"
# 
# F1-SCORE
#   ✅ Usar cuando: Necesitas balance entre precision y recall
#   ✅ Usar cuando: Clases desbalanceadas
#   Ejemplo: Sistemas de recomendación, detección de fraude
# ''')

print()

# ============================================
# RESUMEN
# ============================================
print('--- Resumen ---')

# Descomenta para ver el resumen:

# print(f'''
# Evaluación del Modelo - Resumen:
# ================================
# 
# Modelo: KNeighborsClassifier (K=3)
# Dataset: Iris (3 clases)
# 
# Métricas Globales:
#   - Accuracy:  {accuracy:.2%}
#   - Precision: {precision:.2%}
#   - Recall:    {recall:.2%}
#   - F1-Score:  {f1:.2%}
# 
# Matriz de Confusión:
# {cm}
# 
# ✅ El modelo muestra buen rendimiento en todas las clases
# ''')

print('Ejercicio completado!')
