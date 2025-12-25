"""
Ejercicio 02: Train/Test Split
==============================

Objetivo: Aprender a dividir datos correctamente para ML.

Instrucciones:
1. Lee cada sección
2. Descomenta el código indicado
3. Ejecuta y observa los resultados
"""

# ============================================
# PASO 1: Cargar el Dataset
# ============================================
print('--- Paso 1: Cargar el Dataset ---')

# Cargamos el dataset Iris
# Descomenta las siguientes líneas:

# from sklearn.datasets import load_iris
# import pandas as pd
# import numpy as np
# 
# iris = load_iris()
# X = iris.data  # Features
# y = iris.target  # Target
# 
# print(f'Shape de X (features): {X.shape}')
# print(f'Shape de y (target): {y.shape}')
# print(f'Clases únicas: {np.unique(y)}')

print()

# ============================================
# PASO 2: División Básica (Sin Estratificar)
# ============================================
print('--- Paso 2: División Básica ---')

# train_test_split divide los datos aleatoriamente
# Descomenta las siguientes líneas:

# from sklearn.model_selection import train_test_split
# 
# # División 80% train, 20% test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,      # 20% para test
#     random_state=42     # Semilla para reproducibilidad
# )
# 
# print(f'Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)')
# print(f'Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)')

print()

# ============================================
# PASO 3: Verificar Distribución de Clases
# ============================================
print('--- Paso 3: Verificar Distribución ---')

# Verificamos si la distribución de clases se mantiene
# Descomenta las siguientes líneas:

# def mostrar_distribucion(y_data, nombre):
#     unique, counts = np.unique(y_data, return_counts=True)
#     print(f'{nombre}:')
#     for clase, count in zip(unique, counts):
#         print(f'  Clase {clase}: {count} ({count/len(y_data)*100:.1f}%)')
# 
# print('Distribución original:')
# mostrar_distribucion(y, 'Dataset completo')
# 
# print('\nDistribución después del split (SIN stratify):')
# mostrar_distribucion(y_train, 'Train')
# mostrar_distribucion(y_test, 'Test')

print()

# ============================================
# PASO 4: División Estratificada
# ============================================
print('--- Paso 4: División Estratificada ---')

# stratify=y mantiene las proporciones de clases
# Descomenta las siguientes líneas:

# X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y  # Mantener proporciones de clases
# )
# 
# print('Distribución después del split (CON stratify):')
# mostrar_distribucion(y_train_strat, 'Train')
# mostrar_distribucion(y_test_strat, 'Test')
# 
# print('\n✅ Con stratify, las proporciones se mantienen exactas')

print()

# ============================================
# PASO 5: Importancia de random_state
# ============================================
print('--- Paso 5: Reproducibilidad ---')

# random_state garantiza que obtengamos la misma división
# Descomenta las siguientes líneas:

# # Sin random_state: cada ejecución da resultados diferentes
# X_train1, X_test1, _, _ = train_test_split(X, y, test_size=0.2)
# X_train2, X_test2, _, _ = train_test_split(X, y, test_size=0.2)
# 
# print('Sin random_state:')
# print(f'  Primera división - primer sample de test: {X_test1[0][:2]}...')
# print(f'  Segunda división - primer sample de test: {X_test2[0][:2]}...')
# print('  (Probablemente diferentes)')
# 
# # Con random_state: siempre igual
# X_train3, X_test3, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train4, X_test4, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# print('\nCon random_state=42:')
# print(f'  Primera división - primer sample de test: {X_test3[0][:2]}...')
# print(f'  Segunda división - primer sample de test: {X_test4[0][:2]}...')
# print('  ✅ Siempre iguales')

print()

# ============================================
# PASO 6: División Train/Validation/Test
# ============================================
print('--- Paso 6: División Train/Val/Test ---')

# Para proyectos más robustos, usamos 3 conjuntos
# Descomenta las siguientes líneas:

# # Paso 1: Separar test (20%)
# X_temp, X_test_final, y_temp, y_test_final = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
# 
# # Paso 2: Separar train y validation del resto (75/25 = 60/20 del total)
# X_train_final, X_val, y_train_final, y_val = train_test_split(
#     X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
# )
# 
# print('División Train/Validation/Test:')
# print(f'  Train: {len(X_train_final)} samples ({len(X_train_final)/len(X)*100:.0f}%)')
# print(f'  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.0f}%)')
# print(f'  Test: {len(X_test_final)} samples ({len(X_test_final)/len(X)*100:.0f}%)')
# 
# print('\nUso de cada conjunto:')
# print('  - Train: Entrenar el modelo')
# print('  - Validation: Ajustar hiperparámetros')
# print('  - Test: Evaluación final (solo una vez)')

print()

# ============================================
# PASO 7: Diferentes Tamaños de Test
# ============================================
print('--- Paso 7: Experimentar con Tamaños ---')

# Probar diferentes proporciones de split
# Descomenta las siguientes líneas:

# test_sizes = [0.1, 0.2, 0.3, 0.4]
# 
# print('Comparación de diferentes tamaños de test:')
# print('-' * 50)
# 
# for test_size in test_sizes:
#     X_tr, X_te, y_tr, y_te = train_test_split(
#         X, y, test_size=test_size, random_state=42, stratify=y
#     )
#     print(f'test_size={test_size}: Train={len(X_tr):3d}, Test={len(X_te):3d}')
# 
# print('-' * 50)
# print('\n💡 Recomendación: 0.2 (20%) es un buen balance para datasets medianos')

print()

# ============================================
# RESUMEN
# ============================================
print('--- Resumen ---')

# Descomenta para ver el resumen:

# print('''
# Train/Test Split - Puntos Clave:
# ================================
# 
# 1. test_size: Proporción de datos para test (típico: 0.2)
# 
# 2. random_state: Semilla para reproducibilidad (usar siempre)
# 
# 3. stratify: Mantener proporciones de clases (importante en clasificación)
# 
# 4. División típica:
#    - Simple: 80% train / 20% test
#    - Robusta: 60% train / 20% val / 20% test
# 
# ⚠️  NUNCA usar datos de test para entrenar o ajustar el modelo
# ''')

print('Ejercicio completado!')
