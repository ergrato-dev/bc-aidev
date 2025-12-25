"""
Ejercicio 03: Regresión Logística
=================================
Clasificación binaria: predecir si un estudiante aprueba o no.

Instrucciones:
- Lee cada paso en README.md
- Descomenta el código correspondiente
- Ejecuta y observa los resultados
"""

# ============================================
# PASO 1: Crear Dataset de Clasificación
# ============================================
print('--- Paso 1: Crear Dataset ---')

# Estudiantes: horas de estudio → aprobado (1) o reprobado (0)
# Descomenta las siguientes líneas:
# import numpy as np
# import pandas as pd
#
# np.random.seed(42)
# n = 200
#
# horas_estudio = np.random.uniform(0, 10, n)
# # Probabilidad de aprobar sigue una sigmoide centrada en 5 horas
# prob_aprobar = 1 / (1 + np.exp(-(horas_estudio - 5)))
# aprobado = (np.random.random(n) < prob_aprobar).astype(int)
#
# print(f'Total estudiantes: {n}')
# print(f'Aprobados: {aprobado.sum()} ({aprobado.mean()*100:.1f}%)')
# print(f'Reprobados: {n - aprobado.sum()} ({(1-aprobado.mean())*100:.1f}%)')

print()

# ============================================
# PASO 2: Visualizar Distribución
# ============================================
print('--- Paso 2: Visualizar Distribución ---')

# Scatter plot de los datos
# Descomenta las siguientes líneas:
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 5))
# plt.scatter(horas_estudio[aprobado==0], aprobado[aprobado==0],
#             label='Reprobado', alpha=0.6, color='#EF4444')
# plt.scatter(horas_estudio[aprobado==1], aprobado[aprobado==1],
#             label='Aprobado', alpha=0.6, color='#10B981')
# plt.xlabel('Horas de estudio')
# plt.ylabel('Resultado (0=Reprobado, 1=Aprobado)')
# plt.legend()
# plt.title('Distribución de Datos')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('distribucion_datos.png', dpi=150)
# print('Gráfico guardado: distribucion_datos.png')
# plt.show()

print()

# ============================================
# PASO 3: Entrenar Regresión Logística
# ============================================
print('--- Paso 3: Entrenar Modelo ---')

# LogisticRegression para clasificación binaria
# Descomenta las siguientes líneas:
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
#
# X = horas_estudio.reshape(-1, 1)
# y = aprobado
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# model = LogisticRegression()
# model.fit(X_train, y_train)
#
# print(f'Coeficiente: {model.coef_[0][0]:.4f}')
# print(f'Intercepto: {model.intercept_[0]:.4f}')
# print()
# print('Ecuación: log(p/(1-p)) = {:.2f} + {:.2f} * horas'.format(
#     model.intercept_[0], model.coef_[0][0]))

print()

# ============================================
# PASO 4: Entender Probabilidades
# ============================================
print('--- Paso 4: Probabilidades ---')

# predict_proba devuelve [P(clase_0), P(clase_1)]
# Descomenta las siguientes líneas:
# probs = model.predict_proba(X_test)
#
# print('Primeras 5 predicciones:')
# print('Horas | P(Reprob) | P(Aprob) | Real')
# print('-' * 40)
# for i in range(5):
#     print(f'{X_test[i][0]:5.2f} | {probs[i][0]:9.4f} | {probs[i][1]:8.4f} | {y_test.iloc[i]}')
#
# # Predicción de clases (threshold=0.5)
# preds = model.predict(X_test)
# print(f'\nPredicciones con threshold=0.5: {preds[:10]}')

print()

# ============================================
# PASO 5: Evaluar con Métricas
# ============================================
print('--- Paso 5: Métricas de Evaluación ---')

# Accuracy, Precision, Recall, F1
# Descomenta las siguientes líneas:
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
# accuracy = accuracy_score(y_test, preds)
# precision = precision_score(y_test, preds)
# recall = recall_score(y_test, preds)
# f1 = f1_score(y_test, preds)
#
# print(f'Accuracy:  {accuracy:.4f}  (% aciertos totales)')
# print(f'Precision: {precision:.4f}  (de los que predije 1, cuántos son 1)')
# print(f'Recall:    {recall:.4f}  (de los que son 1, cuántos predije)')
# print(f'F1-Score:  {f1:.4f}  (media armónica precision-recall)')

print()

# ============================================
# PASO 6: Matriz de Confusión
# ============================================
print('--- Paso 6: Matriz de Confusión ---')

# Visualizar TP, TN, FP, FN
# Descomenta las siguientes líneas:
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#
# cm = confusion_matrix(y_test, preds)
# print('Matriz de Confusión:')
# print(f'  TN={cm[0,0]} | FP={cm[0,1]}')
# print(f'  FN={cm[1,0]} | TP={cm[1,1]}')
#
# disp = ConfusionMatrixDisplay(cm, display_labels=['Reprobado', 'Aprobado'])
# disp.plot(cmap='Blues')
# plt.title('Matriz de Confusión')
# plt.tight_layout()
# plt.savefig('confusion_matrix.png', dpi=150)
# print('\nGráfico guardado: confusion_matrix.png')
# plt.show()

print()

# ============================================
# PASO 7: Visualizar Curva Sigmoide
# ============================================
print('--- Paso 7: Curva Sigmoide ---')

# La probabilidad sigue forma de S (sigmoide)
# Descomenta las siguientes líneas:
# X_range = np.linspace(0, 10, 100).reshape(-1, 1)
# probs_range = model.predict_proba(X_range)[:, 1]
#
# plt.figure(figsize=(10, 6))
# plt.scatter(X_test, y_test, alpha=0.5, label='Datos reales', color='#3B82F6')
# plt.plot(X_range, probs_range, color='#EF4444', linewidth=2,
#          label='P(aprobar)')
# plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7,
#             label='Threshold=0.5')
# plt.xlabel('Horas de estudio')
# plt.ylabel('Probabilidad de aprobar')
# plt.legend()
# plt.title('Regresión Logística: Curva Sigmoide')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('curva_sigmoide.png', dpi=150)
# print('Gráfico guardado: curva_sigmoide.png')
# plt.show()

print()

# ============================================
# PASO 8: Cambiar Threshold
# ============================================
print('--- Paso 8: Efecto del Threshold ---')

# Threshold más alto = más conservador (menos falsos positivos)
# Descomenta las siguientes líneas:
# thresholds = [0.3, 0.5, 0.7]
#
# print('Threshold | Accuracy | Precision | Recall')
# print('-' * 45)
# for t in thresholds:
#     preds_t = (model.predict_proba(X_test)[:, 1] >= t).astype(int)
#     acc = accuracy_score(y_test, preds_t)
#     prec = precision_score(y_test, preds_t, zero_division=0)
#     rec = recall_score(y_test, preds_t, zero_division=0)
#     print(f'   {t:.1f}    |  {acc:.4f}  |  {prec:.4f}   | {rec:.4f}')
#
# print()
# print('Observación:')
# print('  - Threshold bajo (0.3): más recall, menos precision')
# print('  - Threshold alto (0.7): más precision, menos recall')

print()
print('=== Ejercicio completado ===')
