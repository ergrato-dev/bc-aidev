"""
Ejercicio 02: Métricas de Clasificación
=======================================
Aprende a calcular e interpretar métricas de clasificación.
"""

# ============================================
# PASO 1: Preparar Datos y Modelo
# ============================================
print('--- Paso 1: Preparar Datos y Modelo ---')

# Descomenta las siguientes líneas:
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# 
# # Crear dataset
# X, y = make_classification(
#     n_samples=1000, n_features=20, n_informative=15,
#     n_redundant=5, random_state=42
# )
# 
# # Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
# 
# # Entrenar modelo
# model = LogisticRegression(max_iter=1000, random_state=42)
# model.fit(X_train, y_train)
# 
# # Predicciones
# y_pred = model.predict(X_test)
# y_proba = model.predict_proba(X_test)[:, 1]
# 
# print(f"Muestras de test: {len(y_test)}")
# print(f"Predicciones realizadas: {len(y_pred)}")

print()

# ============================================
# PASO 2: Matriz de Confusión
# ============================================
print('--- Paso 2: Matriz de Confusión ---')

# Descomenta las siguientes líneas:
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 
# # Calcular matriz
# cm = confusion_matrix(y_test, y_pred)
# print("Matriz de Confusión:")
# print(cm)
# 
# # Extraer componentes
# tn, fp, fn, tp = cm.ravel()
# print(f"\nComponentes:")
# print(f"  TN (True Negatives):  {tn}")
# print(f"  FP (False Positives): {fp} ← Error Tipo I")
# print(f"  FN (False Negatives): {fn} ← Error Tipo II")
# print(f"  TP (True Positives):  {tp}")
# 
# # Visualizar
# fig, ax = plt.subplots(figsize=(6, 5))
# ConfusionMatrixDisplay(cm, display_labels=['Negativo', 'Positivo']).plot(ax=ax)
# plt.title('Matriz de Confusión')
# plt.tight_layout()
# plt.savefig('confusion_matrix.png', dpi=100)
# plt.close()
# print("\n✓ Gráfica guardada: confusion_matrix.png")

print()

# ============================================
# PASO 3: Métricas Básicas
# ============================================
print('--- Paso 3: Métricas Básicas ---')

# Descomenta las siguientes líneas:
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 
# # Calcular métricas
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# 
# print("Métricas de Clasificación:")
# print("-" * 40)
# print(f"Accuracy:  {accuracy:.4f}  (TP+TN)/Total")
# print(f"Precision: {precision:.4f}  TP/(TP+FP)")
# print(f"Recall:    {recall:.4f}  TP/(TP+FN)")
# print(f"F1-Score:  {f1:.4f}  2*P*R/(P+R)")
# 
# # Verificar cálculos manuales
# print("\nVerificación manual:")
# acc_manual = (tp + tn) / (tp + tn + fp + fn)
# prec_manual = tp / (tp + fp)
# rec_manual = tp / (tp + fn)
# f1_manual = 2 * prec_manual * rec_manual / (prec_manual + rec_manual)
# print(f"Accuracy manual:  {acc_manual:.4f}")
# print(f"Precision manual: {prec_manual:.4f}")
# print(f"Recall manual:    {rec_manual:.4f}")
# print(f"F1 manual:        {f1_manual:.4f}")

print()

# ============================================
# PASO 4: Classification Report
# ============================================
print('--- Paso 4: Classification Report ---')

# Descomenta las siguientes líneas:
# from sklearn.metrics import classification_report
# 
# print("Classification Report Completo:")
# print("=" * 55)
# print(classification_report(y_test, y_pred, target_names=['Clase 0', 'Clase 1']))
# 
# print("Interpretación:")
# print("- support: número de muestras de cada clase")
# print("- macro avg: promedio simple de cada métrica")
# print("- weighted avg: promedio ponderado por support")

print()

# ============================================
# PASO 5: Curva ROC y AUC
# ============================================
print('--- Paso 5: Curva ROC y AUC ---')

# Descomenta las siguientes líneas:
# from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
# 
# # Calcular ROC
# fpr, tpr, thresholds = roc_curve(y_test, y_proba)
# auc = roc_auc_score(y_test, y_proba)
# 
# print(f"AUC-ROC: {auc:.4f}")
# print(f"Número de umbrales: {len(thresholds)}")
# 
# # Graficar
# fig, ax = plt.subplots(figsize=(8, 6))
# RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name='LogisticRegression').plot(ax=ax)
# ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
# ax.set_title('Curva ROC')
# ax.legend()
# plt.tight_layout()
# plt.savefig('roc_curve.png', dpi=100)
# plt.close()
# print("✓ Gráfica guardada: roc_curve.png")
# 
# # Interpretación
# print("\nInterpretación de AUC:")
# if auc >= 0.9:
#     print("  → Excelente capacidad discriminativa")
# elif auc >= 0.8:
#     print("  → Buena capacidad discriminativa")
# elif auc >= 0.7:
#     print("  → Aceptable capacidad discriminativa")
# else:
#     print("  → Pobre capacidad discriminativa")

print()

# ============================================
# PASO 6: Curva Precision-Recall
# ============================================
print('--- Paso 6: Curva Precision-Recall ---')

# Descomenta las siguientes líneas:
# from sklearn.metrics import precision_recall_curve, average_precision_score
# from sklearn.metrics import PrecisionRecallDisplay
# 
# # Calcular PR curve
# precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_proba)
# ap = average_precision_score(y_test, y_proba)
# 
# print(f"Average Precision (AP): {ap:.4f}")
# 
# # Graficar
# fig, ax = plt.subplots(figsize=(8, 6))
# PrecisionRecallDisplay(precision=precision_curve, recall=recall_curve, 
#                        average_precision=ap, estimator_name='LogisticRegression').plot(ax=ax)
# ax.set_title('Curva Precision-Recall')
# plt.tight_layout()
# plt.savefig('pr_curve.png', dpi=100)
# plt.close()
# print("✓ Gráfica guardada: pr_curve.png")
# 
# print("\n¿Cuándo usar PR vs ROC?")
# print("- ROC: Clases balanceadas, comparación general")
# print("- PR:  Clases desbalanceadas, positivos raros")

print()

# ============================================
# PASO 7: Ajustar Umbral de Decisión
# ============================================
print('--- Paso 7: Ajustar Umbral de Decisión ---')

# Descomenta las siguientes líneas:
# print("Impacto del umbral de decisión:")
# print("-" * 60)
# print(f"{'Umbral':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>8}")
# print("-" * 60)
# 
# for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
#     y_pred_thresh = (y_proba >= threshold).astype(int)
#     p = precision_score(y_test, y_pred_thresh, zero_division=0)
#     r = recall_score(y_test, y_pred_thresh)
#     f = f1_score(y_test, y_pred_thresh)
#     print(f"{threshold:>8.1f} | {p:>9.4f} | {r:>8.4f} | {f:>8.4f}")
# 
# # Encontrar umbral óptimo para F1
# f1_scores = []
# for thresh in pr_thresholds:
#     y_pred_t = (y_proba >= thresh).astype(int)
#     f1_scores.append(f1_score(y_test, y_pred_t))
# 
# best_idx = np.argmax(f1_scores)
# best_threshold = pr_thresholds[best_idx]
# print(f"\n✓ Umbral óptimo para F1: {best_threshold:.3f} (F1={f1_scores[best_idx]:.4f})")

print()

# ============================================
# PASO 8: Dataset Desbalanceado
# ============================================
print('--- Paso 8: Dataset Desbalanceado ---')

# Descomenta las siguientes líneas:
# # Crear dataset muy desbalanceado (95% clase 0, 5% clase 1)
# X_imb, y_imb = make_classification(
#     n_samples=2000, n_features=20, n_informative=15,
#     n_classes=2, weights=[0.95, 0.05],
#     random_state=42
# )
# 
# print(f"Distribución: Clase 0 = {(y_imb == 0).sum()}, Clase 1 = {(y_imb == 1).sum()}")
# 
# # Split y entrenar
# X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
#     X_imb, y_imb, test_size=0.2, random_state=42, stratify=y_imb
# )
# 
# model_imb = LogisticRegression(max_iter=1000, random_state=42)
# model_imb.fit(X_train_i, y_train_i)
# 
# y_pred_i = model_imb.predict(X_test_i)
# y_proba_i = model_imb.predict_proba(X_test_i)[:, 1]
# 
# print("\nMétricas en dataset desbalanceado:")
# print("-" * 45)
# print(f"Accuracy:    {accuracy_score(y_test_i, y_pred_i):.4f} ← Engañosa!")
# print(f"Precision:   {precision_score(y_test_i, y_pred_i):.4f}")
# print(f"Recall:      {recall_score(y_test_i, y_pred_i):.4f}")
# print(f"F1-Score:    {f1_score(y_test_i, y_pred_i):.4f} ← Más realista")
# print(f"AUC-ROC:     {roc_auc_score(y_test_i, y_proba_i):.4f}")
# print(f"AP (PR):     {average_precision_score(y_test_i, y_proba_i):.4f} ← Mejor para desbalanceado")
# 
# print("\n⚠️ Conclusión:")
# print("En datasets desbalanceados, accuracy puede ser engañosa.")
# print("Preferir F1, AUC-ROC o Average Precision.")

print()

# ============================================
# RESUMEN
# ============================================
print('--- Resumen ---')

# Descomenta las siguientes líneas:
# print("=" * 50)
# print("RESUMEN DE MÉTRICAS DE CLASIFICACIÓN")
# print("=" * 50)
# print("\nMétrica        | Usar cuando...")
# print("-" * 50)
# print("Accuracy       | Clases balanceadas")
# print("Precision      | FP es costoso (spam, recomendaciones)")
# print("Recall         | FN es costoso (diagnóstico médico)")
# print("F1             | Balance entre P y R")
# print("AUC-ROC        | Comparar modelos en general")
# print("AP (PR-AUC)    | Clases muy desbalanceadas")
# print("=" * 50)

print()
print("¡Ejercicio completado!")
