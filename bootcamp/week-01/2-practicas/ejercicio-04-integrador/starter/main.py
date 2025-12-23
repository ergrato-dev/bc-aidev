# ============================================
# EJERCICIO 04: Ejercicio Integrador
# Simulador de Evaluación de Modelos ML
# ============================================
# Descomenta las líneas indicadas en cada paso
# Ejecuta después de cada paso para ver resultados
# ============================================

print("=" * 60)
print("🤖 SIMULADOR DE EVALUACIÓN DE MODELOS ML")
print("=" * 60)
print()

# ============================================
# PASO 1: Definir Datos de Modelos
# ============================================
print('--- Paso 1: Datos de Modelos ---')

# Descomenta las siguientes líneas:
# # Lista de modelos con sus métricas
# modelos = [
#     {"name": "Random Forest", "accuracy": 0.89, "precision": 0.87, "recall": 0.91},
#     {"name": "SVM", "accuracy": 0.85, "precision": 0.88, "recall": 0.82},
#     {"name": "Logistic Regression", "accuracy": 0.78, "precision": 0.80, "recall": 0.75},
#     {"name": "Neural Network", "accuracy": 0.92, "precision": 0.90, "recall": 0.94},
#     {"name": "KNN", "accuracy": 0.72, "precision": 0.70, "recall": 0.74},
#     {"name": "Naive Bayes", "accuracy": 0.68, "precision": 0.65, "recall": 0.72},
# ]

# print(f"Total de modelos a evaluar: {len(modelos)}")
# for modelo in modelos:
#     print(f"  - {modelo['name']}")

print()

# ============================================
# PASO 2: Calcular F1-Score para cada modelo
# ============================================
print('--- Paso 2: Calcular F1-Score ---')

# Descomenta las siguientes líneas:
# # F1 = 2 * (precision * recall) / (precision + recall)
# print("\nMétricas con F1-Score:")
# print("-" * 70)
# print(f"{'Modelo':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
# print("-" * 70)

# for modelo in modelos:
#     precision = modelo["precision"]
#     recall = modelo["recall"]
#     
#     # Calcular F1-Score
#     f1 = 2 * (precision * recall) / (precision + recall)
#     modelo["f1"] = f1  # Agregar al diccionario
#     
#     print(f"{modelo['name']:<25} {modelo['accuracy']:>10.2f} {precision:>10.2f} {recall:>10.2f} {f1:>10.2f}")

# print("-" * 70)

print()

# ============================================
# PASO 3: Clasificar Modelos
# ============================================
print('--- Paso 3: Clasificar Modelos ---')

# Descomenta las siguientes líneas:
# print("\nClasificación de modelos:")
# for modelo in modelos:
#     accuracy = modelo["accuracy"]
#     
#     # Clasificar según accuracy
#     if accuracy >= 0.90:
#         clasificacion = "🌟 Excelente"
#     elif accuracy >= 0.80:
#         clasificacion = "✅ Bueno"
#     elif accuracy >= 0.70:
#         clasificacion = "⚠️ Aceptable"
#     else:
#         clasificacion = "❌ Necesita mejora"
#     
#     modelo["clasificacion"] = clasificacion
#     print(f"  {modelo['name']:<25} → {clasificacion}")

print()

# ============================================
# PASO 4: Encontrar Mejor Modelo
# ============================================
print('--- Paso 4: Encontrar Mejor Modelo ---')

# Descomenta las siguientes líneas:
# # Encontrar el modelo con mayor accuracy
# mejor_modelo = None
# mejor_accuracy = 0

# for modelo in modelos:
#     if modelo["accuracy"] > mejor_accuracy:
#         mejor_accuracy = modelo["accuracy"]
#         mejor_modelo = modelo

# print(f"\n🏆 MEJOR MODELO: {mejor_modelo['name']}")
# print(f"   Accuracy: {mejor_modelo['accuracy']:.2f}")
# print(f"   Precision: {mejor_modelo['precision']:.2f}")
# print(f"   Recall: {mejor_modelo['recall']:.2f}")
# print(f"   F1-Score: {mejor_modelo['f1']:.2f}")

print()

# ============================================
# PASO 5: Filtrar Modelos
# ============================================
print('--- Paso 5: Filtrar Modelos ---')

# Descomenta las siguientes líneas:
# UMBRAL_ACCURACY = 0.80

# # Filtrar usando list comprehension
# modelos_aptos = [m for m in modelos if m["accuracy"] >= UMBRAL_ACCURACY]
# modelos_no_aptos = [m for m in modelos if m["accuracy"] < UMBRAL_ACCURACY]

# print(f"\nUmbral de accuracy: {UMBRAL_ACCURACY}")
# print(f"\n✅ Modelos APTOS ({len(modelos_aptos)}):")
# for m in modelos_aptos:
#     print(f"   - {m['name']} (accuracy: {m['accuracy']:.2f})")

# print(f"\n❌ Modelos NO APTOS ({len(modelos_no_aptos)}):")
# for m in modelos_no_aptos:
#     print(f"   - {m['name']} (accuracy: {m['accuracy']:.2f})")

print()

# ============================================
# PASO 6: Generar Reporte Final
# ============================================
print('--- Paso 6: Reporte Final ---')

# Descomenta las siguientes líneas:
# print("\n" + "=" * 60)
# print("📊 REPORTE DE EVALUACIÓN DE MODELOS")
# print("=" * 60)

# # Estadísticas generales
# total_modelos = len(modelos)
# modelos_excelentes = len([m for m in modelos if m["accuracy"] >= 0.90])
# modelos_buenos = len([m for m in modelos if 0.80 <= m["accuracy"] < 0.90])
# modelos_aceptables = len([m for m in modelos if 0.70 <= m["accuracy"] < 0.80])
# modelos_malos = len([m for m in modelos if m["accuracy"] < 0.70])

# # Calcular promedio de accuracy
# suma_accuracy = sum(m["accuracy"] for m in modelos)
# promedio_accuracy = suma_accuracy / total_modelos

# print(f"\n📈 ESTADÍSTICAS GENERALES")
# print(f"   Total de modelos evaluados: {total_modelos}")
# print(f"   Accuracy promedio: {promedio_accuracy:.2f}")
# print(f"\n📊 DISTRIBUCIÓN POR CLASIFICACIÓN")
# print(f"   🌟 Excelentes (≥90%): {modelos_excelentes}")
# print(f"   ✅ Buenos (80-89%): {modelos_buenos}")
# print(f"   ⚠️ Aceptables (70-79%): {modelos_aceptables}")
# print(f"   ❌ Necesitan mejora (<70%): {modelos_malos}")

# print(f"\n🏆 RECOMENDACIÓN")
# if mejor_modelo["accuracy"] >= 0.85:
#     print(f"   Usar '{mejor_modelo['name']}' para producción")
# else:
#     print(f"   Considerar mejorar modelos antes de producción")

# # Tasa de aprobación
# tasa_aprobacion = (len(modelos_aptos) / total_modelos) * 100
# print(f"\n📉 Tasa de aprobación: {tasa_aprobacion:.1f}%")

# print("\n" + "=" * 60)

print()

# ============================================
# ¡FELICIDADES! Has completado todos los ejercicios
# ============================================
print("=" * 60)
print("🎉 ¡FELICIDADES!")
print("   Has completado todos los ejercicios de la Semana 01")
print("   Ahora puedes continuar con el proyecto semanal")
print("=" * 60)
