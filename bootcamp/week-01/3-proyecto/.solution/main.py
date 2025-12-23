# ============================================
# PROYECTO SEMANA 01: Calculadora de Métricas ML
# SOLUCIÓN COMPLETA
# ============================================

# ============================================
# DATOS DE PRUEBA
# ============================================
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
          0, 1, 0, 0, 1, 1, 0, 1, 1, 0,
          1, 0, 0, 1, 1, 0, 1, 0, 1, 1,
          0, 0, 1, 1, 0, 1, 0, 1, 0, 1,
          1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
          0, 1, 0, 0, 1, 1, 0, 1, 1, 0,
          1, 0, 0, 1, 1, 0, 1, 0, 1, 1,
          0, 0, 1, 1, 0, 1, 0, 1, 0, 1,
          1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
          0, 1, 0, 0, 1, 1, 0, 1, 1, 0]

y_pred = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1,
          0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
          1, 0, 0, 1, 1, 1, 1, 0, 1, 0,
          0, 0, 1, 1, 0, 1, 0, 1, 0, 1,
          1, 0, 1, 0, 0, 0, 1, 0, 1, 1,
          0, 1, 1, 0, 1, 1, 0, 1, 1, 0,
          1, 0, 0, 1, 1, 0, 1, 0, 0, 1,
          0, 0, 1, 1, 0, 1, 0, 1, 0, 1,
          1, 0, 0, 1, 0, 0, 1, 0, 1, 1,
          0, 1, 0, 0, 1, 1, 0, 1, 1, 0]


# ============================================
# FUNCIÓN 1: Contar Matriz de Confusión
# ============================================
def count_confusion_matrix(y_true: list, y_pred: list) -> dict:
    """
    Cuenta los valores de la matriz de confusión.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for real, pred in zip(y_true, y_pred):
        if real == 1 and pred == 1:
            tp += 1
        elif real == 0 and pred == 0:
            tn += 1
        elif real == 0 and pred == 1:
            fp += 1
        elif real == 1 and pred == 0:
            fn += 1
    
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


# ============================================
# FUNCIÓN 2: Calcular Accuracy
# ============================================
def calculate_accuracy(confusion: dict) -> float:
    """
    Calcula el accuracy (exactitud) del modelo.
    Fórmula: (TP + TN) / (TP + TN + FP + FN)
    """
    tp = confusion["TP"]
    tn = confusion["TN"]
    fp = confusion["FP"]
    fn = confusion["FN"]
    
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    
    return (tp + tn) / total


# ============================================
# FUNCIÓN 3: Calcular Precision
# ============================================
def calculate_precision(confusion: dict) -> float:
    """
    Calcula la precision del modelo.
    Fórmula: TP / (TP + FP)
    """
    tp = confusion["TP"]
    fp = confusion["FP"]
    
    denominator = tp + fp
    if denominator == 0:
        return 0.0
    
    return tp / denominator


# ============================================
# FUNCIÓN 4: Calcular Recall
# ============================================
def calculate_recall(confusion: dict) -> float:
    """
    Calcula el recall (sensibilidad) del modelo.
    Fórmula: TP / (TP + FN)
    """
    tp = confusion["TP"]
    fn = confusion["FN"]
    
    denominator = tp + fn
    if denominator == 0:
        return 0.0
    
    return tp / denominator


# ============================================
# FUNCIÓN 5: Calcular F1-Score
# ============================================
def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calcula el F1-Score (media armónica de precision y recall).
    Fórmula: 2 * (precision * recall) / (precision + recall)
    """
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    
    return 2 * (precision * recall) / denominator


# ============================================
# FUNCIÓN 6: Clasificar Modelo
# ============================================
def classify_model(accuracy: float) -> str:
    """
    Clasifica el modelo según su accuracy.
    """
    if accuracy >= 0.90:
        return "🌟 Excelente"
    elif accuracy >= 0.80:
        return "✅ Bueno"
    elif accuracy >= 0.70:
        return "⚠️ Aceptable"
    else:
        return "❌ Necesita mejora"


# ============================================
# FUNCIÓN 7: Generar Reporte
# ============================================
def generate_report(metrics: dict, classification: str) -> None:
    """
    Genera e imprime un reporte de evaluación.
    """
    print("=" * 60)
    print("📊 REPORTE DE EVALUACIÓN")
    print("=" * 60)
    
    accuracy_pct = metrics["accuracy"] * 100
    precision_pct = metrics["precision"] * 100
    recall_pct = metrics["recall"] * 100
    
    print(f"El modelo tiene un accuracy de {accuracy_pct:.1f}%.")
    print(f"Con precision de {precision_pct:.1f}% y recall de {recall_pct:.1f}%.")
    print(f"F1-Score: {metrics['f1']:.2f}")
    print(f"Clasificación: {classification}")
    
    # Recomendación basada en clasificación
    if "Excelente" in classification:
        recomendacion = "Modelo listo para producción."
    elif "Bueno" in classification:
        recomendacion = "Modelo apto para uso en producción con monitoreo."
    elif "Aceptable" in classification:
        recomendacion = "Considerar mejoras antes de producción."
    else:
        recomendacion = "Requiere mejoras significativas. No usar en producción."
    
    print(f"Recomendación: {recomendacion}")
    print("=" * 60)


# ============================================
# PROGRAMA PRINCIPAL
# ============================================
def main():
    print("=" * 60)
    print("🤖 CALCULADORA DE MÉTRICAS ML")
    print("=" * 60)
    print()
    
    # Paso 1: Calcular matriz de confusión
    print("--- Matriz de Confusión ---")
    confusion = count_confusion_matrix(y_true, y_pred)
    print(f"TP (True Positives): {confusion['TP']}")
    print(f"TN (True Negatives): {confusion['TN']}")
    print(f"FP (False Positives): {confusion['FP']}")
    print(f"FN (False Negatives): {confusion['FN']}")
    print()
    
    # Paso 2: Calcular métricas
    print("--- Métricas Calculadas ---")
    accuracy = calculate_accuracy(confusion)
    precision = calculate_precision(confusion)
    recall = calculate_recall(confusion)
    f1 = calculate_f1_score(precision, recall)
    
    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1-Score:  {f1:.2f}")
    print()
    
    # Paso 3: Clasificar modelo
    print("--- Clasificación del Modelo ---")
    classification = classify_model(accuracy)
    print(classification)
    print()
    
    # Paso 4: Generar reporte
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    generate_report(metrics, classification)


if __name__ == "__main__":
    main()
