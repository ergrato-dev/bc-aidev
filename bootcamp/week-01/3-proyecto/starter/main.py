# ============================================
# PROYECTO SEMANA 01: Calculadora de Métricas ML
# ============================================
# Implementa las funciones marcadas con TODO
# Ejecuta el programa para verificar tu solución
# ============================================

# ============================================
# DATOS DE PRUEBA
# ============================================
# Simulamos predicciones de un modelo de clasificación binaria
# 1 = Positivo (ej: tiene la enfermedad)
# 0 = Negativo (ej: no tiene la enfermedad)

# Valores reales (ground truth)
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

# Predicciones del modelo
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
    
    Args:
        y_true: Lista de valores reales (0 o 1)
        y_pred: Lista de predicciones (0 o 1)
    
    Returns:
        dict: Diccionario con TP, TN, FP, FN
        
    Ejemplo:
        Si y_true=[1,0,1,0] y y_pred=[1,0,0,1]
        - TP=1 (predicho 1, real 1)
        - TN=1 (predicho 0, real 0)
        - FP=1 (predicho 1, real 0)
        - FN=1 (predicho 0, real 1)
    """
    # TODO: Inicializar contadores
    tp = 0  # True Positives
    tn = 0  # True Negatives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    
    # TODO: Recorrer ambas listas y contar cada caso
    # Hint: Usa zip(y_true, y_pred) para iterar ambas listas
    # Hint: if real == 1 and pred == 1: es un TP
    
    # TODO: Retornar diccionario con los conteos
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


# ============================================
# FUNCIÓN 2: Calcular Accuracy
# ============================================
def calculate_accuracy(confusion: dict) -> float:
    """
    Calcula el accuracy (exactitud) del modelo.
    
    Fórmula: (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        confusion: Diccionario con TP, TN, FP, FN
    
    Returns:
        float: Accuracy entre 0 y 1
    """
    # TODO: Extraer valores del diccionario
    # TODO: Aplicar la fórmula
    # TODO: Retornar el resultado
    
    return 0.0  # Placeholder


# ============================================
# FUNCIÓN 3: Calcular Precision
# ============================================
def calculate_precision(confusion: dict) -> float:
    """
    Calcula la precision del modelo.
    
    Fórmula: TP / (TP + FP)
    
    Precision responde: De todos los que predije positivos,
    ¿cuántos realmente lo eran?
    
    Args:
        confusion: Diccionario con TP, TN, FP, FN
    
    Returns:
        float: Precision entre 0 y 1
    """
    # TODO: Extraer TP y FP
    # TODO: Cuidado con división por cero (si TP + FP == 0)
    # TODO: Aplicar la fórmula
    
    return 0.0  # Placeholder


# ============================================
# FUNCIÓN 4: Calcular Recall
# ============================================
def calculate_recall(confusion: dict) -> float:
    """
    Calcula el recall (sensibilidad) del modelo.
    
    Fórmula: TP / (TP + FN)
    
    Recall responde: De todos los positivos reales,
    ¿cuántos detecté correctamente?
    
    Args:
        confusion: Diccionario con TP, TN, FP, FN
    
    Returns:
        float: Recall entre 0 y 1
    """
    # TODO: Extraer TP y FN
    # TODO: Cuidado con división por cero
    # TODO: Aplicar la fórmula
    
    return 0.0  # Placeholder


# ============================================
# FUNCIÓN 5: Calcular F1-Score
# ============================================
def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calcula el F1-Score (media armónica de precision y recall).
    
    Fórmula: 2 * (precision * recall) / (precision + recall)
    
    Args:
        precision: Valor de precision
        recall: Valor de recall
    
    Returns:
        float: F1-Score entre 0 y 1
    """
    # TODO: Cuidado con división por cero
    # TODO: Aplicar la fórmula
    
    return 0.0  # Placeholder


# ============================================
# FUNCIÓN 6: Clasificar Modelo
# ============================================
def classify_model(accuracy: float) -> str:
    """
    Clasifica el modelo según su accuracy.
    
    Clasificación:
    - >= 0.90: "🌟 Excelente"
    - >= 0.80: "✅ Bueno"  
    - >= 0.70: "⚠️ Aceptable"
    - < 0.70:  "❌ Necesita mejora"
    
    Args:
        accuracy: Valor de accuracy
    
    Returns:
        str: Clasificación del modelo
    """
    # TODO: Usar if/elif/else para clasificar
    
    return ""  # Placeholder


# ============================================
# FUNCIÓN 7: Generar Reporte
# ============================================
def generate_report(metrics: dict, classification: str) -> None:
    """
    Genera e imprime un reporte de evaluación.
    
    Args:
        metrics: Diccionario con accuracy, precision, recall, f1
        classification: Clasificación del modelo
    """
    # TODO: Imprimir reporte formateado
    # Incluir:
    # - Accuracy como porcentaje
    # - Precision como porcentaje
    # - Recall como porcentaje
    # - F1-Score
    # - Clasificación
    # - Recomendación basada en la clasificación
    
    print("TODO: Implementar reporte")


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


# Ejecutar programa
if __name__ == "__main__":
    main()
