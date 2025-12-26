# 📖 Glosario - Semana 15

## Validación Cruzada y Métricas de Evaluación

Términos clave ordenados alfabéticamente.

---

### A

**Accuracy (Exactitud)**
Proporción de predicciones correctas sobre el total.
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
⚠️ Engañosa con clases desbalanceadas.

**AP (Average Precision)**
Área bajo la curva Precision-Recall. Mejor que AUC-ROC para clases desbalanceadas.

**AUC (Area Under Curve)**
Área bajo la curva ROC. Mide la capacidad discriminativa del modelo (0.5 = random, 1.0 = perfecto).

---

### B

**Bias**
Error sistemático del modelo. Un modelo con alto bias es demasiado simple y no captura patrones (underfitting).

**Bias-Variance Tradeoff**
Compromiso entre la complejidad del modelo: muy simple = alto bias, muy complejo = alta varianza.

---

### C

**Classification Report**
Resumen de métricas de clasificación: precision, recall, F1 por clase.

```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
```

**Confusion Matrix (Matriz de Confusión)**
Tabla que muestra TP, TN, FP, FN. Base para calcular todas las métricas de clasificación.

**Cross-Validation (Validación Cruzada)**
Técnica que divide los datos en K partes para entrenar y evaluar múltiples veces, obteniendo una estimación más robusta del rendimiento.

**cross_val_score**
Función de scikit-learn para realizar cross-validation en una línea.

```python
scores = cross_val_score(model, X, y, cv=5)
```

---

### D

**Data Leakage**
Cuando información del conjunto de test "se filtra" al entrenamiento, causando métricas optimistas pero poco realistas.

---

### F

**F1-Score**
Media armónica de Precision y Recall. Balancea ambas métricas.
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**False Negative (FN)**
Positivo real clasificado incorrectamente como negativo. Error Tipo II.

**False Positive (FP)**
Negativo real clasificado incorrectamente como positivo. Error Tipo I.

**FPR (False Positive Rate)**
Tasa de falsos positivos: FP / (FP + TN). Eje X de la curva ROC.

---

### G

**GridSearchCV**
Búsqueda exhaustiva de hiperparámetros probando todas las combinaciones posibles.

```python
grid = GridSearchCV(model, param_grid, cv=5)
```

**Group K-Fold**
Variante de K-Fold que garantiza que grupos (ej: pacientes) no se mezclen entre train y test.

---

### H

**Holdout**
Método de validación simple: dividir datos en train y test una sola vez.

**Hyperparameter (Hiperparámetro)**
Parámetro del modelo que se define antes del entrenamiento (ej: n_estimators, learning_rate).

---

### K

**K-Fold Cross-Validation**
Divide los datos en K partes. En cada iteración, K-1 partes para train, 1 para validación.

---

### L

**Leave-One-Out (LOO)**
Cross-validation donde K = número de muestras. Cada muestra es un fold de test.

---

### M

**MAE (Mean Absolute Error)**
Error absoluto promedio. Robusto a outliers.
$$MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|$$

**MAPE (Mean Absolute Percentage Error)**
Error porcentual promedio. Útil cuando necesitas error relativo.

**MSE (Mean Squared Error)**
Error cuadrático promedio. Penaliza errores grandes.
$$MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$$

---

### N

**Nested Cross-Validation**
CV anidado: CV externo para evaluación, CV interno para selección de hiperparámetros. Evita sesgo optimista.

---

### O

**Overfitting (Sobreajuste)**
Modelo demasiado complejo que memoriza el training pero no generaliza. Error bajo en train, alto en test.

---

### P

**Precision (Precisión)**
De todos los predichos positivos, ¿cuántos son realmente positivos?
$$\text{Precision} = \frac{TP}{TP + FP}$$
Importante cuando FP es costoso.

**Precision-Recall Curve**
Curva que muestra Precision vs Recall a diferentes umbrales. Mejor que ROC para clases desbalanceadas.

---

### R

**R² (Coeficiente de Determinación)**
Proporción de varianza explicada por el modelo. 1.0 = perfecto, 0.0 = igual que predecir la media.
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

**RandomizedSearchCV**
Búsqueda aleatoria de hiperparámetros. Más eficiente que GridSearch para espacios grandes.

**Recall (Sensibilidad / TPR)**
De todos los positivos reales, ¿cuántos detectó el modelo?
$$\text{Recall} = \frac{TP}{TP + FN}$$
Importante cuando FN es costoso.

**RMSE (Root Mean Squared Error)**
Raíz del MSE. En las mismas unidades que la variable objetivo.
$$RMSE = \sqrt{MSE}$$

**ROC Curve**
Curva que grafica TPR vs FPR a diferentes umbrales de clasificación.

---

### S

**Scoring**
Parámetro de cross_val_score que indica qué métrica optimizar ('accuracy', 'f1', 'roc_auc', etc.).

**Stratified K-Fold**
K-Fold que mantiene la proporción de clases en cada fold. Esencial para clases desbalanceadas.

---

### T

**Test Set (Conjunto de Prueba)**
Datos reservados para evaluación final. NUNCA se usan para entrenamiento ni selección de hiperparámetros.

**TPR (True Positive Rate)**
Igual que Recall. TP / (TP + FN). Eje Y de la curva ROC.

**Train Set (Conjunto de Entrenamiento)**
Datos usados para entrenar el modelo.

**True Negative (TN)**
Negativo real correctamente clasificado como negativo.

**True Positive (TP)**
Positivo real correctamente clasificado como positivo.

---

### U

**Umbral (Threshold)**
Valor que determina la clasificación. Por defecto 0.5 en clasificación binaria. Ajustable según necesidades.

**Underfitting (Subajuste)**
Modelo demasiado simple que no captura patrones. Error alto en train y test.

---

### V

**Validation Set (Conjunto de Validación)**
Datos usados para ajustar hiperparámetros durante el desarrollo. Diferente de test set.

**Variance (Varianza)**
Sensibilidad del modelo a cambios en los datos de entrenamiento. Alta varianza = overfitting.

---

## 📊 Tabla de Métricas Rápida

| Clasificación | Regresión |
| ------------- | --------- |
| Accuracy      | R²        |
| Precision     | MSE       |
| Recall        | RMSE      |
| F1-Score      | MAE       |
| AUC-ROC       | MAPE      |
| AP (PR-AUC)   |           |

---

## 🔗 Referencias

- [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-learn Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
