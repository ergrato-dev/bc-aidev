# Ejercicio 04: Feature Importance y Selección

## 🎯 Objetivo

Usar feature importance de Random Forest para selección de características, reducir dimensionalidad y mejorar modelos.

## 📋 Conceptos Clave

- `feature_importances_` de Random Forest
- Selección de features basada en importancia
- `SelectFromModel` para selección automática
- Permutation Importance como alternativa

## ⏱️ Tiempo Estimado

35 minutos

---

## 📝 Instrucciones

### Paso 1: Importar Librerías

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Cargar y Preparar Datos

```python
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
feature_names = cancer.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Features totales: {X.shape[1]}")
print(f"Muestras: {X.shape[0]}")
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Entrenar Random Forest

```python
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
baseline_acc = rf.score(X_test, y_test)
print(f"Baseline Accuracy (30 features): {baseline_acc:.4f}")
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Obtener Feature Importance

```python
importance = rf.feature_importances_

# Crear DataFrame ordenado
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\n--- Feature Importance (Top 10) ---")
print(importance_df.head(10).to_string(index=False))
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Visualizar Feature Importance

```python
top_n = 15
top_features = importance_df.head(top_n)

plt.figure(figsize=(10, 8))
plt.barh(range(top_n), top_features['importance'].values[::-1], color='steelblue')
plt.yticks(range(top_n), top_features['feature'].values[::-1])
plt.xlabel('Importancia')
plt.title(f'Top {top_n} Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig('feature_importance_analysis.png', dpi=150)
plt.show()
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Selección Manual de Features

Seleccionamos solo las features más importantes.

```python
# Top 10 features
top_10_features = importance_df.head(10)['feature'].values
top_10_indices = [list(feature_names).index(f) for f in top_10_features]

X_train_top10 = X_train[:, top_10_indices]
X_test_top10 = X_test[:, top_10_indices]

rf_top10 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_top10.fit(X_train_top10, y_train)
top10_acc = rf_top10.score(X_test_top10, y_test)

print(f"\n--- Selección Manual ---")
print(f"Accuracy con 30 features: {baseline_acc:.4f}")
print(f"Accuracy con 10 features: {top10_acc:.4f}")
print(f"Diferencia: {(top10_acc - baseline_acc)*100:.2f}%")
```

**Descomenta** la sección del Paso 6.

---

### Paso 7: SelectFromModel Automático

Sklearn puede seleccionar features automáticamente.

```python
# Selección automática basada en importancia media
selector = SelectFromModel(rf, threshold='mean', prefit=True)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print(f"\n--- SelectFromModel (threshold='mean') ---")
print(f"Features originales: {X_train.shape[1]}")
print(f"Features seleccionados: {X_train_selected.shape[1]}")

# Features seleccionados
selected_mask = selector.get_support()
selected_features = feature_names[selected_mask]
print(f"Features: {list(selected_features)}")

# Evaluar
rf_selected = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_selected.fit(X_train_selected, y_train)
selected_acc = rf_selected.score(X_test_selected, y_test)
print(f"Accuracy: {selected_acc:.4f}")
```

**Descomenta** la sección del Paso 7.

---

### Paso 8: Permutation Importance

Método alternativo más robusto para evaluar importancia.

```python
perm_importance = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Crear DataFrame
perm_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("\n--- Permutation Importance (Top 10) ---")
print(perm_df.head(10).to_string(index=False))
```

**Descomenta** la sección del Paso 8.

---

### Paso 9: Comparar Métodos de Importancia

```python
# Comparar rankings
rf_ranking = importance_df.reset_index(drop=True)
rf_ranking['rf_rank'] = range(1, len(rf_ranking) + 1)

perm_ranking = perm_df.reset_index(drop=True)
perm_ranking['perm_rank'] = range(1, len(perm_ranking) + 1)

comparison = pd.merge(
    rf_ranking[['feature', 'rf_rank']],
    perm_ranking[['feature', 'perm_rank']],
    on='feature'
).sort_values('rf_rank')

print("\n--- Comparación de Rankings (Top 10) ---")
print(comparison.head(10).to_string(index=False))
```

**Descomenta** la sección del Paso 9.

---

### Paso 10: Curva de Features vs Accuracy

¿Cuántas features necesitamos realmente?

```python
n_features_range = [1, 3, 5, 7, 10, 15, 20, 25, 30]
accuracies = []

sorted_indices = np.argsort(importance)[::-1]

for n in n_features_range:
    top_indices = sorted_indices[:n]
    X_train_n = X_train[:, top_indices]
    X_test_n = X_test[:, top_indices]

    rf_n = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_n.fit(X_train_n, y_train)
    accuracies.append(rf_n.score(X_test_n, y_test))

plt.figure(figsize=(10, 6))
plt.plot(n_features_range, accuracies, 'b-o', linewidth=2, markersize=8)
plt.xlabel('Número de Features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Número de Features (ordenadas por importancia)')
plt.grid(True, alpha=0.3)
plt.xticks(n_features_range)
plt.savefig('features_vs_accuracy.png', dpi=150)
plt.show()

print("\n--- Accuracy vs Features ---")
for n, acc in zip(n_features_range, accuracies):
    print(f"{n:2d} features: {acc:.4f}")
```

**Descomenta** la sección del Paso 10.

---

## ✅ Resultado Esperado

1. Feature importance extraída y visualizada
2. Top features identificados (worst perimeter, worst concave points, etc.)
3. Modelo con menos features mantiene rendimiento similar
4. Comparación entre métodos de importancia
5. Curva mostrando features óptimas (~10-15 suficientes)

---

## 🔬 Experimenta

1. Prueba `threshold='median'` en SelectFromModel
2. Usa solo las 5 features más importantes
3. ¿Qué pasa si usas las 5 menos importantes?

---

## 📚 Recursos

- [Feature Importance - sklearn](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
- [SelectFromModel - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)
- [Permutation Importance - sklearn](https://scikit-learn.org/stable/modules/permutation_importance.html)
