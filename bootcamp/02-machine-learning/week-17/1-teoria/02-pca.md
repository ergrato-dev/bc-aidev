# 📊 PCA: Análisis de Componentes Principales

## 🎯 Objetivos

- Comprender la teoría matemática de PCA
- Implementar PCA desde cero con NumPy
- Usar PCA con Scikit-learn
- Seleccionar el número óptimo de componentes
- Interpretar los resultados de PCA

---

## 📚 ¿Qué es PCA?

**PCA (Principal Component Analysis)** es una técnica de reducción dimensional lineal que transforma los datos a un nuevo sistema de coordenadas donde:

1. La primera coordenada (PC1) captura la máxima varianza
2. La segunda (PC2) captura la máxima varianza restante, ortogonal a PC1
3. Y así sucesivamente...

![Concepto de PCA](../0-assets/02-pca-concept.svg)

---

## 🧮 Matemáticas de PCA

### Paso 1: Centrar los Datos

$$X_{centered} = X - \mu$$

Donde $\mu$ es la media de cada feature.

### Paso 2: Calcular Matriz de Covarianza

$$\Sigma = \frac{1}{n-1} X_{centered}^T X_{centered}$$

### Paso 3: Calcular Autovalores y Autovectores

$$\Sigma v = \lambda v$$

- $\lambda$: autovalores (varianza en cada dirección)
- $v$: autovectores (direcciones principales)

### Paso 4: Ordenar por Autovalor

Ordenar componentes de mayor a menor $\lambda$.

### Paso 5: Proyectar

$$X_{reduced} = X_{centered} \cdot W_k$$

Donde $W_k$ son los top-k autovectores.

---

## 💻 Implementación desde Cero

```python
import numpy as np

class PCAFromScratch:
    """PCA implementado desde cero."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray):
        """Ajusta PCA a los datos."""
        # Paso 1: Centrar datos
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Paso 2: Matriz de covarianza
        cov_matrix = np.cov(X_centered.T)

        # Paso 3: Autovalores y autovectores
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Convertir a real (pueden ser complejos por errores numéricos)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        # Paso 4: Ordenar por autovalor descendente
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Guardar top-k componentes
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / eigenvalues.sum()

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Proyecta datos al espacio reducido."""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit y transform en un paso."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """Reconstruye datos originales aproximados."""
        return X_reduced @ self.components_ + self.mean_


# Ejemplo de uso
# X = np.random.randn(100, 10)  # 100 muestras, 10 features
# pca = PCAFromScratch(n_components=3)
# X_reduced = pca.fit_transform(X)
# print(f"Shape original: {X.shape}")
# print(f"Shape reducido: {X_reduced.shape}")
# print(f"Varianza explicada: {pca.explained_variance_ratio_.sum()*100:.1f}%")
```

---

## 🔧 PCA con Scikit-learn

### Uso Básico

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generar datos de ejemplo
np.random.seed(42)
X = np.random.randn(200, 20)  # 200 muestras, 20 features

# Siempre escalar antes de PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA con número fijo de componentes
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

print(f"Shape original: {X.shape}")
print(f"Shape reducido: {X_pca.shape}")
print(f"Varianza explicada por componente: {pca.explained_variance_ratio_}")
print(f"Varianza total explicada: {pca.explained_variance_ratio_.sum()*100:.1f}%")
```

### Seleccionar por Varianza

```python
# PCA que mantiene 95% de varianza
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)

print(f"Componentes para 95% varianza: {pca_95.n_components_}")
```

---

## 📈 Visualización de Resultados

### Scree Plot (Varianza por Componente)

```python
import matplotlib.pyplot as plt

def plot_scree(pca, title="Scree Plot"):
    """Visualiza varianza explicada por componente."""
    n_components = len(pca.explained_variance_ratio_)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Varianza individual
    axes[0].bar(range(1, n_components + 1),
                pca.explained_variance_ratio_ * 100,
                color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Componente Principal')
    axes[0].set_ylabel('Varianza Explicada (%)')
    axes[0].set_title('Varianza por Componente')

    # Varianza acumulada
    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
    axes[1].plot(range(1, n_components + 1), cumulative,
                 'go-', linewidth=2, markersize=8)
    axes[1].axhline(y=95, color='r', linestyle='--', label='95% varianza')
    axes[1].set_xlabel('Número de Componentes')
    axes[1].set_ylabel('Varianza Acumulada (%)')
    axes[1].set_title('Varianza Acumulada')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# pca_full = PCA()
# pca_full.fit(X_scaled)
# plot_scree(pca_full)
```

### Proyección 2D

```python
def plot_pca_2d(X_pca, y=None, title="PCA 2D"):
    """Visualiza datos en primeros 2 componentes."""
    plt.figure(figsize=(10, 7))

    if y is not None:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                              cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Clase')
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

# plot_pca_2d(X_pca[:, :2], y=labels)
```

---

## 🔍 Interpretación de Componentes

### Ver qué Features Contribuyen

```python
def interpret_components(pca, feature_names, n_top=5):
    """Muestra features más importantes por componente."""
    for i, component in enumerate(pca.components_):
        print(f"\n=== PC{i+1} (var: {pca.explained_variance_ratio_[i]*100:.1f}%) ===")

        # Índices ordenados por valor absoluto
        top_idx = np.argsort(np.abs(component))[::-1][:n_top]

        for idx in top_idx:
            print(f"  {feature_names[idx]}: {component[idx]:.3f}")

# feature_names = [f'feature_{i}' for i in range(X.shape[1])]
# interpret_components(pca, feature_names)
```

### Biplot

```python
def biplot(X_pca, pca, feature_names, scale=1):
    """Biplot: datos + vectores de features."""
    plt.figure(figsize=(12, 8))

    # Datos
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, s=30)

    # Vectores de features
    for i, (name, vec) in enumerate(zip(feature_names, pca.components_.T)):
        plt.arrow(0, 0, vec[0]*scale, vec[1]*scale,
                  color='red', alpha=0.7, head_width=0.05)
        plt.text(vec[0]*scale*1.1, vec[1]*scale*1.1, name,
                 color='red', fontsize=9)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Biplot PCA')
    plt.grid(True, alpha=0.3)
    plt.show()
```

---

## 🎯 Selección del Número de Componentes

### Métodos Comunes

```python
def find_optimal_components(X, variance_threshold=0.95):
    """Encuentra K óptimo para umbral de varianza."""
    pca = PCA()
    pca.fit(X)

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative >= variance_threshold) + 1

    print(f"Componentes para {variance_threshold*100}% varianza: {n_components}")
    print(f"De {X.shape[1]} features originales")

    return n_components, pca

# n_opt, pca_full = find_optimal_components(X_scaled, 0.95)
```

### Regla del Codo

Buscar el "codo" en el scree plot donde la varianza marginal disminuye significativamente.

### Criterio de Kaiser

Mantener componentes con autovalor > 1 (solo datos estandarizados).

```python
def kaiser_criterion(pca):
    """Componentes con autovalor > 1."""
    n = sum(pca.explained_variance_ > 1)
    print(f"Kaiser: mantener {n} componentes")
    return n
```

---

## 🔄 Reconstrucción de Datos

```python
# Datos originales → reducidos → reconstruidos
X_reconstructed = pca.inverse_transform(X_pca)

# Error de reconstrucción
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"Error de reconstrucción (MSE): {reconstruction_error:.4f}")
```

---

## ⚠️ Consideraciones Importantes

### 1. Siempre Escalar

```python
# ✅ CORRECTO
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
pca.fit(X_scaled)

# ❌ INCORRECTO (features con diferentes escalas dominarán)
pca.fit(X)  # Sin escalar
```

### 2. PCA es Lineal

No captura relaciones no lineales. Considerar t-SNE/UMAP para esos casos.

### 3. Sensible a Outliers

Los outliers afectan la dirección de componentes principales.

### 4. Pérdida de Interpretabilidad

Los componentes son combinaciones de features originales.

---

## 📊 PCA en Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Pipeline con PCA
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # 95% varianza
    ('classifier', SVC(kernel='rbf'))
])

# Cross-validation
# scores = cross_val_score(pipeline, X, y, cv=5)
# print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## ✅ Resumen

| Aspecto           | Detalle                                 |
| ----------------- | --------------------------------------- |
| **Qué hace**      | Proyección lineal que maximiza varianza |
| **Entrada**       | Matriz de datos (n × d)                 |
| **Salida**        | Matriz reducida (n × k)                 |
| **Parámetro**     | n_components (fijo o % varianza)        |
| **Prerrequisito** | Datos escalados                         |
| **Fortalezas**    | Rápido, interpretable, determinístico   |
| **Debilidades**   | Solo lineal, sensible a outliers        |

---

## 🔗 Navegación

| ⬅️ Anterior                                          | 🏠 Semana 17           | Siguiente ➡️        |
| ---------------------------------------------------- | ---------------------- | ------------------- |
| [Intro Reducción](01-intro-reduccion-dimensional.md) | [README](../README.md) | [t-SNE](03-tsne.md) |
