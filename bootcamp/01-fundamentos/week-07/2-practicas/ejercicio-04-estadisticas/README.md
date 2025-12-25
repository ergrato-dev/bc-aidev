# 🏋️ Ejercicio 04: Estadísticas y Álgebra Lineal

## 🎯 Objetivo

Dominar las funciones de agregación estadística y operaciones básicas de álgebra lineal en NumPy.

---

## 📋 Pasos

### Paso 1: Agregaciones básicas

Funciones que reducen un array a un valor escalar.

```python
arr = np.array([1, 2, 3, 4, 5])
np.sum(arr)   # 15
np.mean(arr)  # 3.0
np.min(arr)   # 1
np.max(arr)   # 5
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Agregaciones con axis

Aplicar agregaciones a lo largo de un eje específico.

```python
matrix = np.array([[1, 2], [3, 4], [5, 6]])
np.sum(matrix, axis=0)  # [9 12] - suma columnas
np.sum(matrix, axis=1)  # [3 7 11] - suma filas
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Estadísticas descriptivas

Varianza, desviación estándar, percentiles y más.

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
np.std(arr)          # Desviación estándar
np.var(arr)          # Varianza
np.median(arr)       # Mediana
np.percentile(arr, 75)  # Percentil 75
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Encontrar valores

Localizar mínimos, máximos y valores específicos.

```python
arr = np.array([3, 1, 4, 1, 5, 9])
np.argmin(arr)  # 1 - índice del mínimo
np.argmax(arr)  # 5 - índice del máximo
np.where(arr > 3)  # Índices donde condición es True
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Producto de matrices

Multiplicación de matrices con `@`, `dot` y `matmul`.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
A @ B          # Producto matricial
np.dot(A, B)   # Equivalente
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Operaciones de álgebra lineal

Transpuesta, determinante, inversa y más.

```python
A = np.array([[1, 2], [3, 4]])
A.T                    # Transpuesta
np.linalg.det(A)       # Determinante
np.linalg.inv(A)       # Inversa
np.linalg.eig(A)       # Eigenvalores
```

**Descomenta** la sección del Paso 6.

---

### Paso 7: Caso práctico - Análisis de datos

Aplicar todo lo aprendido a un dataset real.

**Descomenta** la sección del Paso 7.

---

## ✅ Verificación

Al completar deberías poder:

- [ ] Calcular suma, media, min, max
- [ ] Usar axis para agregaciones por fila/columna
- [ ] Calcular std, var, mediana, percentiles
- [ ] Encontrar índices de valores específicos
- [ ] Realizar multiplicación de matrices
- [ ] Calcular transpuesta, determinante, inversa
- [ ] Analizar datos reales con NumPy
