# 🏋️ Ejercicio 02: Indexing y Slicing

## 🎯 Objetivo

Dominar el acceso y manipulación de elementos en arrays NumPy usando indexing, slicing y selección avanzada.

---

## 📋 Pasos

### Paso 1: Indexing básico 1D

Acceder a elementos individuales con índices positivos y negativos.

```python
arr = np.array([10, 20, 30, 40, 50])
arr[0]   # 10 - primer elemento
arr[-1]  # 50 - último elemento
arr[2]   # 30 - tercer elemento
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Indexing 2D y 3D

En arrays multidimensionales usamos múltiples índices.

```python
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix[0, 0]  # 1 - esquina superior izquierda
matrix[1, 2]  # 6 - fila 1, columna 2
matrix[-1, -1]  # 9 - última posición
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Slicing básico

Extraer secciones con la sintaxis `start:stop:step`.

```python
arr = np.arange(10)
arr[2:7]    # [2 3 4 5 6]
arr[::2]    # [0 2 4 6 8] - cada 2
arr[::-1]   # Invertir array
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Slicing 2D

Combinar slicing en filas y columnas.

```python
matrix = np.arange(20).reshape((4, 5))
matrix[1:3]       # Filas 1 y 2
matrix[:, 2]      # Columna 2
matrix[1:3, 2:4]  # Submatriz
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Fancy Indexing

Usar arrays de índices para seleccionar múltiples elementos.

```python
arr = np.array([10, 20, 30, 40, 50])
indices = [0, 2, 4]
arr[indices]  # [10 30 50]
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Boolean Indexing

Filtrar elementos usando condiciones booleanas.

```python
arr = np.array([1, 2, 3, 4, 5, 6])
mask = arr > 3
arr[mask]  # [4 5 6]
arr[arr % 2 == 0]  # [2 4 6] - pares
```

**Descomenta** la sección del Paso 6.

---

### Paso 7: Modificar con indexing

Asignar valores usando indexing y slicing.

```python
arr = np.zeros(5)
arr[0] = 10
arr[2:4] = [5, 6]
arr[arr == 0] = -1  # Reemplazar ceros
```

**Descomenta** la sección del Paso 7.

---

## ✅ Verificación

Al completar deberías poder:

- [ ] Acceder a elementos con índices positivos/negativos
- [ ] Usar indexing en arrays 2D y 3D
- [ ] Extraer secciones con slicing
- [ ] Seleccionar con arrays de índices
- [ ] Filtrar con máscaras booleanas
- [ ] Modificar elementos usando indexing
