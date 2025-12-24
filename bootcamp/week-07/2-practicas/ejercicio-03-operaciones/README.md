# 🏋️ Ejercicio 03: Operaciones Vectorizadas

## 🎯 Objetivo

Dominar las operaciones elemento a elemento, broadcasting y funciones universales (ufuncs) de NumPy.

---

## 📋 Pasos

### Paso 1: Operaciones aritméticas básicas

Las operaciones se aplican elemento a elemento automáticamente.

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

a + b   # [11 22 33 44]
a * b   # [10 40 90 160]
a ** 2  # [1 4 9 16]
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Operaciones con escalares

Los escalares se aplican a todos los elementos (broadcasting).

```python
arr = np.array([1, 2, 3, 4])
arr + 10   # [11 12 13 14]
arr * 2    # [2 4 6 8]
arr / 10   # [0.1 0.2 0.3 0.4]
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Broadcasting con diferentes shapes

NumPy expande automáticamente arrays para operaciones compatibles.

```python
matrix = np.ones((3, 4))
row = np.array([1, 2, 3, 4])
matrix + row  # Suma el row a cada fila
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Funciones matemáticas (ufuncs)

NumPy ofrece funciones optimizadas que operan elemento a elemento.

```python
arr = np.array([1, 4, 9, 16])
np.sqrt(arr)   # [1. 2. 3. 4.]
np.exp(arr)    # Exponencial
np.log(arr)    # Logaritmo natural
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Funciones trigonométricas

Funciones para cálculos con ángulos (en radianes).

```python
angles = np.array([0, np.pi/2, np.pi])
np.sin(angles)  # [0. 1. 0.]
np.cos(angles)  # [1. 0. -1.]
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Comparaciones y operaciones lógicas

Comparaciones retornan arrays booleanos.

```python
arr = np.array([1, 2, 3, 4, 5])
arr > 3        # [False False False True True]
arr == 3       # [False False True False False]
np.all(arr > 0)  # True
np.any(arr > 4)  # True
```

**Descomenta** la sección del Paso 6.

---

### Paso 7: Operaciones in-place vs copias

Entender cuándo se modifica el array original.

```python
arr = np.array([1, 2, 3])
arr += 10      # In-place: modifica arr
result = arr + 10  # Copia: arr no cambia
```

**Descomenta** la sección del Paso 7.

---

## ✅ Verificación

Al completar deberías poder:

- [ ] Realizar operaciones elemento a elemento
- [ ] Aplicar escalares a arrays
- [ ] Entender broadcasting básico
- [ ] Usar funciones matemáticas (sqrt, exp, log)
- [ ] Usar funciones trigonométricas
- [ ] Realizar comparaciones vectorizadas
- [ ] Distinguir operaciones in-place vs copias
