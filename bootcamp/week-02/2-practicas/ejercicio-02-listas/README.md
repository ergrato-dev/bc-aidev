# 📚 Ejercicio 02: Listas y Tuplas

## 🎯 Objetivo

Practicar la manipulación de listas y tuplas en Python.

---

## 📋 Pasos

### Paso 1: Crear y Acceder

Crea listas y accede a elementos con índices:

```python
fruits = ["apple", "banana", "cherry", "date"]
print(fruits[0])   # Primero
print(fruits[-1])  # Último
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Slicing

Extrae sublistas con slicing:

```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(numbers[2:5])   # [2, 3, 4]
print(numbers[::2])   # Pares de índice
print(numbers[::-1])  # Reverso
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Métodos de Listas

Usa los métodos más comunes:

```python
colors = ["red", "green"]
colors.append("blue")
colors.insert(0, "yellow")
colors.remove("green")
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Comprensiones de Lista

Crea listas de forma pythónica:

```python
squares = [x ** 2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Tuplas

Trabaja con tuplas inmutables:

```python
point = (10, 20, 30)
x, y, z = point  # Desempaquetado
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Listas en ML

Aplica listas en contexto de ML:

```python
# Normalización min-max
values = [10, 20, 30, 40, 50]
normalized = [(v - min(values)) / (max(values) - min(values)) for v in values]
```

**Descomenta** la sección del Paso 6.

---

## ▶️ Ejecución

```bash
cd bootcamp/week-02/2-practicas/ejercicio-02-listas
python starter/main.py
```

---

## ✅ Resultado Esperado

```
--- Paso 1: Crear y Acceder ---
Lista: ['apple', 'banana', 'cherry', 'date']
Primero: apple
Último: date
Índice 2: cherry

--- Paso 2: Slicing ---
Original: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[2:5]: [2, 3, 4]
[::2]: [0, 2, 4, 6, 8]
[::-1]: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

--- Paso 3: Métodos de Listas ---
append, insert, remove...
Lista final: ['yellow', 'red', 'blue']

--- Paso 4: Comprensiones de Lista ---
Cuadrados: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
Pares: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

--- Paso 5: Tuplas ---
Punto: (10, 20, 30)
Desempaquetado: x=10, y=20, z=30

--- Paso 6: Listas en ML ---
Original: [10, 20, 30, 40, 50]
Normalizado: [0.0, 0.25, 0.5, 0.75, 1.0]
```

---

_Siguiente: [Ejercicio 03 - Diccionarios](../ejercicio-03-diccionarios/)_
