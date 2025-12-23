# 📖 Ejercicio 03: Diccionarios y Sets

## 🎯 Objetivo

Practicar el uso de diccionarios y sets en Python.

---

## 📋 Pasos

### Paso 1: Crear y Acceder Diccionarios

Crea diccionarios y accede a valores:

```python
person = {"name": "Ana", "age": 25, "city": "Madrid"}
print(person["name"])
print(person.get("email", "N/A"))
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Modificar Diccionarios

Agrega, modifica y elimina elementos:

```python
person["email"] = "ana@mail.com"
person["age"] = 26
del person["city"]
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Iterar Diccionarios

Recorre keys, values e items:

```python
for key, value in person.items():
    print(f"{key}: {value}")
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Comprensiones de Diccionario

Crea diccionarios con comprehensions:

```python
squares = {x: x**2 for x in range(6)}
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Sets Básicos

Trabaja con conjuntos:

```python
numbers = {1, 2, 3, 4, 5}
numbers.add(6)
numbers.discard(1)
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Operaciones de Sets

Aplica unión, intersección, diferencia:

```python
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
print(A | B)  # Unión
print(A & B)  # Intersección
```

**Descomenta** la sección del Paso 6.

---

## ▶️ Ejecución

```bash
cd bootcamp/week-02/2-practicas/ejercicio-03-diccionarios
python starter/main.py
```

---

## ✅ Resultado Esperado

```
--- Paso 1: Crear y Acceder ---
Persona: {'name': 'Ana', 'age': 25, 'city': 'Madrid'}
Nombre: Ana
Email: N/A

--- Paso 2: Modificar ---
Agregado email, modificado age, eliminado city
Resultado: {'name': 'Ana', 'age': 26, 'email': 'ana@mail.com'}

--- Paso 3: Iterar ---
name: Ana
age: 26
email: ana@mail.com

--- Paso 4: Comprensiones ---
Cuadrados: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

--- Paso 5: Sets Básicos ---
Set inicial: {1, 2, 3, 4, 5}
Después de add(6): {1, 2, 3, 4, 5, 6}
Después de discard(1): {2, 3, 4, 5, 6}

--- Paso 6: Operaciones de Sets ---
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
Unión: {1, 2, 3, 4, 5, 6}
Intersección: {3, 4}
Diferencia A-B: {1, 2}
```

---

_Siguiente: [Ejercicio 04 - Integrador](../ejercicio-04-integrador/)_
