# 🔧 Ejercicio 01: Funciones

## 🎯 Objetivo

Practicar la definición y uso de funciones en Python.

---

## 📋 Pasos

### Paso 1: Función Básica

Define tu primera función con parámetros y return:

```python
def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"

message = greet("Ana")
print(message)  # Hello, Ana!
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Parámetros por Defecto

Usa valores por defecto para parámetros opcionales:

```python
def power(base: int, exponent: int = 2) -> int:
    """Calculate base raised to exponent."""
    return base ** exponent

print(power(5))      # 25 (usa default exponent=2)
print(power(2, 10))  # 1024
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Retorno Múltiple

Las funciones pueden retornar múltiples valores como tupla:

```python
def get_stats(numbers: list) -> tuple:
    """Return min, max, and average."""
    return min(numbers), max(numbers), sum(numbers) / len(numbers)

minimum, maximum, average = get_stats([10, 20, 30, 40, 50])
print(f"Min: {minimum}, Max: {maximum}, Avg: {average}")
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: \*args y \*\*kwargs

Acepta un número variable de argumentos:

```python
def sum_all(*args) -> int:
    """Sum any number of arguments."""
    return sum(args)

def print_info(**kwargs) -> None:
    """Print key-value pairs."""
    for key, value in kwargs.items():
        print(f"{key}: {value}")
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Funciones Lambda

Crea funciones anónimas de una línea:

```python
# Lambda para elevar al cuadrado
square = lambda x: x ** 2

# Usar con sorted
words = ["Python", "is", "awesome"]
by_length = sorted(words, key=lambda w: len(w))
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Función para ML

Implementa una función útil en Machine Learning:

```python
def calculate_accuracy(y_true: list, y_pred: list) -> float:
    """Calculate classification accuracy."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)
```

**Descomenta** la sección del Paso 6.

---

## ▶️ Ejecución

```bash
cd bootcamp/week-02/2-practicas/ejercicio-01-funciones
python starter/main.py
```

---

## ✅ Resultado Esperado

```
--- Paso 1: Función Básica ---
Hello, Ana!
Hello, World!

--- Paso 2: Parámetros por Defecto ---
5^2 = 25
2^10 = 1024
3^3 = 27

--- Paso 3: Retorno Múltiple ---
Números: [10, 20, 30, 40, 50]
Min: 10, Max: 50, Avg: 30.0

--- Paso 4: *args y **kwargs ---
sum_all(1, 2, 3) = 6
sum_all(1, 2, 3, 4, 5) = 15
name: Ana
age: 25
city: Madrid

--- Paso 5: Funciones Lambda ---
square(5) = 25
Ordenadas por longitud: ['is', 'Python', 'awesome']

--- Paso 6: Función para ML ---
Accuracy: 83.33%
```

---

_Siguiente: [Ejercicio 02 - Listas](../ejercicio-02-listas/)_
