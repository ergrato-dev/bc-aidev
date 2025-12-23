# ➕ Ejercicio 02: Operadores

## 🎯 Objetivos

- Dominar operadores aritméticos
- Aplicar operadores de comparación
- Usar operadores lógicos
- Entender precedencia de operadores

---

## 📋 Instrucciones

Abre el archivo `starter/main.py` y sigue los pasos descomentando el código indicado.

---

### Paso 1: Operadores Aritméticos Básicos

Python soporta las operaciones matemáticas estándar:

```python
suma = 10 + 5       # 15
resta = 10 - 5      # 5
multiplicacion = 10 * 5  # 50
division = 10 / 5   # 2.0 (siempre retorna float)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: División Entera, Módulo y Potencia

Operadores especiales muy útiles en programación:

```python
division_entera = 10 // 3   # 3 (trunca decimales)
modulo = 10 % 3             # 1 (resto de división)
potencia = 2 ** 3           # 8 (2 elevado a 3)
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Precedencia de Operadores

Python sigue el orden matemático estándar (PEMDAS):

```python
# Sin paréntesis: multiplicación primero
resultado = 2 + 3 * 4   # 14, no 20

# Con paréntesis: suma primero
resultado = (2 + 3) * 4  # 20
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Operadores de Comparación

Comparan valores y retornan `True` o `False`:

```python
print(5 == 5)   # True (igual a)
print(5 != 3)   # True (diferente de)
print(5 > 3)    # True (mayor que)
print(5 < 3)    # False (menor que)
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Operadores Lógicos

Combinan expresiones booleanas:

```python
print(True and False)  # False (ambos deben ser True)
print(True or False)   # True (al menos uno True)
print(not True)        # False (invierte)
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Operadores de Asignación Compuesta

Atajos para modificar y asignar en una operación:

```python
x = 10
x += 5   # x = x + 5 = 15
x -= 3   # x = x - 3 = 12
x *= 2   # x = x * 2 = 24
```

**Descomenta** la sección del Paso 6.

---

### Paso 7: Operadores de Identidad y Membresía

Verifican identidad y pertenencia:

```python
# Identidad
x is None       # True si x es None

# Membresía
5 in [1, 2, 3, 4, 5]  # True
```

**Descomenta** la sección del Paso 7.

---

## ✅ Verificación

Al finalizar, tu programa debe mostrar resultados para cada operación.

---

## 📚 Recursos

- [Python Docs - Expressions](https://docs.python.org/3/reference/expressions.html)

---

_Anterior: [Ejercicio 01](../ejercicio-01-variables/) | Siguiente: [Ejercicio 03](../ejercicio-03-control-flujo/)_
