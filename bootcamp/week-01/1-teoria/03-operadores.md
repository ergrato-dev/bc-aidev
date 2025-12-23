# ➕ Operadores en Python

## 🎯 Objetivos

- Dominar los operadores aritméticos
- Comprender los operadores de comparación
- Aplicar operadores lógicos
- Conocer los operadores de asignación

---

## 📋 Contenido

### 1. Operadores Aritméticos

Realizan operaciones matemáticas básicas:

```python
a = 10
b = 3

# Operaciones básicas
print(a + b)    # 13  - Suma
print(a - b)    # 7   - Resta
print(a * b)    # 30  - Multiplicación
print(a / b)    # 3.333... - División (siempre retorna float)
print(a // b)   # 3   - División entera (floor division)
print(a % b)    # 1   - Módulo (resto de división)
print(a ** b)   # 1000 - Potencia (10³)
```

#### Tabla de Operadores Aritméticos

| Operador | Nombre          | Ejemplo  | Resultado  |
| -------- | --------------- | -------- | ---------- |
| `+`      | Suma            | `5 + 3`  | `8`        |
| `-`      | Resta           | `5 - 3`  | `2`        |
| `*`      | Multiplicación  | `5 * 3`  | `15`       |
| `/`      | División        | `5 / 3`  | `1.666...` |
| `//`     | División entera | `5 // 3` | `1`        |
| `%`      | Módulo          | `5 % 3`  | `2`        |
| `**`     | Potencia        | `5 ** 3` | `125`      |

#### Precedencia de Operadores

```python
# Sin paréntesis: sigue orden matemático
result = 2 + 3 * 4      # 14 (no 20)
result = (2 + 3) * 4    # 20

# Orden de precedencia (mayor a menor):
# 1. ** (potencia)
# 2. *, /, //, % (multiplicación, división)
# 3. +, - (suma, resta)

result = 2 ** 3 * 4 + 5  # = 8 * 4 + 5 = 32 + 5 = 37
```

---

### 2. Operadores de Comparación

Comparan dos valores y retornan `True` o `False`:

```python
x = 10
y = 5

print(x == y)   # False - Igual a
print(x != y)   # True  - Diferente de
print(x > y)    # True  - Mayor que
print(x < y)    # False - Menor que
print(x >= y)   # True  - Mayor o igual
print(x <= y)   # False - Menor o igual
```

#### Tabla de Operadores de Comparación

| Operador | Nombre        | Ejemplo  | Resultado |
| -------- | ------------- | -------- | --------- |
| `==`     | Igual a       | `5 == 5` | `True`    |
| `!=`     | Diferente de  | `5 != 3` | `True`    |
| `>`      | Mayor que     | `5 > 3`  | `True`    |
| `<`      | Menor que     | `5 < 3`  | `False`   |
| `>=`     | Mayor o igual | `5 >= 5` | `True`    |
| `<=`     | Menor o igual | `5 <= 3` | `False`   |

#### Comparación de Strings

```python
# Strings se comparan lexicográficamente (orden alfabético)
print("apple" < "banana")  # True
print("Apple" < "apple")   # True (mayúsculas van primero)
print("10" < "9")          # True (¡cuidado! Es comparación de strings)
print(10 < 9)              # False (comparación numérica)
```

---

### 3. Operadores Lógicos

Combinan expresiones booleanas:

```python
a = True
b = False

print(a and b)   # False - AND: ambos deben ser True
print(a or b)    # True  - OR: al menos uno True
print(not a)     # False - NOT: invierte el valor
```

#### Tabla de Verdad

```
AND (y)                 OR (o)                  NOT (no)
─────────────────       ─────────────────       ──────────────
A     B     A and B     A     B     A or B      A      not A
True  True  True        True  True  True        True   False
True  False False       True  False True        False  True
False True  False       False True  True
False False False       False False False
```

#### Ejemplos Prácticos

```python
age = 25
has_license = True
has_car = False

# Puede manejar si tiene edad Y licencia
can_drive = age >= 18 and has_license
print(can_drive)  # True

# Puede viajar si tiene auto O transporte público
can_travel = has_car or True  # Asumiendo que hay transporte público
print(can_travel)  # True

# No es menor de edad
is_not_minor = not (age < 18)
print(is_not_minor)  # True
```

#### Short-Circuit Evaluation

Python evalúa de izquierda a derecha y se detiene cuando puede:

```python
# Con AND: si el primero es False, no evalúa el segundo
False and print("No se ejecuta")

# Con OR: si el primero es True, no evalúa el segundo
True or print("No se ejecuta")

# Útil para evitar errores
name = None
# Esto evita error si name es None
if name and len(name) > 0:
    print(name)
```

---

### 4. Operadores de Asignación

Asignan y modifican valores:

```python
# Asignación simple
x = 10

# Asignación compuesta (operación + asignación)
x += 5    # x = x + 5  → 15
x -= 3    # x = x - 3  → 12
x *= 2    # x = x * 2  → 24
x /= 4    # x = x / 4  → 6.0
x //= 2   # x = x // 2 → 3.0
x %= 2    # x = x % 2  → 1.0
x **= 3   # x = x ** 3 → 1.0
```

#### Tabla de Operadores de Asignación

| Operador | Ejemplo   | Equivalente  |
| -------- | --------- | ------------ |
| `=`      | `x = 5`   | Asignación   |
| `+=`     | `x += 5`  | `x = x + 5`  |
| `-=`     | `x -= 5`  | `x = x - 5`  |
| `*=`     | `x *= 5`  | `x = x * 5`  |
| `/=`     | `x /= 5`  | `x = x / 5`  |
| `//=`    | `x //= 5` | `x = x // 5` |
| `%=`     | `x %= 5`  | `x = x % 5`  |
| `**=`    | `x **= 5` | `x = x ** 5` |

---

### 5. Operadores de Identidad y Membresía

#### Identidad: `is` / `is not`

Verifican si dos variables apuntan al **mismo objeto** en memoria:

```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)    # True  - Mismo valor
print(a is b)    # False - Diferentes objetos
print(a is c)    # True  - Mismo objeto

# Útil para comparar con None
x = None
print(x is None)      # True (RECOMENDADO)
print(x == None)      # True (pero menos idiomático)
```

#### Membresía: `in` / `not in`

Verifican si un elemento está en una secuencia:

```python
# En listas
numbers = [1, 2, 3, 4, 5]
print(3 in numbers)      # True
print(10 in numbers)     # False
print(10 not in numbers) # True

# En strings
text = "Hello World"
print("World" in text)   # True
print("world" in text)   # False (case sensitive)

# En diccionarios (verifica keys)
data = {"name": "Ana", "age": 25}
print("name" in data)    # True
print("Ana" in data)     # False (Ana es valor, no key)
```

---

### 6. Operadores para IA/ML

En IA usarás frecuentemente estos patrones:

```python
# División entera para batches
total_samples = 1000
batch_size = 32
num_batches = total_samples // batch_size  # 31

# Módulo para índices cíclicos
epoch = 5
log_every = 2
should_log = epoch % log_every == 0  # False (epoch 5 no es divisible por 2)

# Comparaciones encadenadas
accuracy = 0.85
if 0.7 <= accuracy <= 0.9:
    print("Buen modelo")

# Operador ternario
status = "good" if accuracy > 0.8 else "needs improvement"
```

---

### 7. Ejercicio Mental

¿Cuál es el resultado?

```python
a = 10
b = 3

r1 = a + b * 2
r2 = (a + b) * 2
r3 = a / b == a // b
r4 = a % b + a // b
r5 = True and False or True
r6 = not (a > b and b > 0)
```

<details>
<summary>Ver respuestas</summary>

```python
r1 = 10 + 3 * 2 = 10 + 6 = 16
r2 = (10 + 3) * 2 = 13 * 2 = 26
r3 = 3.333... == 3 = False
r4 = 1 + 3 = 4
r5 = False or True = True
r6 = not (True and True) = not True = False
```

</details>

---

## 📊 Resumen Visual

```
            OPERADORES EN PYTHON
            ====================

    ARITMÉTICOS          COMPARACIÓN          LÓGICOS
    ───────────          ───────────          ───────
    +  Suma              ==  Igual            and  Y
    -  Resta             !=  Diferente        or   O
    *  Multiplicación    >   Mayor            not  No
    /  División          <   Menor
    // División entera   >=  Mayor o igual
    %  Módulo            <=  Menor o igual
    ** Potencia

    ASIGNACIÓN           IDENTIDAD            MEMBRESÍA
    ──────────           ─────────            ─────────
    =   Asignar          is     Es mismo      in      Está en
    +=  Sumar y asignar  is not No es mismo   not in  No está
    -=  Restar y asignar
    *=  etc...
```

---

## ✅ Checklist de Verificación

- [ ] Domino los operadores aritméticos (+, -, \*, /, //, %, \*\*)
- [ ] Entiendo la precedencia de operadores
- [ ] Uso correctamente operadores de comparación
- [ ] Aplico operadores lógicos (and, or, not)
- [ ] Conozco los operadores de asignación compuesta
- [ ] Distingo entre `==` e `is`
- [ ] Uso `in` para verificar membresía

---

## 📚 Recursos Adicionales

- [Python Docs - Expressions](https://docs.python.org/3/reference/expressions.html)
- [Real Python - Operators](https://realpython.com/python-operators-expressions/)

---

_Anterior: [02 - Variables](02-variables-tipos-datos.md) | Siguiente: [04 - Estructuras de Control](04-estructuras-control.md)_
