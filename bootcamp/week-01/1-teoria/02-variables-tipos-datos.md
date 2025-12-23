# 📦 Variables y Tipos de Datos

## 🎯 Objetivos

- Comprender qué son las variables y cómo funcionan en Python
- Conocer los tipos de datos básicos
- Aplicar conversiones de tipo (type casting)
- Usar naming conventions de Python

---

## 📋 Contenido

### 1. ¿Qué es una Variable?

Una variable es un **nombre que referencia un valor** almacenado en memoria.

```python
# Asignación de variable
age = 25
name = "Ana"
is_student = True

# Python es dinámicamente tipado
# No necesitas declarar el tipo
x = 10        # x es int
x = "hello"   # ahora x es str (¡válido en Python!)
```

#### Cómo funciona en memoria

```
Variable          Memoria
─────────         ──────────────
  age    ──────►  │    25      │
  name   ──────►  │   "Ana"    │
  is_student ──►  │   True     │
```

---

### 2. Tipos de Datos Básicos

Python tiene varios tipos de datos fundamentales:

#### 📊 Tipos Numéricos

```python
# int - Números enteros
count = 42
negative = -10
big_number = 1_000_000  # Separadores para legibilidad

# float - Números decimales
price = 19.99
pi = 3.14159
scientific = 2.5e-3  # 0.0025

# Verificar tipo
print(type(count))    # <class 'int'>
print(type(price))    # <class 'float'>
```

#### 📝 Strings (Cadenas de texto)

```python
# str - Cadenas de texto
name = "Python"
message = 'Hello World'
multiline = """
Este es un texto
en múltiples líneas
"""

# f-strings (Python 3.6+) - RECOMENDADO
language = "Python"
version = 3.11
print(f"Usando {language} versión {version}")

# Operaciones con strings
greeting = "Hola"
full_greeting = greeting + " Mundo"  # Concatenación
repeated = greeting * 3               # "HolaHolaHola"
```

#### ✅ Booleanos

```python
# bool - Verdadero o Falso
is_active = True
has_error = False

# Resultados de comparaciones
result = 5 > 3      # True
result = 10 == 20   # False

# Valores "falsy" en Python
# False, 0, 0.0, "", [], {}, None → se evalúan como False
```

#### 🚫 None

```python
# None - Ausencia de valor
result = None

# Útil para inicializar variables
user_input = None

# Verificar None
if user_input is None:
    print("No hay valor")
```

---

### 3. Resumen de Tipos

| Tipo    | Ejemplo      | Descripción       |
| ------- | ------------ | ----------------- |
| `int`   | `42`         | Números enteros   |
| `float` | `3.14`       | Números decimales |
| `str`   | `"hola"`     | Texto             |
| `bool`  | `True/False` | Valores lógicos   |
| `None`  | `None`       | Ausencia de valor |

```python
# Verificar tipos con type()
print(type(42))        # <class 'int'>
print(type(3.14))      # <class 'float'>
print(type("hola"))    # <class 'str'>
print(type(True))      # <class 'bool'>
print(type(None))      # <class 'NoneType'>

# Verificar si es de un tipo específico
print(isinstance(42, int))       # True
print(isinstance("hola", str))   # True
```

---

### 4. Conversión de Tipos (Type Casting)

A veces necesitas convertir entre tipos:

```python
# String a número
age_str = "25"
age_int = int(age_str)      # 25 (int)
price_str = "19.99"
price_float = float(price_str)  # 19.99 (float)

# Número a string
count = 42
count_str = str(count)      # "42"

# Float a int (trunca decimales)
pi = 3.99
pi_int = int(pi)            # 3 (no redondea, trunca)

# Bool conversions
bool(1)       # True
bool(0)       # False
bool("hello") # True
bool("")      # False
```

#### ⚠️ Errores comunes

```python
# Esto causa error
int("hello")    # ValueError: invalid literal

# Esto también
int("3.14")     # ValueError - usa float() primero
float("3.14")   # 3.14 ✓
int(float("3.14"))  # 3 ✓
```

---

### 5. Naming Conventions (Convenciones de Nombres)

Python tiene convenciones estándar definidas en **PEP 8**:

```python
# ✅ CORRECTO - snake_case para variables y funciones
user_name = "Ana"
total_count = 100
is_valid = True

# ✅ CORRECTO - UPPER_SNAKE_CASE para constantes
MAX_RETRIES = 3
API_KEY = "secret123"
PI = 3.14159

# ✅ CORRECTO - PascalCase para clases
class UserProfile:
    pass

# ❌ INCORRECTO - Evitar estos estilos
userName = "Ana"      # camelCase (usar en JavaScript, no Python)
UserName = "Ana"      # PascalCase para variables
user-name = "Ana"     # kebab-case (no válido en Python)
```

#### Reglas de nombres válidos

```python
# ✅ Válido
name = "ok"
_private = "ok"
name2 = "ok"
__dunder__ = "ok"

# ❌ Inválido
2name = "error"     # No puede empezar con número
my-var = "error"    # No puede tener guiones
class = "error"     # No puede ser palabra reservada
```

#### Palabras reservadas (no usar como nombres)

```python
import keyword
print(keyword.kwlist)
# ['False', 'None', 'True', 'and', 'as', 'assert', 'async',
#  'await', 'break', 'class', 'continue', 'def', 'del', 'elif',
#  'else', 'except', 'finally', 'for', 'from', 'global', 'if',
#  'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or',
#  'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
```

---

### 6. Type Hints (Python Moderno)

Desde Python 3.5+, puedes agregar **anotaciones de tipo**:

```python
# Sin type hints
def greet(name):
    return f"Hola, {name}"

# Con type hints (RECOMENDADO)
def greet(name: str) -> str:
    return f"Hola, {name}"

# Variables con type hints
age: int = 25
price: float = 19.99
name: str = "Python"
is_active: bool = True
```

> 💡 Los type hints son **opcionales** y no afectan la ejecución, pero mejoran la documentación y ayudan a los IDEs.

---

### 7. Ejercicio Mental

¿Qué tipo tiene cada variable?

```python
a = 42
b = 42.0
c = "42"
d = True
e = None
f = 3 + 4
g = "3" + "4"
h = 3.0 == 3
```

<details>
<summary>Ver respuestas</summary>

```python
a = 42        # int
b = 42.0      # float
c = "42"      # str
d = True      # bool
e = None      # NoneType
f = 3 + 4     # int (7)
g = "3" + "4" # str ("34")
h = 3.0 == 3  # bool (True)
```

</details>

---

## 📊 Resumen Visual

```
                    TIPOS DE DATOS EN PYTHON
                    ========================

    ┌─────────────────────────────────────────────────────────┐
    │                    NUMÉRICOS                             │
    │   ┌─────────┐    ┌─────────┐    ┌─────────┐            │
    │   │   int   │    │  float  │    │ complex │            │
    │   │   42    │    │  3.14   │    │  2+3j   │            │
    │   └─────────┘    └─────────┘    └─────────┘            │
    └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────┐
    │                    SECUENCIAS                            │
    │   ┌─────────┐    ┌─────────┐    ┌─────────┐            │
    │   │   str   │    │  list   │    │  tuple  │            │
    │   │ "hello" │    │ [1,2,3] │    │ (1,2,3) │            │
    │   └─────────┘    └─────────┘    └─────────┘            │
    └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────┐
    │                    OTROS                                 │
    │   ┌─────────┐    ┌─────────┐    ┌─────────┐            │
    │   │  bool   │    │  None   │    │  dict   │            │
    │   │True/Fals│    │  None   │    │ {k: v}  │            │
    │   └─────────┘    └─────────┘    └─────────┘            │
    └─────────────────────────────────────────────────────────┘
```

---

## ✅ Checklist de Verificación

- [ ] Entiendo qué es una variable
- [ ] Conozco los tipos básicos: int, float, str, bool, None
- [ ] Puedo usar type() para verificar tipos
- [ ] Sé hacer conversiones de tipo (int(), str(), float())
- [ ] Aplico snake_case para variables
- [ ] Entiendo los type hints básicos

---

## 📚 Recursos Adicionales

- [Python Docs - Built-in Types](https://docs.python.org/3/library/stdtypes.html)
- [PEP 8 - Style Guide](https://pep8.org/)
- [Real Python - Variables](https://realpython.com/python-variables/)

---

_Anterior: [01 - Introducción](01-introduccion-python.md) | Siguiente: [03 - Operadores](03-operadores.md)_
