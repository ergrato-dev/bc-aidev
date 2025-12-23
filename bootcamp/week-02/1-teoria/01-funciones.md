# 🔧 Funciones en Python

## 🎯 Objetivos

- Comprender qué son las funciones y por qué son esenciales
- Definir funciones con parámetros y valores de retorno
- Usar argumentos posicionales, keyword y valores por defecto
- Entender el scope (ámbito) de las variables
- Aplicar \*args y \*\*kwargs

---

## 📋 Contenido

![Anatomía de una Función](../0-assets/01-anatomia-funcion.svg)

### 1. ¿Qué es una Función?

Una función es un **bloque de código reutilizable** que realiza una tarea específica.

```python
# Definición de función
def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"

# Llamada a la función
message = greet("Ana")
print(message)  # Hello, Ana!
```

#### ¿Por qué usar funciones?

| Beneficio         | Descripción                       |
| ----------------- | --------------------------------- |
| **Reutilización** | Escribe una vez, usa muchas veces |
| **Organización**  | Código modular y estructurado     |
| **Mantenimiento** | Cambios en un solo lugar          |
| **Testing**       | Fácil de probar individualmente   |
| **Abstracción**   | Oculta complejidad                |

---

### 2. Anatomía de una Función

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Docstring: Descripción de la función.

    Args:
        param1: Descripción del parámetro 1
        param2: Descripción del parámetro 2

    Returns:
        Descripción del valor retornado
    """
    # Cuerpo de la función
    result = param1 + param2
    return result
```

#### Componentes

| Componente       | Descripción                      |
| ---------------- | -------------------------------- |
| `def`            | Keyword para definir función     |
| `function_name`  | Nombre en snake_case             |
| `parameters`     | Valores de entrada               |
| `-> return_type` | Type hint del retorno (opcional) |
| `docstring`      | Documentación de la función      |
| `return`         | Valor que devuelve la función    |

---

### 3. Parámetros y Argumentos

#### Parámetros Posicionales

```python
def calculate_area(width: float, height: float) -> float:
    """Calculate rectangle area."""
    return width * height

# Orden importa
area = calculate_area(10, 5)  # width=10, height=5
print(area)  # 50
```

#### Argumentos Keyword (nombrados)

```python
# Orden no importa con keywords
area = calculate_area(height=5, width=10)  # Mismo resultado
print(area)  # 50
```

#### Valores por Defecto

```python
def greet(name: str, greeting: str = "Hello") -> str:
    """Greet with customizable message."""
    return f"{greeting}, {name}!"

print(greet("Ana"))              # Hello, Ana!
print(greet("Ana", "Hi"))        # Hi, Ana!
print(greet("Ana", greeting="Hola"))  # Hola, Ana!
```

> ⚠️ **Importante**: Los parámetros con valores por defecto van al final.

```python
# ✅ CORRECTO
def func(required, optional="default"):
    pass

# ❌ INCORRECTO - SyntaxError
def func(optional="default", required):
    pass
```

---

### 4. Return: Valores de Retorno

#### Retorno Simple

```python
def square(x: int) -> int:
    """Return the square of x."""
    return x ** 2

result = square(5)
print(result)  # 25
```

#### Retorno Múltiple (Tupla)

```python
def get_min_max(numbers: list) -> tuple:
    """Return min and max values."""
    return min(numbers), max(numbers)

minimum, maximum = get_min_max([3, 1, 4, 1, 5, 9])
print(f"Min: {minimum}, Max: {maximum}")  # Min: 1, Max: 9
```

#### Sin Return (None)

```python
def print_message(msg: str) -> None:
    """Print a message (no return value)."""
    print(msg)
    # return None implícito

result = print_message("Hello")
print(result)  # None
```

---

### 5. Scope (Ámbito de Variables)

#### Variables Locales vs Globales

```python
global_var = "I'm global"

def my_function():
    local_var = "I'm local"
    print(global_var)   # ✅ Puede leer global
    print(local_var)    # ✅ Puede leer local

my_function()
print(global_var)       # ✅ Accesible
# print(local_var)      # ❌ NameError: local_var no existe aquí
```

#### Regla LEGB

Python busca variables en este orden:

| Scope         | Descripción                 | Ejemplo                           |
| ------------- | --------------------------- | --------------------------------- |
| **L**ocal     | Dentro de la función actual | Variables definidas en la función |
| **E**nclosing | Funciones contenedoras      | En funciones anidadas             |
| **G**lobal    | Nivel del módulo            | Variables del archivo             |
| **B**uilt-in  | Python interno              | `print`, `len`, `range`           |

```python
x = "global"

def outer():
    x = "enclosing"

    def inner():
        x = "local"
        print(x)  # local

    inner()
    print(x)  # enclosing

outer()
print(x)  # global
```

#### Modificar Variables Globales

```python
counter = 0

def increment():
    global counter  # Declarar que usamos la global
    counter += 1

increment()
increment()
print(counter)  # 2
```

> 💡 **Consejo**: Evita usar `global`. Pasa valores como parámetros y usa return.

---

### 6. \*args y \*\*kwargs

#### \*args: Argumentos Posicionales Variables

```python
def sum_all(*args) -> int:
    """Sum any number of arguments."""
    print(f"args es una tupla: {args}")
    return sum(args)

print(sum_all(1, 2, 3))        # 6
print(sum_all(1, 2, 3, 4, 5))  # 15
```

#### \*\*kwargs: Argumentos Keyword Variables

```python
def print_info(**kwargs) -> None:
    """Print key-value pairs."""
    print(f"kwargs es un dict: {kwargs}")
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Ana", age=25, city="Madrid")
# name: Ana
# age: 25
# city: Madrid
```

#### Combinando Todo

```python
def flexible_function(required, *args, default="value", **kwargs):
    """Function with all parameter types."""
    print(f"required: {required}")
    print(f"args: {args}")
    print(f"default: {default}")
    print(f"kwargs: {kwargs}")

flexible_function("must", 1, 2, 3, default="custom", extra="data")
```

> 📌 **Orden de parámetros**: `positional, *args, keyword, **kwargs`

---

### 7. Funciones Lambda

Funciones anónimas de una línea:

```python
# Función tradicional
def square(x):
    return x ** 2

# Lambda equivalente
square_lambda = lambda x: x ** 2

print(square(5))        # 25
print(square_lambda(5)) # 25
```

#### Uso Común: Con map, filter, sorted

```python
numbers = [1, 2, 3, 4, 5]

# map: aplicar función a cada elemento
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# filter: filtrar elementos
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4]

# sorted: ordenar por criterio
words = ["python", "ai", "machine", "learning"]
by_length = sorted(words, key=lambda x: len(x))
print(by_length)  # ['ai', 'python', 'machine', 'learning']
```

---

### 8. Funciones en ML/IA

Las funciones son esenciales en Machine Learning:

```python
def calculate_accuracy(y_true: list, y_pred: list) -> float:
    """
    Calculate classification accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score between 0 and 1
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Lists must have same length")

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)

# Ejemplo
true_labels = [1, 0, 1, 1, 0, 1]
predictions = [1, 0, 0, 1, 0, 1]

accuracy = calculate_accuracy(true_labels, predictions)
print(f"Accuracy: {accuracy:.2%}")  # Accuracy: 83.33%
```

---

## 📚 Recursos Adicionales

- [Python Functions - Real Python](https://realpython.com/defining-your-own-python-function/)
- [\*args and \*\*kwargs - Real Python](https://realpython.com/python-kwargs-and-args/)
- [Python Scope - W3Schools](https://www.w3schools.com/python/python_scope.asp)

---

## ✅ Checklist de Verificación

- [ ] Puedo definir funciones con parámetros
- [ ] Entiendo la diferencia entre args posicionales y keyword
- [ ] Sé usar valores por defecto
- [ ] Comprendo el scope de variables (LEGB)
- [ ] Puedo usar \*args y \*\*kwargs
- [ ] Sé cuándo usar funciones lambda

---

_Siguiente: [02-listas-tuplas.md](02-listas-tuplas.md)_
