# 📖 Glosario - Semana 02

Términos técnicos clave de la semana, ordenados alfabéticamente.

---

## A

### \*args

Sintaxis para aceptar un número variable de argumentos posicionales. Dentro de la función, `args` es una tupla.

```python
def func(*args):
    print(args)  # (1, 2, 3)
```

### Argumento

Valor que se pasa a una función al llamarla. Puede ser posicional o keyword.

---

## C

### Callable

Objeto que puede ser llamado con paréntesis `()`. Incluye funciones, métodos, clases y lambdas.

### Comprehension

Sintaxis concisa para crear colecciones. Hay list, dict y set comprehensions.

```python
[x**2 for x in range(5)]  # List comprehension
```

---

## D

### Defaultdict

Diccionario que proporciona un valor por defecto para keys inexistentes.

```python
from collections import defaultdict
d = defaultdict(int)  # Default: 0
```

### Desempaquetado (Unpacking)

Extraer valores de una estructura de datos en variables individuales.

```python
x, y, z = (1, 2, 3)
```

### Diccionario (dict)

Colección de pares key-value. Keys deben ser hashables. Ordenado desde Python 3.7+.

```python
{"name": "Ana", "age": 25}
```

### Docstring

String de documentación en la primera línea de una función, clase o módulo.

```python
def func():
    """This is a docstring."""
    pass
```

---

## F

### First-Class Function

Concepto donde las funciones pueden ser tratadas como cualquier otro valor: asignadas a variables, pasadas como argumentos, retornadas.

### Frozenset

Versión inmutable de un set. Puede ser key de diccionario.

### Función

Bloque de código reutilizable que realiza una tarea específica.

---

## H

### Hashable

Objeto que tiene un valor hash constante durante su vida. Requerido para ser key de dict o elemento de set. Inmutables son hashables.

---

## I

### Inmutable

Objeto cuyo valor no puede cambiar después de crearlo. Ejemplos: str, tuple, frozenset, int.

### Items

Método de diccionario que retorna pares (key, value).

```python
for k, v in dict.items():
    print(k, v)
```

---

## K

### Key (Clave)

Identificador único en un diccionario que mapea a un valor.

### \*\*kwargs

Sintaxis para aceptar un número variable de argumentos keyword. Dentro de la función, `kwargs` es un diccionario.

```python
def func(**kwargs):
    print(kwargs)  # {'a': 1, 'b': 2}
```

---

## L

### Lambda

Función anónima de una línea.

```python
square = lambda x: x ** 2
```

### LEGB Rule

Orden de búsqueda de variables: Local → Enclosing → Global → Built-in.

### Lista (list)

Colección ordenada y mutable de elementos.

```python
[1, 2, 3, "hello"]
```

---

## M

### Método

Función asociada a un objeto. Se llama con `objeto.metodo()`.

### Módulo

Archivo Python (.py) que contiene código reutilizable.

### Mutable

Objeto cuyo valor puede cambiar después de crearlo. Ejemplos: list, dict, set.

---

## P

### Parámetro

Variable en la definición de una función que recibe un argumento.

### Parámetro por Defecto

Parámetro con valor predefinido si no se proporciona argumento.

```python
def greet(name="World"):
    print(f"Hello, {name}")
```

---

## R

### Return

Palabra clave para devolver un valor desde una función. Termina la ejecución de la función.

---

## S

### Scope (Ámbito)

Región del código donde una variable es accesible.

### Set

Colección no ordenada de elementos únicos.

```python
{1, 2, 3}  # Set literal
```

### Slicing

Técnica para extraer una porción de una secuencia.

```python
lista[start:stop:step]
```

---

## T

### Tupla (tuple)

Colección ordenada e inmutable de elementos.

```python
(1, 2, 3)
```

### Type Hint

Anotación opcional que indica el tipo esperado.

```python
def func(x: int) -> str:
```

---

## U

### Union (Unión)

Operación de sets que combina elementos de ambos conjuntos.

```python
A | B  # o A.union(B)
```

---

## V

### Value (Valor)

Dato asociado a una key en un diccionario.

---

## Operaciones de Sets

| Operación            | Símbolo | Método                    | Resultado                    |
| -------------------- | ------- | ------------------------- | ---------------------------- |
| Unión                | `\|`    | `.union()`                | Elementos en A o B           |
| Intersección         | `&`     | `.intersection()`         | Elementos en A y B           |
| Diferencia           | `-`     | `.difference()`           | Elementos en A, no en B      |
| Diferencia Simétrica | `^`     | `.symmetric_difference()` | Elementos en A o B, no ambos |

---

## Métodos Comunes

### Lista

| Método         | Descripción                 |
| -------------- | --------------------------- |
| `append(x)`    | Agregar al final            |
| `insert(i, x)` | Insertar en posición        |
| `remove(x)`    | Eliminar primera ocurrencia |
| `pop(i)`       | Eliminar y retornar índice  |
| `sort()`       | Ordenar in-place            |
| `reverse()`    | Reversar in-place           |

### Diccionario

| Método            | Descripción                   |
| ----------------- | ----------------------------- |
| `get(k, default)` | Obtener con valor por defecto |
| `keys()`          | Obtener keys                  |
| `values()`        | Obtener valores               |
| `items()`         | Obtener pares (k, v)          |
| `pop(k)`          | Eliminar y retornar           |
| `update(d)`       | Fusionar diccionarios         |

---

_Volver a: [Semana 02](../README.md)_
