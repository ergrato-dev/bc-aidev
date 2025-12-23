# 📖 Glosario - Semana 03

Términos técnicos clave de OOP en Python, ordenados alfabéticamente.

---

## A

### ABC (Abstract Base Class)

Clase que no puede ser instanciada directamente. Define una interfaz que las subclases deben implementar.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass
```

### Abstracción

Ocultar detalles complejos y mostrar solo la funcionalidad esencial. Se logra con clases abstractas e interfaces.

### Atributo

Variable que pertenece a un objeto o clase. Puede ser de instancia (único por objeto) o de clase (compartido).

```python
class Dog:
    species = "Canis"  # Atributo de clase

    def __init__(self, name):
        self.name = name  # Atributo de instancia
```

### Atributo de Clase

Variable definida en el cuerpo de la clase, compartida por todas las instancias.

### Atributo de Instancia

Variable definida en `__init__` con `self`, única para cada objeto.

---

## C

### Clase

Plantilla o "molde" para crear objetos. Define atributos y métodos.

```python
class MyClass:
    pass
```

### @classmethod

Decorador que define un método que recibe la clase (`cls`) como primer argumento en lugar de la instancia.

```python
@classmethod
def from_string(cls, data: str):
    return cls(data)
```

### Composición

Relación "tiene un" (has-a). Un objeto contiene otros objetos como atributos.

```python
class Car:
    def __init__(self):
        self.engine = Engine()  # Composición
```

### Constructor

Método especial `__init__` que inicializa un objeto al crearlo.

---

## D

### Dataclass

Decorador que genera automáticamente `__init__`, `__repr__`, `__eq__` y más.

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
```

### Duck Typing

Filosofía de Python: "Si camina como pato y grazna como pato, es un pato". No importa el tipo, solo que tenga los métodos necesarios.

### Dunder Method

Método con doble guion bajo (`__nombre__`). También llamado "magic method" o método especial.

---

## E

### Encapsulamiento

Ocultar detalles internos de un objeto y exponer solo lo necesario. En Python se usa convención `_` y `__`.

### `__eq__`

Método especial para definir igualdad (`==`) entre objetos.

```python
def __eq__(self, other):
    return self.id == other.id
```

---

## H

### Herencia

Mecanismo donde una clase (hija) adquiere atributos y métodos de otra (padre).

```python
class Child(Parent):
    pass
```

### Herencia Múltiple

Clase que hereda de múltiples clases padre.

```python
class Child(Parent1, Parent2):
    pass
```

---

## I

### `__init__`

Constructor de la clase. Se ejecuta automáticamente al crear un objeto.

```python
def __init__(self, name):
    self.name = name
```

### Instancia

Objeto concreto creado a partir de una clase.

```python
obj = MyClass()  # obj es una instancia
```

### isinstance()

Función para verificar si un objeto es instancia de una clase.

```python
isinstance(obj, MyClass)  # True/False
```

### issubclass()

Función para verificar si una clase hereda de otra.

```python
issubclass(Child, Parent)  # True/False
```

---

## M

### Método

Función definida dentro de una clase. Puede ser de instancia, de clase o estático.

### Método de Instancia

Método que recibe `self` y opera sobre la instancia.

```python
def method(self):
    return self.value
```

### MRO (Method Resolution Order)

Orden en que Python busca métodos en herencia múltiple. Se puede ver con `Class.__mro__`.

---

## N

### Name Mangling

Transformación de `__atributo` a `_Clase__atributo` para evitar colisiones en herencia.

---

## O

### Objeto

Instancia de una clase. Combina datos (atributos) y comportamiento (métodos).

### Override (Sobrescritura)

Redefinir un método heredado en la clase hija.

```python
class Child(Parent):
    def method(self):  # Override
        return "new behavior"
```

---

## P

### Polimorfismo

Capacidad de diferentes clases de responder al mismo método de formas distintas.

### @property

Decorador que permite acceder a un método como si fuera un atributo.

```python
@property
def name(self):
    return self._name
```

### Protocol

Forma de definir interfaces estructurales sin herencia (Python 3.8+).

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...
```

---

## R

### `__repr__`

Método especial que retorna representación para desarrolladores/debugging.

```python
def __repr__(self):
    return f"MyClass(value={self.value})"
```

---

## S

### self

Referencia al objeto actual. Primer parámetro de métodos de instancia.

### Setter

Método que establece el valor de un atributo, típicamente con validación.

```python
@name.setter
def name(self, value):
    if not value:
        raise ValueError("Empty name")
    self._name = value
```

### @staticmethod

Decorador para métodos que no necesitan acceso a la instancia ni a la clase.

```python
@staticmethod
def utility_function():
    return "result"
```

### `__str__`

Método especial que retorna representación legible para usuarios.

```python
def __str__(self):
    return f"Name: {self.name}"
```

### super()

Función para acceder a métodos de la clase padre.

```python
super().__init__(name)
```

---

## T

### Type Hint

Anotación opcional que indica tipos esperados.

```python
def method(self, value: int) -> str:
```

---

## Tabla de Dunder Methods

| Método         | Uso            | Ejemplo          |
| -------------- | -------------- | ---------------- |
| `__init__`     | Constructor    | `obj = Class()`  |
| `__str__`      | String legible | `print(obj)`     |
| `__repr__`     | Representación | `repr(obj)`      |
| `__eq__`       | Igualdad       | `obj1 == obj2`   |
| `__lt__`       | Menor que      | `obj1 < obj2`    |
| `__len__`      | Longitud       | `len(obj)`       |
| `__getitem__`  | Índice         | `obj[key]`       |
| `__setitem__`  | Asignar        | `obj[key] = val` |
| `__contains__` | Membresía      | `x in obj`       |
| `__iter__`     | Iteración      | `for x in obj`   |
| `__call__`     | Llamable       | `obj()`          |
| `__hash__`     | Hash           | `hash(obj)`      |

---

## Principios SOLID

| Principio                 | Descripción                                    |
| ------------------------- | ---------------------------------------------- |
| **S**ingle Responsibility | Una clase, una responsabilidad                 |
| **O**pen/Closed           | Abierto a extensión, cerrado a modificación    |
| **L**iskov Substitution   | Subclases intercambiables con padres           |
| **I**nterface Segregation | Interfaces pequeñas y específicas              |
| **D**ependency Inversion  | Depender de abstracciones, no implementaciones |

---

_Volver a: [Semana 03](../README.md)_
