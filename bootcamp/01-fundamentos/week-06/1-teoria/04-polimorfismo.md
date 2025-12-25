# 📘 Polimorfismo en Python

## 🎯 Objetivos

- Comprender el polimorfismo y duck typing
- Implementar métodos especiales (dunder methods)
- Usar clases abstractas con `abc`
- Aplicar polimorfismo en diseños reales

---

## 📋 Contenido

1. [¿Qué es el Polimorfismo?](#1-qué-es-el-polimorfismo)
2. [Duck Typing](#2-duck-typing)
3. [Métodos Especiales (Dunder Methods)](#3-métodos-especiales-dunder-methods)
4. [Operadores Sobrecargados](#4-operadores-sobrecargados)
5. [Clases Abstractas (ABC)](#5-clases-abstractas-abc)
6. [Protocolos (Typing)](#6-protocolos-typing)

---

## 1. ¿Qué es el Polimorfismo?

**Polimorfismo** significa "muchas formas". Permite que diferentes objetos respondan al mismo mensaje de manera diferente.

### Tipos de Polimorfismo

| Tipo            | Descripción              | Python                |
| --------------- | ------------------------ | --------------------- |
| **Ad-hoc**      | Sobrecarga de operadores | `__add__`, `__mul__`  |
| **Paramétrico** | Genéricos                | Type hints, `TypeVar` |
| **Subtipo**     | Herencia                 | Override de métodos   |
| **Duck typing** | Comportamiento, no tipo  | "If it quacks..."     |

![Polimorfismo y Duck Typing](../0-assets/05-polimorfismo-duck.svg)

### Ejemplo Básico

```python
class Dog:
    def speak(self) -> str:
        return "Woof!"

class Cat:
    def speak(self) -> str:
        return "Meow!"

class Duck:
    def speak(self) -> str:
        return "Quack!"

# Polimorfismo: mismo método, diferente comportamiento
def animal_sound(animal) -> str:
    return animal.speak()

# Works with any object that has speak()
animals = [Dog(), Cat(), Duck()]
for animal in animals:
    print(animal_sound(animal))
# Woof!
# Meow!
# Quack!
```

---

## 2. Duck Typing

> "If it walks like a duck and quacks like a duck, then it must be a duck."

Python no verifica tipos en runtime, sino **comportamiento**.

### Ejemplo

```python
class FileWriter:
    def write(self, data: str) -> None:
        print(f"Writing to file: {data}")

class NetworkWriter:
    def write(self, data: str) -> None:
        print(f"Sending over network: {data}")

class ConsoleWriter:
    def write(self, data: str) -> None:
        print(f"Console: {data}")

# Function doesn't care about type, only behavior
def save_data(writer, data: str) -> None:
    writer.write(data)  # Any object with write() works

# All work!
save_data(FileWriter(), "Hello")      # Writing to file: Hello
save_data(NetworkWriter(), "Hello")   # Sending over network: Hello
save_data(ConsoleWriter(), "Hello")   # Console: Hello
```

### Duck Typing con Built-ins

```python
# len() works with anything that has __len__
class Playlist:
    def __init__(self, songs: list[str]):
        self.songs = songs

    def __len__(self) -> int:
        return len(self.songs)

playlist = Playlist(["Song A", "Song B", "Song C"])
print(len(playlist))  # 3 - works because __len__ exists
```

---

## 3. Métodos Especiales (Dunder Methods)

Los **dunder methods** (double underscore) permiten que objetos personalizados se comporten como tipos built-in.

![Métodos Especiales (Dunder Methods)](../0-assets/06-dunder-methods.svg)

### Métodos Más Comunes

| Método        | Propósito      | Ejemplo                  |
| ------------- | -------------- | ------------------------ |
| `__init__`    | Constructor    | `obj = Class()`          |
| `__str__`     | String legible | `str(obj)`, `print(obj)` |
| `__repr__`    | String técnico | `repr(obj)`, debugger    |
| `__len__`     | Longitud       | `len(obj)`               |
| `__eq__`      | Igualdad       | `obj1 == obj2`           |
| `__lt__`      | Menor que      | `obj1 < obj2`            |
| `__hash__`    | Hash           | `hash(obj)`, dict keys   |
| `__bool__`    | Booleano       | `if obj:`                |
| `__getitem__` | Indexación     | `obj[key]`               |
| `__iter__`    | Iteración      | `for x in obj:`          |
| `__call__`    | Callable       | `obj()`                  |

### Ejemplo: `__str__` y `__repr__`

```python
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        """Human-readable string."""
        return f"Point at ({self.x}, {self.y})"

    def __repr__(self) -> str:
        """Technical string for debugging."""
        return f"Point(x={self.x}, y={self.y})"

p = Point(3, 4)
print(str(p))   # Point at (3, 4)
print(repr(p))  # Point(x=3, y=4)
print(p)        # Point at (3, 4) - uses __str__

# In collections, __repr__ is used
points = [Point(1, 2), Point(3, 4)]
print(points)   # [Point(x=1, y=2), Point(x=3, y=4)]
```

### Ejemplo: Comparación

```python
from functools import total_ordering

@total_ordering  # Generates other comparisons from __eq__ and __lt__
class Version:
    def __init__(self, major: int, minor: int, patch: int):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __eq__(self, other: 'Version') -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) == \
               (other.major, other.minor, other.patch)

    def __lt__(self, other: 'Version') -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) < \
               (other.major, other.minor, other.patch)

    def __repr__(self) -> str:
        return f"Version({self.major}.{self.minor}.{self.patch})"

v1 = Version(1, 0, 0)
v2 = Version(2, 0, 0)
v3 = Version(1, 0, 0)

print(v1 == v3)  # True
print(v1 < v2)   # True
print(v2 > v1)   # True (generated by @total_ordering)
print(sorted([v2, v1, v3]))  # [Version(1.0.0), Version(1.0.0), Version(2.0.0)]
```

### Ejemplo: Container

```python
class Inventory:
    def __init__(self):
        self._items: dict[str, int] = {}

    def add(self, item: str, quantity: int = 1) -> None:
        self._items[item] = self._items.get(item, 0) + quantity

    def __len__(self) -> int:
        """Total unique items."""
        return len(self._items)

    def __contains__(self, item: str) -> bool:
        """Check if item exists."""
        return item in self._items

    def __getitem__(self, item: str) -> int:
        """Get quantity of item."""
        return self._items.get(item, 0)

    def __iter__(self):
        """Iterate over items."""
        return iter(self._items.items())

    def __bool__(self) -> bool:
        """True if not empty."""
        return bool(self._items)

inv = Inventory()
inv.add("sword", 1)
inv.add("potion", 5)

print(len(inv))           # 2
print("sword" in inv)     # True
print(inv["potion"])      # 5

for item, qty in inv:
    print(f"{item}: {qty}")
# sword: 1
# potion: 5

if inv:
    print("Inventory has items")
```

---

## 4. Operadores Sobrecargados

### Operadores Aritméticos

| Operador | Método         | Reverso         |
| -------- | -------------- | --------------- |
| `+`      | `__add__`      | `__radd__`      |
| `-`      | `__sub__`      | `__rsub__`      |
| `*`      | `__mul__`      | `__rmul__`      |
| `/`      | `__truediv__`  | `__rtruediv__`  |
| `//`     | `__floordiv__` | `__rfloordiv__` |
| `%`      | `__mod__`      | `__rmod__`      |
| `**`     | `__pow__`      | `__rpow__`      |

### Ejemplo: Vector

```python
from __future__ import annotations
import math

class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: Vector) -> Vector:
        """Vector addition."""
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector) -> Vector:
        """Vector subtraction."""
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vector:
        """Scalar multiplication."""
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vector:
        """Reverse multiplication (scalar * vector)."""
        return self.__mul__(scalar)

    def __abs__(self) -> float:
        """Magnitude of vector."""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def __eq__(self, other: Vector) -> bool:
        return self.x == other.x and self.y == other.y

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"

v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1 + v2)      # Vector(4, 6)
print(v1 - v2)      # Vector(2, 2)
print(v1 * 2)       # Vector(6, 8)
print(3 * v1)       # Vector(9, 12) - uses __rmul__
print(abs(v1))      # 5.0
```

---

## 5. Clases Abstractas (ABC)

Las **Abstract Base Classes** definen interfaces que las subclases deben implementar.

### Módulo `abc`

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract base class for shapes."""

    @abstractmethod
    def area(self) -> float:
        """Calculate area - must be implemented."""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """Calculate perimeter - must be implemented."""
        pass

    def describe(self) -> str:
        """Concrete method - inherited by all."""
        return f"Shape with area {self.area():.2f}"

# Cannot instantiate abstract class
# shape = Shape()  # TypeError!

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        import math
        return math.pi * self.radius ** 2

    def perimeter(self) -> float:
        import math
        return 2 * math.pi * self.radius

# Now we can instantiate
rect = Rectangle(10, 5)
circle = Circle(7)

print(rect.describe())    # Shape with area 50.00
print(circle.describe())  # Shape with area 153.94
```

### Abstract Properties

```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    @property
    @abstractmethod
    def max_speed(self) -> int:
        """Must be implemented as property."""
        pass

    @abstractmethod
    def start(self) -> str:
        pass

class Car(Vehicle):
    @property
    def max_speed(self) -> int:
        return 200

    def start(self) -> str:
        return "Car engine starting..."

car = Car()
print(car.max_speed)  # 200
print(car.start())    # Car engine starting...
```

---

## 6. Protocolos (Typing)

Python 3.8+ introduce **Protocol** para duck typing estático.

### Ejemplo

```python
from typing import Protocol

class Drawable(Protocol):
    """Protocol for drawable objects."""

    def draw(self) -> str:
        ...

class Circle:
    def __init__(self, radius: float):
        self.radius = radius

    def draw(self) -> str:
        return f"Drawing circle with radius {self.radius}"

class Square:
    def __init__(self, side: float):
        self.side = side

    def draw(self) -> str:
        return f"Drawing square with side {self.side}"

# Function accepts anything with draw()
def render(shape: Drawable) -> None:
    print(shape.draw())

# No inheritance needed!
render(Circle(5))   # Drawing circle with radius 5
render(Square(10))  # Drawing square with side 10
```

---

## 🔑 Resumen

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self) -> str:
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__} says: {self.speak()}"

    def __eq__(self, other) -> bool:
        return type(self) == type(other)

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return "Meow!"

# Polymorphism in action
animals = [Dog(), Cat()]
for animal in animals:
    print(animal)  # Uses __str__ which calls speak()
```

| Concepto       | Descripción                          |
| -------------- | ------------------------------------ |
| Duck typing    | Comportamiento > tipo                |
| Dunder methods | `__str__`, `__eq__`, `__len__`, etc. |
| ABC            | Interfaces con métodos obligatorios  |
| Protocol       | Duck typing con type hints           |

---

## 📚 Referencias

- [Python Data Model](https://docs.python.org/3/reference/datamodel.html)
- [ABC Module](https://docs.python.org/3/library/abc.html)
- [Typing Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol)

---

## 🔗 Navegación

| Anterior                                   | Índice      | Siguiente                               |
| ------------------------------------------ | ----------- | --------------------------------------- |
| [← Encapsulamiento](03-encapsulamiento.md) | [Teoría](.) | [Prácticas →](../2-practicas/README.md) |
