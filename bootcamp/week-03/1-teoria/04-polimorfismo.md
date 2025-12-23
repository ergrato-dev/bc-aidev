# 🦆 Polimorfismo y Duck Typing

## 🎯 Objetivos

- Comprender el polimorfismo en Python
- Aplicar duck typing efectivamente
- Conocer protocolos y ABC (Abstract Base Classes)
- Implementar `@classmethod` y `@staticmethod`
- Usar dataclasses para simplificar código

---

## 📋 Contenido

### 1. ¿Qué es el Polimorfismo?

El **polimorfismo** permite que diferentes objetos respondan al mismo método de formas distintas.

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

# Polimorfismo en acción
def make_speak(animal) -> str:
    return animal.speak()

animals = [Dog(), Cat(), Duck()]
for animal in animals:
    print(make_speak(animal))
# Woof!
# Meow!
# Quack!
```

---

### 2. Duck Typing

> "If it walks like a duck and quacks like a duck, then it must be a duck."

Python no verifica el tipo, sino que el objeto tenga el método necesario:

```python
class Dog:
    def speak(self) -> str:
        return "Woof!"

class Robot:
    def speak(self) -> str:
        return "Beep boop!"

class FileReader:
    def speak(self) -> str:
        return "Reading file..."

# No importa QUÉ es, solo que tenga speak()
def make_speak(thing) -> str:
    return thing.speak()

# Todos funcionan porque tienen speak()
print(make_speak(Dog()))        # Woof!
print(make_speak(Robot()))      # Beep boop!
print(make_speak(FileReader())) # Reading file...
```

#### Duck Typing con Protocolos Implícitos

```python
# Cualquier objeto con __iter__ es "iterable"
class Counter:
    def __init__(self, limit: int):
        self.limit = limit
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self.current >= self.limit:
            raise StopIteration
        self.current += 1
        return self.current

# Funciona en for porque tiene __iter__ y __next__
for num in Counter(5):
    print(num)  # 1, 2, 3, 4, 5
```

---

### 3. Protocolos Estructurales (Python 3.8+)

`Protocol` permite definir duck typing con type hints:

```python
from typing import Protocol

class Drawable(Protocol):
    """Any object that can be drawn."""
    def draw(self) -> str: ...

class Circle:
    def draw(self) -> str:
        return "Drawing a circle"

class Square:
    def draw(self) -> str:
        return "Drawing a square"

class Text:
    def draw(self) -> str:
        return "Drawing text"

def render(shape: Drawable) -> str:
    """Accepts any object with draw() method."""
    return shape.draw()

# Todos funcionan sin heredar de Drawable
print(render(Circle()))  # Drawing a circle
print(render(Square()))  # Drawing a square
print(render(Text()))    # Drawing text
```

---

### 4. Abstract Base Classes (ABC)

Para forzar implementación de métodos en subclases:

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
        return f"Area: {self.area():.2f}, Perimeter: {self.perimeter():.2f}"

# shape = Shape()  # TypeError: Can't instantiate abstract class

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

# Uso
rect = Rectangle(4, 5)
circle = Circle(3)

print(rect.describe())    # Area: 20.00, Perimeter: 18.00
print(circle.describe())  # Area: 28.27, Perimeter: 18.85
```

---

### 5. @classmethod y @staticmethod

#### @classmethod

Recibe la clase como primer argumento (`cls`):

```python
class Employee:
    raise_amount = 1.04  # 4% raise

    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary

    def apply_raise(self) -> None:
        self.salary *= self.raise_amount

    @classmethod
    def set_raise_amount(cls, amount: float) -> None:
        """Change raise amount for all employees."""
        cls.raise_amount = amount

    @classmethod
    def from_string(cls, emp_str: str) -> "Employee":
        """Alternative constructor from string."""
        name, salary = emp_str.split("-")
        return cls(name, float(salary))

# Uso de classmethod
Employee.set_raise_amount(1.05)  # Afecta a todos

# Constructor alternativo
emp = Employee.from_string("John-50000")
print(emp.name)    # John
print(emp.salary)  # 50000.0
```

#### @staticmethod

No recibe `self` ni `cls`, es una función normal en el namespace de la clase:

```python
class MathUtils:
    @staticmethod
    def is_even(n: int) -> bool:
        """Check if number is even."""
        return n % 2 == 0

    @staticmethod
    def factorial(n: int) -> int:
        """Calculate factorial."""
        if n < 0:
            raise ValueError("Negative numbers not allowed")
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

# No necesita instancia
print(MathUtils.is_even(4))    # True
print(MathUtils.factorial(5))  # 120

# Pero también funciona con instancia
utils = MathUtils()
print(utils.is_even(3))  # False
```

#### Comparación

| Tipo            | Primer arg | Uso típico                           |
| --------------- | ---------- | ------------------------------------ |
| Método normal   | `self`     | Operar con la instancia              |
| `@classmethod`  | `cls`      | Factory methods, modificar clase     |
| `@staticmethod` | (ninguno)  | Utilidades relacionadas con la clase |

---

### 6. Dataclasses (Python 3.7+)

Simplifican la creación de clases de datos:

```python
from dataclasses import dataclass, field

# Sin dataclass (verboso)
class PersonOld:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

    def __repr__(self):
        return f"PersonOld(name='{self.name}', age={self.age}, email='{self.email}')"

    def __eq__(self, other):
        if not isinstance(other, PersonOld):
            return False
        return self.name == other.name and self.age == other.age and self.email == other.email

# Con dataclass (conciso)
@dataclass
class Person:
    name: str
    age: int
    email: str

# __init__, __repr__, __eq__ se generan automáticamente!
p1 = Person("Alice", 30, "alice@example.com")
p2 = Person("Alice", 30, "alice@example.com")

print(p1)          # Person(name='Alice', age=30, email='alice@example.com')
print(p1 == p2)    # True
```

#### Dataclass con Valores por Defecto

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0  # Valor por defecto
    tags: list[str] = field(default_factory=list)  # Lista vacía por defecto

    @property
    def total_value(self) -> float:
        return self.price * self.quantity

    def add_tag(self, tag: str) -> None:
        if tag not in self.tags:
            self.tags.append(tag)

laptop = Product("Laptop", 999.99, 5)
laptop.add_tag("electronics")
print(laptop)
# Product(name='Laptop', price=999.99, quantity=5, tags=['electronics'])
print(laptop.total_value)  # 4999.95
```

#### Dataclass Inmutable

```python
@dataclass(frozen=True)
class Point:
    x: float
    y: float

p = Point(3, 4)
# p.x = 5  # FrozenInstanceError - inmutable!

# Puede ser key de diccionario
points = {p: "origin"}
```

---

### 7. Ejemplo Completo: Sistema de Pagos

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

# Protocol para cualquier cosa procesable
class Processable(Protocol):
    def process(self) -> str: ...

# ABC para métodos de pago
class PaymentMethod(ABC):
    @abstractmethod
    def pay(self, amount: float) -> str:
        pass

    @abstractmethod
    def validate(self) -> bool:
        pass

@dataclass
class CreditCard(PaymentMethod):
    number: str
    expiry: str
    cvv: str
    holder: str

    def pay(self, amount: float) -> str:
        if not self.validate():
            return "Payment failed: Invalid card"
        return f"Paid ${amount:.2f} with credit card ending in {self.number[-4:]}"

    def validate(self) -> bool:
        return len(self.number) == 16 and len(self.cvv) == 3

@dataclass
class PayPal(PaymentMethod):
    email: str

    def pay(self, amount: float) -> str:
        if not self.validate():
            return "Payment failed: Invalid email"
        return f"Paid ${amount:.2f} via PayPal ({self.email})"

    def validate(self) -> bool:
        return "@" in self.email

@dataclass
class BankTransfer(PaymentMethod):
    account_number: str
    bank_code: str

    def pay(self, amount: float) -> str:
        return f"Transferred ${amount:.2f} to account {self.account_number}"

    def validate(self) -> bool:
        return len(self.account_number) >= 10

class PaymentProcessor:
    """Process any payment method polymorphically."""

    @staticmethod
    def process_payment(method: PaymentMethod, amount: float) -> str:
        if not method.validate():
            return "Payment validation failed"
        return method.pay(amount)

    @classmethod
    def process_multiple(cls, payments: list[tuple[PaymentMethod, float]]) -> list[str]:
        return [cls.process_payment(method, amount) for method, amount in payments]

# Uso polimórfico
card = CreditCard("1234567890123456", "12/25", "123", "John Doe")
paypal = PayPal("john@example.com")
bank = BankTransfer("1234567890123", "SWIFT123")

# Mismo método, diferentes implementaciones
print(PaymentProcessor.process_payment(card, 99.99))
print(PaymentProcessor.process_payment(paypal, 49.99))
print(PaymentProcessor.process_payment(bank, 199.99))

# Procesar múltiples pagos
payments = [(card, 100), (paypal, 50), (bank, 200)]
results = PaymentProcessor.process_multiple(payments)
for result in results:
    print(result)
```

---

## 💡 Buenas Prácticas

1. **Duck typing primero**: No verificar tipos innecesariamente
2. **Protocol para type hints**: Define interfaces sin herencia
3. **ABC para contratos**: Cuando DEBES implementar métodos
4. **Dataclasses para datos**: Reduce boilerplate
5. **@classmethod para factories**: Constructores alternativos
6. **@staticmethod para utilidades**: Funciones relacionadas

---

## ✅ Checklist de Verificación

- [ ] Entiendo polimorfismo y duck typing
- [ ] Puedo usar Protocol para type hints
- [ ] Sé cuándo usar ABC
- [ ] Conozco la diferencia entre @classmethod y @staticmethod
- [ ] Puedo crear dataclasses

---

## 🔗 Recursos

- [Python Docs - ABC](https://docs.python.org/3/library/abc.html)
- [PEP 544 - Protocols](https://peps.python.org/pep-0544/)
- [Dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Real Python - Duck Typing](https://realpython.com/duck-typing-python/)

---

_Anterior: [Encapsulamiento](03-encapsulamiento.md)_
