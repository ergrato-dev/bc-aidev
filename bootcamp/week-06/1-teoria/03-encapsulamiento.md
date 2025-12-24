# 📘 Encapsulamiento en Python

## 🎯 Objetivos

- Comprender el concepto de encapsulamiento
- Usar convenciones de privacidad (`_` y `__`)
- Implementar propiedades con `@property`
- Crear getters y setters con validación

---

## 📋 Contenido

1. [¿Qué es el Encapsulamiento?](#1-qué-es-el-encapsulamiento)
2. [Convenciones de Privacidad](#2-convenciones-de-privacidad)
3. [Propiedades con `@property`](#3-propiedades-con-property)
4. [Setters y Deleters](#4-setters-y-deleters)
5. [Validación en Propiedades](#5-validación-en-propiedades)
6. [Name Mangling](#6-name-mangling)

---

## 1. ¿Qué es el Encapsulamiento?

El **encapsulamiento** es el principio de ocultar los detalles internos de un objeto y exponer solo lo necesario a través de una interfaz pública.

### Beneficios

| Beneficio          | Descripción                                 |
| ------------------ | ------------------------------------------- |
| **Protección**     | Evita modificaciones accidentales           |
| **Validación**     | Controla cómo se modifican los datos        |
| **Flexibilidad**   | Cambiar implementación sin afectar usuarios |
| **Mantenibilidad** | Código más fácil de mantener                |

![Encapsulamiento con @property](../0-assets/04-encapsulamiento-property.svg)

### Ejemplo: Sin Encapsulamiento (Problema)

```python
# ❌ Sin encapsulamiento - cualquiera puede modificar
class BankAccount:
    def __init__(self, balance: float):
        self.balance = balance

account = BankAccount(1000)
account.balance = -5000  # ¡Peligro! Balance negativo
print(account.balance)   # -5000
```

### Ejemplo: Con Encapsulamiento (Solución)

```python
# ✅ Con encapsulamiento - acceso controlado
class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance

    @property
    def balance(self) -> float:
        return self._balance

    def deposit(self, amount: float) -> None:
        if amount > 0:
            self._balance += amount

    def withdraw(self, amount: float) -> bool:
        if 0 < amount <= self._balance:
            self._balance -= amount
            return True
        return False

account = BankAccount(1000)
account.deposit(500)     # OK
account.withdraw(200)    # OK
# account._balance = -5000  # Posible pero señala "no tocar"
print(account.balance)   # 1300
```

---

## 2. Convenciones de Privacidad

Python no tiene modificadores de acceso estrictos como otros lenguajes. Usa **convenciones** basadas en underscores.

### Tipos de Atributos

| Prefijo    | Tipo         | Acceso        | Uso                          |
| ---------- | ------------ | ------------- | ---------------------------- |
| `name`     | Público      | Libre         | API pública                  |
| `_name`    | Protegido    | Convención    | Uso interno (no privado)     |
| `__name`   | Privado      | Name mangling | Evitar colisiones            |
| `__name__` | Dunder/Magic | Sistema       | Métodos especiales de Python |

### Ejemplo Completo

```python
class Example:
    def __init__(self):
        self.public = "Anyone can access"
        self._protected = "Internal use, but accessible"
        self.__private = "Name mangled"

    def __str__(self):  # Dunder method
        return "Example instance"

obj = Example()

# Public - normal access
print(obj.public)         # Anyone can access

# Protected - accessible but "don't touch"
print(obj._protected)     # Internal use, but accessible

# Private - name mangled
# print(obj.__private)    # AttributeError!
print(obj._Example__private)  # Name mangled access

# Dunder - system method
print(str(obj))           # Example instance
```

### Cuándo Usar Cada Uno

```python
class User:
    def __init__(self, username: str, password: str):
        self.username = username    # Public: parte de la API
        self._email = None          # Protected: uso interno
        self.__password = password  # Private: sensible

    def set_email(self, email: str) -> None:
        """Public method to set email with validation."""
        if '@' in email:
            self._email = email

    def verify_password(self, password: str) -> bool:
        """Public method - no expone el password."""
        return self.__password == password
```

---

## 3. Propiedades con `@property`

El decorador `@property` convierte un método en un atributo de solo lectura.

### Sintaxis Básica

```python
class Circle:
    def __init__(self, radius: float):
        self._radius = radius

    @property
    def radius(self) -> float:
        """Get the radius."""
        return self._radius

    @property
    def diameter(self) -> float:
        """Computed property."""
        return self._radius * 2

    @property
    def area(self) -> float:
        """Computed property."""
        import math
        return math.pi * self._radius ** 2

circle = Circle(5)
print(circle.radius)    # 5 (acceso como atributo)
print(circle.diameter)  # 10 (computed)
print(circle.area)      # 78.54... (computed)

# circle.radius = 10    # Error! No setter defined
```

### Ventajas de `@property`

```python
# ❌ Sin property - métodos verbose
class TemperatureOld:
    def __init__(self, celsius):
        self._celsius = celsius

    def get_celsius(self):
        return self._celsius

    def get_fahrenheit(self):
        return self._celsius * 9/5 + 32

temp = TemperatureOld(25)
print(temp.get_celsius())     # 25
print(temp.get_fahrenheit())  # 77.0


# ✅ Con property - acceso limpio
class Temperature:
    def __init__(self, celsius: float):
        self._celsius = celsius

    @property
    def celsius(self) -> float:
        return self._celsius

    @property
    def fahrenheit(self) -> float:
        return self._celsius * 9/5 + 32

temp = Temperature(25)
print(temp.celsius)     # 25 (como atributo)
print(temp.fahrenheit)  # 77.0 (como atributo)
```

---

## 4. Setters y Deleters

### Setter con `@name.setter`

```python
class Person:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """Get the name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the name."""
        if not value:
            raise ValueError("Name cannot be empty")
        self._name = value.strip().title()

person = Person("alice")
print(person.name)  # Alice

person.name = "bob smith"
print(person.name)  # Bob Smith

# person.name = ""  # ValueError: Name cannot be empty
```

### Deleter con `@name.deleter`

```python
class CachedData:
    def __init__(self):
        self._cache = None

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = value

    @cache.deleter
    def cache(self):
        print("Clearing cache...")
        self._cache = None

data = CachedData()
data.cache = {"key": "value"}
print(data.cache)  # {'key': 'value'}

del data.cache     # Clearing cache...
print(data.cache)  # None
```

---

## 5. Validación en Propiedades

Las propiedades son ideales para validar datos.

### Ejemplo: Validación Completa

```python
class Product:
    def __init__(self, name: str, price: float, quantity: int = 0):
        self.name = name      # Uses setter
        self.price = price    # Uses setter
        self.quantity = quantity

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not value or not value.strip():
            raise ValueError("Name cannot be empty")
        self._name = value.strip()

    @property
    def price(self) -> float:
        return self._price

    @price.setter
    def price(self, value: float) -> None:
        if value < 0:
            raise ValueError("Price cannot be negative")
        self._price = float(value)

    @property
    def quantity(self) -> int:
        return self._quantity

    @quantity.setter
    def quantity(self, value: int) -> None:
        if value < 0:
            raise ValueError("Quantity cannot be negative")
        self._quantity = int(value)

    @property
    def total_value(self) -> float:
        """Computed property - read only."""
        return self._price * self._quantity


# Usage
laptop = Product("Laptop", 999.99, 10)
print(laptop.total_value)  # 9999.9

laptop.price = 899.99
print(laptop.total_value)  # 8999.9

# Validation in action
try:
    laptop.price = -100  # ValueError
except ValueError as e:
    print(f"Error: {e}")  # Error: Price cannot be negative
```

### Patrón: Validación con Type Hints

```python
from typing import Optional

class User:
    def __init__(self, email: str, age: Optional[int] = None):
        self.email = email
        self.age = age

    @property
    def email(self) -> str:
        return self._email

    @email.setter
    def email(self, value: str) -> None:
        if '@' not in value:
            raise ValueError("Invalid email format")
        self._email = value.lower()

    @property
    def age(self) -> Optional[int]:
        return self._age

    @age.setter
    def age(self, value: Optional[int]) -> None:
        if value is not None:
            if not isinstance(value, int) or value < 0 or value > 150:
                raise ValueError("Age must be between 0 and 150")
        self._age = value
```

---

## 6. Name Mangling

Python transforma atributos con `__` (doble underscore) para evitar colisiones en herencia.

### Cómo Funciona

```python
class Parent:
    def __init__(self):
        self.__secret = "parent secret"

    def reveal(self) -> str:
        return self.__secret

class Child(Parent):
    def __init__(self):
        super().__init__()
        self.__secret = "child secret"  # Different attribute!

    def reveal_child(self) -> str:
        return self.__secret

child = Child()
print(child.reveal())        # parent secret
print(child.reveal_child())  # child secret

# Both exist as different attributes
print(child._Parent__secret)  # parent secret
print(child._Child__secret)   # child secret
```

### Cuándo Usar `__`

```python
# ✅ Usar cuando necesitas evitar colisiones en subclases
class Base:
    def __init__(self):
        self.__id = id(self)  # Unique per instance, not overridable

# ❌ No usar para "ocultar" datos del usuario
class BadExample:
    def __init__(self):
        self.__password = "secret"  # No es seguridad real
```

---

## 🔑 Resumen

```python
class Example:
    def __init__(self, value: int):
        self._value = value  # Protected by convention

    @property
    def value(self) -> int:
        """Getter - read access."""
        return self._value

    @value.setter
    def value(self, new_value: int) -> None:
        """Setter - write access with validation."""
        if new_value < 0:
            raise ValueError("Value must be positive")
        self._value = new_value

    @property
    def doubled(self) -> int:
        """Computed property - read only."""
        return self._value * 2
```

| Concepto       | Uso                                  |
| -------------- | ------------------------------------ |
| `_name`        | Convención: "uso interno"            |
| `__name`       | Name mangling para evitar colisiones |
| `@property`    | Getter como atributo                 |
| `@name.setter` | Setter con validación                |

---

## 📚 Referencias

- [Python @property](https://docs.python.org/3/library/functions.html#property)
- [Descriptor HowTo Guide](https://docs.python.org/3/howto/descriptor.html)
- [Real Python - Properties](https://realpython.com/python-property/)

---

## 🔗 Navegación

| Anterior                     | Índice      | Siguiente                            |
| ---------------------------- | ----------- | ------------------------------------ |
| [← Herencia](02-herencia.md) | [Teoría](.) | [Polimorfismo →](04-polimorfismo.md) |
