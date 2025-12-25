# 📖 Glosario - Semana 06

## Programación Orientada a Objetos (POO)

---

## A

### Abstract Base Class (ABC)

Clase que no puede ser instanciada directamente y define una interfaz que las clases hijas deben implementar. Se usa el módulo `abc` de Python.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass
```

### Abstracción

Pilar de POO que consiste en ocultar la complejidad interna y exponer solo lo necesario. Se logra mediante interfaces y clases abstractas.

### Atributo

Variable asociada a una clase u objeto. Puede ser de instancia (único por objeto) o de clase (compartido).

```python
class Dog:
    species = "Canis familiaris"  # Class attribute

    def __init__(self, name):
        self.name = name  # Instance attribute
```

### Atributo de Clase

Variable definida directamente en la clase, compartida por todas las instancias.

### Atributo de Instancia

Variable única para cada objeto, normalmente definida en `__init__`.

---

## C

### Clase

Plantilla o blueprint para crear objetos. Define atributos y métodos que tendrán las instancias.

```python
class Person:
    def __init__(self, name: str):
        self.name = name
```

### @classmethod

Decorador que define un método que recibe la clase (`cls`) como primer argumento en lugar de la instancia.

```python
class Date:
    @classmethod
    def from_string(cls, date_str: str):
        return cls(*map(int, date_str.split('-')))
```

### Composición

Técnica de diseño donde una clase contiene instancias de otras clases como atributos. Alternativa a la herencia ("tiene un" vs "es un").

### Constructor

Método especial que inicializa una nueva instancia. En Python es `__init__`.

---

## D

### Decorador

Función que modifica el comportamiento de otra función o método. Ejemplos: `@property`, `@staticmethod`, `@classmethod`.

### Duck Typing

Filosofía de Python: "Si camina como pato y hace cuac como pato, es un pato". No importa el tipo, solo que tenga los métodos requeridos.

### Dunder Method

"Double underscore method" - Métodos especiales de Python rodeados por doble guión bajo: `__init__`, `__str__`, `__eq__`, etc.

---

## E

### Encapsulamiento

Pilar de POO que oculta los detalles internos de una clase. En Python se usa convención de nombres: `_protected`, `__private`.

```python
class Account:
    def __init__(self):
        self._balance = 0  # Protected by convention
        self.__secret = 42  # Name mangling: _Account__secret
```

---

## G

### Getter

Método que retorna el valor de un atributo. En Python se implementa con `@property`.

```python
@property
def name(self) -> str:
    return self._name
```

---

## H

### Herencia

Pilar de POO que permite crear nuevas clases basadas en clases existentes, heredando atributos y métodos.

```python
class Animal:
    pass

class Dog(Animal):  # Dog hereda de Animal
    pass
```

### Herencia Múltiple

Cuando una clase hereda de más de una clase padre.

```python
class FlyingFish(Fish, Bird):
    pass
```

---

## I

### `__init__`

Método constructor que se ejecuta automáticamente al crear una instancia.

```python
def __init__(self, name: str):
    self.name = name
```

### Instancia

Objeto concreto creado a partir de una clase.

```python
fido = Dog("Fido")  # fido es una instancia de Dog
```

### `isinstance()`

Función que verifica si un objeto es instancia de una clase.

```python
isinstance(fido, Dog)  # True
isinstance(fido, Animal)  # True (si Dog hereda de Animal)
```

---

## M

### Método

Función definida dentro de una clase que opera sobre sus instancias.

```python
class Dog:
    def bark(self) -> str:
        return "Woof!"
```

### Método de Clase

Método que recibe la clase como primer argumento (`cls`) en lugar de la instancia.

### Método Estático

Método que no recibe `self` ni `cls`. Es una función regular dentro de la clase.

```python
@staticmethod
def validate_age(age: int) -> bool:
    return 0 <= age <= 150
```

### MRO (Method Resolution Order)

Orden en que Python busca métodos en la jerarquía de herencia. Se consulta con `Class.__mro__` o `Class.mro()`.

---

## N

### Name Mangling

Mecanismo de Python que renombra atributos `__name` a `_ClassName__name` para evitar colisiones en herencia.

---

## O

### Objeto

Instancia de una clase. Tiene estado (atributos) y comportamiento (métodos).

### Override (Sobrescritura)

Redefinir un método heredado en la clase hija para cambiar su comportamiento.

```python
class Dog(Animal):
    def speak(self) -> str:  # Override
        return "Woof!"
```

---

## P

### Polimorfismo

Pilar de POO donde diferentes clases pueden responder al mismo método de diferentes maneras.

```python
for animal in [Dog(), Cat(), Bird()]:
    print(animal.speak())  # Cada uno responde diferente
```

### @property

Decorador que permite acceder a un método como si fuera un atributo.

```python
@property
def full_name(self) -> str:
    return f"{self.first} {self.last}"

# Uso: person.full_name (sin paréntesis)
```

### Private (Privado)

Atributo con prefijo `__` que activa name mangling. Convención de "muy privado".

### Protected (Protegido)

Atributo con prefijo `_`. Convención que indica "uso interno, no tocar desde fuera".

---

## S

### `self`

Referencia a la instancia actual dentro de un método. Siempre es el primer parámetro de métodos de instancia.

### Setter

Método que asigna un valor a un atributo, típicamente con validación.

```python
@name.setter
def name(self, value: str) -> None:
    if not value:
        raise ValueError("Name cannot be empty")
    self._name = value
```

### @staticmethod

Decorador para métodos que no necesitan acceso a la instancia ni a la clase.

### `super()`

Función que retorna un objeto proxy para acceder a métodos de la clase padre.

```python
def __init__(self, name, breed):
    super().__init__(name)  # Llama a Parent.__init__
    self.breed = breed
```

---

## T

### Type Hints

Anotaciones de tipo que documentan qué tipos espera y retorna una función.

```python
def greet(name: str) -> str:
    return f"Hello, {name}"
```

---

## Métodos Especiales (Dunder Methods)

| Método         | Propósito              | Ejemplo de uso  |
| -------------- | ---------------------- | --------------- |
| `__init__`     | Constructor            | `obj = Class()` |
| `__str__`      | Representación legible | `print(obj)`    |
| `__repr__`     | Representación técnica | `repr(obj)`     |
| `__eq__`       | Igualdad               | `obj1 == obj2`  |
| `__lt__`       | Menor que              | `obj1 < obj2`   |
| `__len__`      | Longitud               | `len(obj)`      |
| `__getitem__`  | Indexación             | `obj[0]`        |
| `__iter__`     | Iteración              | `for x in obj`  |
| `__contains__` | Membresía              | `x in obj`      |
| `__add__`      | Suma                   | `obj1 + obj2`   |
| `__call__`     | Llamar como función    | `obj()`         |

---

## 🔗 Navegación

| Anterior                              | Inicio                    | Siguiente                              |
| ------------------------------------- | ------------------------- | -------------------------------------- |
| [← Recursos](../4-recursos/README.md) | [Semana 06](../README.md) | [Semana 07 →](../../week-07/README.md) |
