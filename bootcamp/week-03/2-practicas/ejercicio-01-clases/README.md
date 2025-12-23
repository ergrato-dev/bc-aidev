# 🏗️ Ejercicio 01: Clases Básicas

## 🎯 Objetivo

Aprender a crear clases con atributos, métodos y métodos especiales.

---

## 📋 Conceptos Cubiertos

- Definir clases con `class`
- El método `__init__` (constructor)
- Atributos de instancia y de clase
- Métodos de instancia
- Métodos especiales: `__str__`, `__repr__`

---

## 🚀 Instrucciones

### Paso 1: Clase Básica

Crea una clase simple sin atributos:

```python
class Dog:
    pass

# Crear instancias
dog1 = Dog()
dog2 = Dog()
print(type(dog1))  # <class '__main__.Dog'>
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Constructor `__init__`

Añade atributos al crear objetos:

```python
class Dog:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

buddy = Dog("Buddy", 3)
print(buddy.name)  # Buddy
print(buddy.age)   # 3
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Métodos de Instancia

Define comportamiento para los objetos:

```python
class Dog:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def bark(self) -> str:
        return f"{self.name} says: Woof!"

    def birthday(self) -> None:
        self.age += 1
        print(f"Happy birthday {self.name}! Now {self.age} years old.")
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Atributos de Clase

Atributos compartidos por todas las instancias:

```python
class Dog:
    species = "Canis familiaris"  # Atributo de clase
    count = 0

    def __init__(self, name: str):
        self.name = name  # Atributo de instancia
        Dog.count += 1

print(Dog.species)  # Acceso sin instancia
print(Dog.count)    # Contador de instancias
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: `__str__` y `__repr__`

Representación legible de objetos:

```python
class Dog:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def __str__(self) -> str:
        """Para usuarios - print()"""
        return f"{self.name}, {self.age} years old"

    def __repr__(self) -> str:
        """Para desarrolladores - debugging"""
        return f"Dog(name='{self.name}', age={self.age})"

dog = Dog("Buddy", 3)
print(dog)        # Buddy, 3 years old
print(repr(dog))  # Dog(name='Buddy', age=3)
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Clase Completa - BankAccount

Combina todo lo aprendido:

```python
class BankAccount:
    bank_name = "Python Bank"

    def __init__(self, owner: str, balance: float = 0.0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> str: ...
    def withdraw(self, amount: float) -> str: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
```

**Descomenta** la sección del Paso 6.

---

## ✅ Resultado Esperado

```
=== Paso 1: Clase Básica ===
<class '__main__.Dog'>
False

=== Paso 2: Constructor __init__ ===
Buddy
3

=== Paso 3: Métodos de Instancia ===
Buddy says: Woof!
Happy birthday Buddy! Now 4 years old.

=== Paso 4: Atributos de Clase ===
Canis familiaris
Total dogs: 2

=== Paso 5: __str__ y __repr__ ===
Max, 5 years old
Dog(name='Max', age=5)

=== Paso 6: BankAccount ===
Deposited $100.00. Balance: $100.00
Withdrew $30.00. Balance: $70.00
Insufficient funds
Alice's Account: $70.00
```

---

## 🔗 Recursos

- [Python Classes](https://docs.python.org/3/tutorial/classes.html)
- [Special Methods](https://docs.python.org/3/reference/datamodel.html#special-method-names)

---

_Siguiente: [Ejercicio 02 - Herencia](../ejercicio-02-herencia/)_
