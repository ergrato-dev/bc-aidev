"""
Ejercicio 01: Clases Básicas
============================
Aprende a crear clases, atributos y métodos en Python.

Instrucciones:
1. Lee cada paso en el README.md
2. Descomenta el código correspondiente
3. Ejecuta para ver los resultados
"""

# ============================================
# PASO 1: Clase Básica
# ============================================
print('=== Paso 1: Clase Básica ===')

# Una clase vacía - la forma más simple
# Descomenta las siguientes líneas:

# class Dog:
#     pass
#
# # Crear instancias (objetos)
# dog1 = Dog()
# dog2 = Dog()
#
# print(type(dog1))      # <class '__main__.Dog'>
# print(dog1 == dog2)    # False - son objetos diferentes

print()

# ============================================
# PASO 2: Constructor __init__
# ============================================
print('=== Paso 2: Constructor __init__ ===')

# __init__ se ejecuta al crear un objeto
# self es la referencia al objeto actual
# Descomenta las siguientes líneas:

# class Dog:
#     def __init__(self, name: str, age: int):
#         self.name = name  # Atributo de instancia
#         self.age = age    # Atributo de instancia
#
# # Crear objetos con datos
# buddy = Dog("Buddy", 3)
# max_dog = Dog("Max", 5)
#
# print(buddy.name)   # Buddy
# print(buddy.age)    # 3

print()

# ============================================
# PASO 3: Métodos de Instancia
# ============================================
print('=== Paso 3: Métodos de Instancia ===')

# Los métodos definen comportamiento
# Siempre reciben self como primer parámetro
# Descomenta las siguientes líneas:

# class Dog:
#     def __init__(self, name: str, age: int):
#         self.name = name
#         self.age = age
#
#     def bark(self) -> str:
#         """Make the dog bark."""
#         return f"{self.name} says: Woof!"
#
#     def birthday(self) -> None:
#         """Celebrate birthday - increase age."""
#         self.age += 1
#         print(f"Happy birthday {self.name}! Now {self.age} years old.")
#
# buddy = Dog("Buddy", 3)
# print(buddy.bark())    # Buddy says: Woof!
# buddy.birthday()       # Happy birthday Buddy! Now 4 years old.

print()

# ============================================
# PASO 4: Atributos de Clase
# ============================================
print('=== Paso 4: Atributos de Clase ===')

# Atributos de clase son compartidos por todas las instancias
# Se definen fuera de __init__
# Descomenta las siguientes líneas:

# class Dog:
#     # Atributo de CLASE - compartido
#     species = "Canis familiaris"
#     count = 0
#
#     def __init__(self, name: str):
#         self.name = name  # Atributo de INSTANCIA - único
#         Dog.count += 1    # Incrementar contador de clase
#
# # Acceso sin instancia
# print(Dog.species)  # Canis familiaris
#
# # Crear instancias
# dog1 = Dog("Buddy")
# dog2 = Dog("Max")
#
# # Atributo de clase accesible desde instancias
# print(dog1.species)  # Canis familiaris
#
# # Contador compartido
# print(f"Total dogs: {Dog.count}")  # 2

print()

# ============================================
# PASO 5: __str__ y __repr__
# ============================================
print('=== Paso 5: __str__ y __repr__ ===')

# __str__: representación para usuarios (print)
# __repr__: representación para desarrolladores (debugging)
# Descomenta las siguientes líneas:

# class Dog:
#     def __init__(self, name: str, age: int):
#         self.name = name
#         self.age = age
#
#     def __str__(self) -> str:
#         """Human-readable string."""
#         return f"{self.name}, {self.age} years old"
#
#     def __repr__(self) -> str:
#         """Developer string for debugging."""
#         return f"Dog(name='{self.name}', age={self.age})"
#
# max_dog = Dog("Max", 5)
#
# print(max_dog)        # Usa __str__: Max, 5 years old
# print(repr(max_dog))  # Usa __repr__: Dog(name='Max', age=5)
#
# # En listas se usa __repr__
# dogs = [Dog("A", 1), Dog("B", 2)]
# print(dogs)  # [Dog(name='A', age=1), Dog(name='B', age=2)]

print()

# ============================================
# PASO 6: Clase Completa - BankAccount
# ============================================
print('=== Paso 6: BankAccount ===')

# Combina todos los conceptos en una clase útil
# Descomenta las siguientes líneas:

# class BankAccount:
#     """A simple bank account."""
#
#     # Atributo de clase
#     bank_name = "Python Bank"
#
#     def __init__(self, owner: str, balance: float = 0.0):
#         """Initialize account with owner and optional balance."""
#         self.owner = owner
#         self.balance = balance
#
#     def deposit(self, amount: float) -> str:
#         """Deposit money into account."""
#         if amount <= 0:
#             return "Amount must be positive"
#         self.balance += amount
#         return f"Deposited ${amount:.2f}. Balance: ${self.balance:.2f}"
#
#     def withdraw(self, amount: float) -> str:
#         """Withdraw money from account."""
#         if amount <= 0:
#             return "Amount must be positive"
#         if amount > self.balance:
#             return "Insufficient funds"
#         self.balance -= amount
#         return f"Withdrew ${amount:.2f}. Balance: ${self.balance:.2f}"
#
#     def __str__(self) -> str:
#         return f"{self.owner}'s Account: ${self.balance:.2f}"
#
#     def __repr__(self) -> str:
#         return f"BankAccount(owner='{self.owner}', balance={self.balance})"
#
#
# # Crear cuenta
# account = BankAccount("Alice")
#
# # Operaciones
# print(account.deposit(100))   # Deposited $100.00. Balance: $100.00
# print(account.withdraw(30))   # Withdrew $30.00. Balance: $70.00
# print(account.withdraw(100))  # Insufficient funds
#
# # Representación
# print(account)  # Alice's Account: $70.00

print()
print('✅ Ejercicio completado!')
