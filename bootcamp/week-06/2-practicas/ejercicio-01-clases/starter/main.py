"""
Ejercicio 01: Clases Básicas
============================
Aprende a definir clases, crear instancias y trabajar con atributos y métodos.

Instrucciones:
1. Lee cada sección
2. Descomenta el código
3. Ejecuta y observa los resultados
"""

# ============================================
# PASO 1: Definir una Clase Básica
# ============================================
print('--- Paso 1: Definir Clase ---')

# Una clase es una plantilla para crear objetos.
# __init__ es el constructor que se ejecuta al crear una instancia.
# self es una referencia al objeto actual.

# Descomenta las siguientes líneas:
# class Dog:
#     """Represents a dog."""
#
#     def __init__(self, name: str, age: int):
#         """Initialize a new Dog instance."""
#         self.name = name  # Instance attribute
#         self.age = age    # Instance attribute
#
# print("Dog class defined")

print()

# ============================================
# PASO 2: Crear Instancias
# ============================================
print('--- Paso 2: Crear Instancias ---')

# Las instancias son objetos concretos creados a partir de la clase.
# Cada instancia tiene sus propios valores de atributos.

# Descomenta las siguientes líneas:
# fido = Dog("Fido", 3)
# rex = Dog("Rex", 5)
#
# # Acceder a atributos de instancia
# print(f"Name: {fido.name}, Age: {fido.age}")
# print(f"Name: {rex.name}, Age: {rex.age}")
#
# # Cada instancia es un objeto único
# print(f"fido is rex: {fido is rex}")  # False

print()

# ============================================
# PASO 3: Agregar Métodos
# ============================================
print('--- Paso 3: Métodos ---')

# Los métodos son funciones que pertenecen a la clase.
# Siempre reciben self como primer parámetro.

# Descomenta las siguientes líneas:
# class Dog:
#     """Represents a dog with methods."""
#
#     def __init__(self, name: str, age: int):
#         self.name = name
#         self.age = age
#
#     def bark(self) -> str:
#         """Make the dog bark."""
#         return f"{self.name} says Woof!"
#
#     def have_birthday(self) -> None:
#         """Increment the dog's age."""
#         self.age += 1
#
#     def describe(self) -> str:
#         """Return a description of the dog."""
#         return f"{self.name} is a {self.age}-year-old dog"
#
# # Create instance and call methods
# fido = Dog("Fido", 3)
# print(fido.bark())
#
# rex = Dog("Rex", 5)
# rex.have_birthday()
# print(f"Rex is now {rex.age} years old")

print()

# ============================================
# PASO 4: Atributos de Clase
# ============================================
print('--- Paso 4: Atributos de Clase ---')

# Los atributos de clase son compartidos por todas las instancias.
# Se definen directamente en la clase, fuera de __init__.

# Descomenta las siguientes líneas:
# class Dog:
#     """Dog class with class attributes."""
#
#     # Class attributes - shared by all instances
#     species = "Canis familiaris"
#     count = 0
#
#     def __init__(self, name: str, age: int):
#         self.name = name
#         self.age = age
#         # Increment class attribute when creating instance
#         Dog.count += 1
#
#     def bark(self) -> str:
#         return f"{self.name} says Woof!"
#
# # Create instances
# dog1 = Dog("Fido", 3)
# dog2 = Dog("Rex", 5)
#
# # Class attribute is shared
# print(f"Species: {dog1.species}")
# print(f"Total dogs created: {Dog.count}")

print()

# ============================================
# PASO 5: Clase Completa - BankAccount
# ============================================
print('--- Paso 5: BankAccount ---')

# Aplica lo aprendido creando una clase más completa.
# BankAccount tiene métodos para depositar, retirar y ver balance.

# Descomenta las siguientes líneas:
# class BankAccount:
#     """Represents a bank account."""
#
#     def __init__(self, owner: str, balance: float = 0.0):
#         """Initialize account with owner and optional balance."""
#         self.owner = owner
#         self.balance = balance
#
#     def deposit(self, amount: float) -> None:
#         """Add money to the account."""
#         if amount > 0:
#             self.balance += amount
#
#     def withdraw(self, amount: float) -> bool:
#         """Remove money from account. Returns True if successful."""
#         if 0 < amount <= self.balance:
#             self.balance -= amount
#             return True
#         return False
#
#     def get_info(self) -> str:
#         """Return account information."""
#         return f"{self.owner}: ${self.balance:.2f}"
#
# # Create account
# account = BankAccount("Alice", 1000.0)
# print(account.get_info())
#
# # Deposit money
# account.deposit(500)
# print(f"After deposit: ${account.balance:.2f}")
#
# # Withdraw money
# success = account.withdraw(200)
# print(f"Withdrawal successful: {success}")
# print(f"Final balance: ${account.balance:.2f}")
#
# # Try to withdraw too much
# success = account.withdraw(5000)
# print(f"Withdrawal failed: {success}")

print()

# ============================================
# 🎯 DESAFÍO EXTRA (Opcional)
# ============================================
# Crea una clase Product con:
# - Atributos: name, price, quantity
# - Método: total_value() que retorna price * quantity
# - Método: sell(amount) que reduce quantity si hay suficiente stock
#
# Descomenta y completa:
# class Product:
#     def __init__(self, name: str, price: float, quantity: int):
#         pass  # Tu código aquí
#
#     def total_value(self) -> float:
#         pass  # Tu código aquí
#
#     def sell(self, amount: int) -> bool:
#         pass  # Tu código aquí
