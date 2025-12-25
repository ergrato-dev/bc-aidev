"""
Ejercicio 03: Properties y Encapsulamiento
==========================================
Aprende a usar @property para encapsular y validar datos.

Instrucciones:
1. Lee cada paso en el README.md
2. Descomenta el código correspondiente
3. Ejecuta para ver los resultados
"""

# ============================================
# PASO 1: Atributos Protegidos
# ============================================
print('=== Paso 1: Atributos Protegidos ===')

# El prefijo _ indica "no modificar directamente"
# Es una convención, Python no lo impide
# Descomenta las siguientes líneas:

# class BankAccount:
#     def __init__(self, balance: float):
#         self._balance = balance  # Protegido por convención
#
#     def get_balance(self) -> float:
#         """Getter tradicional."""
#         return self._balance
#
#     def deposit(self, amount: float) -> None:
#         if amount > 0:
#             self._balance += amount
#
#
# account = BankAccount(100)
# print(f"Balance: {account.get_balance()}")  # Forma correcta
#
# # Esto funciona pero NO se recomienda
# # account._balance = 1000000  # Mal estilo!

print()

# ============================================
# PASO 2: Property Básica
# ============================================
print('=== Paso 2: Property Básica ===')

# @property permite acceso como atributo pero con control
# Descomenta las siguientes líneas:

# class BankAccount:
#     def __init__(self, balance: float):
#         self._balance = balance
#
#     @property
#     def balance(self) -> float:
#         """Get the current balance."""
#         return self._balance
#
#     def deposit(self, amount: float) -> None:
#         if amount > 0:
#             self._balance += amount
#
#
# account = BankAccount(100)
#
# # Acceso como atributo (no como método)
# print(f"Balance: {account.balance}")  # No paréntesis!
#
# # Sin setter, no se puede asignar
# # account.balance = 500  # AttributeError!

print()

# ============================================
# PASO 3: Setter con Validación
# ============================================
print('=== Paso 3: Setter con Validación ===')

# El setter valida antes de asignar
# Descomenta las siguientes líneas:

# class Circle:
#     def __init__(self, radius: float):
#         self.radius = radius  # Llama al setter
#
#     @property
#     def radius(self) -> float:
#         """Get the radius."""
#         return self._radius
#
#     @radius.setter
#     def radius(self, value: float) -> None:
#         """Set radius with validation."""
#         if value <= 0:
#             raise ValueError("Radius must be positive")
#         self._radius = value
#
#
# # Crear círculo válido
# circle = Circle(5)
# print(f"Radius: {circle.radius}")
#
# # Cambiar radio (válido)
# circle.radius = 10
# print(f"New radius: {circle.radius}")
#
# # Intentar valor inválido
# try:
#     circle.radius = -5
# except ValueError as e:
#     print(f"Error: {e}")

print()

# ============================================
# PASO 4: Property Read-Only
# ============================================
print('=== Paso 4: Property Read-Only ===')

# Properties sin setter son de solo lectura
# Útil para valores calculados
# Descomenta las siguientes líneas:

# import math
#
# class Circle:
#     def __init__(self, radius: float):
#         self._radius = radius
#
#     @property
#     def radius(self) -> float:
#         return self._radius
#
#     @radius.setter
#     def radius(self, value: float) -> None:
#         if value <= 0:
#             raise ValueError("Radius must be positive")
#         self._radius = value
#
#     @property
#     def area(self) -> float:
#         """Calculated area (read-only)."""
#         return math.pi * self._radius ** 2
#
#     @property
#     def diameter(self) -> float:
#         """Get diameter."""
#         return self._radius * 2
#
#     @diameter.setter
#     def diameter(self, value: float) -> None:
#         """Set diameter (updates radius)."""
#         self.radius = value / 2
#
#
# circle = Circle(5)
# print(f"Radius: {circle.radius}, Area: {circle.area:.2f}")
# print(f"Diameter: {circle.diameter}")
#
# # Intentar modificar area (read-only)
# try:
#     circle.area = 100
# except AttributeError:
#     print("Cannot set area (read-only)")

print()

# ============================================
# PASO 5: User con Validaciones
# ============================================
print('=== Paso 5: User con Validaciones ===')

# Múltiples properties con diferentes validaciones
# Descomenta las siguientes líneas:

# class User:
#     def __init__(self, name: str, email: str, age: int):
#         self.name = name    # Usa setter
#         self.email = email  # Usa setter
#         self.age = age      # Usa setter
#
#     @property
#     def name(self) -> str:
#         return self._name
#
#     @name.setter
#     def name(self, value: str) -> None:
#         if not value or not value.strip():
#             raise ValueError("Name cannot be empty")
#         self._name = value.strip()
#
#     @property
#     def email(self) -> str:
#         return self._email
#
#     @email.setter
#     def email(self, value: str) -> None:
#         if "@" not in value:
#             raise ValueError("Invalid email format")
#         self._email = value.lower()
#
#     @property
#     def age(self) -> int:
#         return self._age
#
#     @age.setter
#     def age(self, value: int) -> None:
#         if not isinstance(value, int) or value < 0 or value > 150:
#             raise ValueError("Age must be between 0 and 150")
#         self._age = value
#
#
# # Crear usuario válido
# user = User("Alice", "Alice@Example.com", 25)
# print(f"Name: {user.name}")
# print(f"Email: {user.email}")  # Normalizado a minúsculas
#
# # Intentar email inválido
# try:
#     user.email = "invalid-email"
# except ValueError as e:
#     print(f"Error: {e}")

print()

# ============================================
# PASO 6: Producto con Stock
# ============================================
print('=== Paso 6: Producto con Stock ===')

# Ejemplo completo con múltiples properties
# Descomenta las siguientes líneas:

# class Product:
#     """Product with price and stock management."""
#
#     def __init__(self, name: str, price: float, stock: int = 0):
#         self._name = name
#         self.price = price  # Usa setter
#         self.stock = stock  # Usa setter
#
#     @property
#     def name(self) -> str:
#         """Product name (read-only)."""
#         return self._name
#
#     @property
#     def price(self) -> float:
#         return self._price
#
#     @price.setter
#     def price(self, value: float) -> None:
#         if value < 0:
#             raise ValueError("Price cannot be negative")
#         self._price = round(value, 2)
#
#     @property
#     def stock(self) -> int:
#         return self._stock
#
#     @stock.setter
#     def stock(self, value: int) -> None:
#         if not isinstance(value, int) or value < 0:
#             raise ValueError("Stock must be non-negative integer")
#         self._stock = value
#
#     @property
#     def value(self) -> float:
#         """Total value of stock (read-only)."""
#         return self._price * self._stock
#
#     @property
#     def is_available(self) -> bool:
#         """Check if in stock."""
#         return self._stock > 0
#
#     def sell(self, quantity: int = 1) -> float:
#         """Sell product and return total."""
#         if quantity > self._stock:
#             raise ValueError(f"Not enough stock. Available: {self._stock}")
#         self.stock -= quantity
#         return self._price * quantity
#
#     def __str__(self) -> str:
#         status = "In Stock" if self.is_available else "Out of Stock"
#         return f"{self._name}: ${self._price:.2f} ({self._stock} units) - {status}"
#
#
# # Crear producto
# laptop = Product("Laptop", 999.99, 10)
# print(laptop)
#
# # Propiedades computadas
# print(f"Total value: ${laptop.value:,.2f}")
#
# # Vender
# total = laptop.sell(3)
# print(f"Sold 3 for ${total:,.2f}")
# print(laptop)

print()
print('✅ Ejercicio completado!')
