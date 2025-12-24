"""
Ejercicio 03: Propiedades
=========================
Aprende a usar @property para crear getters y setters con validación.

Instrucciones:
1. Lee cada sección
2. Descomenta el código
3. Ejecuta y observa los resultados
"""

# ============================================
# PASO 1: Problema sin Propiedades
# ============================================
print('--- Paso 1: Problema sin Propiedades ---')

# Sin propiedades, cualquiera puede asignar valores inválidos.
# Esto rompe la integridad de nuestros datos.

# Descomenta las siguientes líneas:
# class PersonBad:
#     """Person class without data validation."""
#
#     def __init__(self, name: str, age: int):
#         self.name = name
#         self.age = age  # No validation!
#
# person = PersonBad("Alice", 25)
# person.age = -5  # This is invalid but allowed!
# print(f"Age: {person.age} (¡Esto no debería ser válido!)")

print()

# ============================================
# PASO 2: Propiedad Básica (Getter)
# ============================================
print('--- Paso 2: Propiedad Básica ---')

# @property convierte un método en un atributo de solo lectura.
# El atributo real se guarda con prefijo _ (convención).

# Descomenta las siguientes líneas:
# class PersonReadOnly:
#     """Person with read-only age property."""
#
#     def __init__(self, name: str, age: int):
#         self.name = name
#         self._age = age  # Private attribute (convention)
#
#     @property
#     def age(self) -> int:
#         """Get the person's age (read-only)."""
#         return self._age
#
# person = PersonReadOnly("Bob", 25)
# print(f"Age (read-only): {person.age}")
#
# # This would raise AttributeError:
# # person.age = 30  # Can't set attribute

print()

# ============================================
# PASO 3: Setter con Validación
# ============================================
print('--- Paso 3: Setter con Validación ---')

# @property.setter permite asignar valores con validación.
# Si el valor es inválido, lanzamos una excepción.

# Descomenta las siguientes líneas:
# class PersonValidated:
#     """Person with validated age property."""
#
#     def __init__(self, name: str, age: int):
#         self.name = name
#         self.age = age  # Uses setter for validation
#
#     @property
#     def age(self) -> int:
#         """Get the person's age."""
#         return self._age
#
#     @age.setter
#     def age(self, value: int) -> None:
#         """Set age with validation."""
#         if not isinstance(value, int):
#             raise TypeError("Age must be an integer")
#         if value < 0:
#             raise ValueError("Age cannot be negative")
#         if value > 150:
#             raise ValueError("Age cannot exceed 150")
#         self._age = value
#
# # Valid age
# person = PersonValidated("Charlie", 25)
# person.age = 30
# print(f"Age set to: {person.age}")
#
# # Invalid age - caught by validation
# try:
#     person.age = -5
# except ValueError as e:
#     print(f"Error caught: {e}")

print()

# ============================================
# PASO 4: Propiedades Calculadas
# ============================================
print('--- Paso 4: Propiedades Calculadas ---')

# Las propiedades pueden calcular valores dinámicamente.
# No necesitan un atributo de respaldo (_attribute).

# Descomenta las siguientes líneas:
# class Person:
#     """Person with computed properties."""
#
#     def __init__(self, first_name: str, last_name: str):
#         self.first_name = first_name
#         self.last_name = last_name
#
#     @property
#     def full_name(self) -> str:
#         """Computed property: combines first and last name."""
#         return f"{self.first_name} {self.last_name}"
#
#     @property
#     def email(self) -> str:
#         """Computed property: generates email from name."""
#         name = self.full_name.lower().replace(" ", ".")
#         return f"{name}@example.com"
#
# person = Person("John", "Doe")
# print(f"Full name: {person.full_name}")
# print(f"Email: {person.email}")
#
# # Change first name - computed properties update automatically
# person.first_name = "Jane"
# # print(f"Updated full name: {person.full_name}")

print()

# ============================================
# PASO 5: Clase Temperature
# ============================================
print('--- Paso 5: Temperature ---')

# Implementa una clase con conversión automática Celsius/Fahrenheit.
# Internamente guarda Celsius, pero permite leer/escribir en ambas escalas.

# Descomenta las siguientes líneas:
# class Temperature:
#     """Temperature class with Celsius and Fahrenheit properties."""
#
#     def __init__(self, celsius: float = 0.0):
#         self.celsius = celsius
#
#     @property
#     def celsius(self) -> float:
#         """Get temperature in Celsius."""
#         return self._celsius
#
#     @celsius.setter
#     def celsius(self, value: float) -> None:
#         """Set temperature in Celsius."""
#         if value < -273.15:
#             raise ValueError("Temperature below absolute zero!")
#         self._celsius = value
#
#     @property
#     def fahrenheit(self) -> float:
#         """Get temperature in Fahrenheit (computed)."""
#         return self._celsius * 9 / 5 + 32
#
#     @fahrenheit.setter
#     def fahrenheit(self, value: float) -> None:
#         """Set temperature in Fahrenheit (converts to Celsius)."""
#         self.celsius = (value - 32) * 5 / 9
#
# # Create temperature
# temp = Temperature(25)
# print(f"Celsius: {temp.celsius}")
# print(f"Fahrenheit: {temp.fahrenheit}")
#
# # Set using Fahrenheit
# temp.fahrenheit = 32  # Freezing point
# print("After setting Fahrenheit to 32:")
# print(f"Celsius: {temp.celsius}")

print()

# ============================================
# 🎯 DESAFÍO EXTRA (Opcional)
# ============================================
# Crea una clase Circle con propiedades:
# - radius: con validación (debe ser positivo)
# - diameter: propiedad calculada (2 * radius)
# - area: propiedad calculada (pi * radius^2)
# - El diameter debe tener setter que actualice radius
#
# import math
#
# class Circle:
#     def __init__(self, radius: float):
#         pass  # Tu código aquí
#
#     @property
#     def radius(self) -> float:
#         pass  # Tu código aquí
#
#     @radius.setter
#     def radius(self, value: float) -> None:
#         pass  # Tu código aquí
