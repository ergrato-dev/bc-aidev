"""
Ejercicio 02: Herencia
======================
Aprende a crear jerarquías de clases usando herencia y super().

Instrucciones:
1. Lee cada sección
2. Descomenta el código
3. Ejecuta y observa los resultados
"""

# ============================================
# PASO 1: Herencia Básica
# ============================================
print('--- Paso 1: Herencia Básica ---')

# La clase hija hereda todos los atributos y métodos del padre.
# Sintaxis: class Child(Parent)

# Descomenta las siguientes líneas:
# class Animal:
#     """Base class for all animals."""
#
#     def __init__(self, name: str):
#         self.name = name
#
#     def speak(self) -> str:
#         return "Some generic sound"
#
# class Dog(Animal):
#     """Dog inherits from Animal."""
#     pass  # Inherits everything from Animal
#
# # Create a Dog - it has Animal's __init__ and attributes
# fido = Dog("Fido")
# print(f"{fido.name} (inherited from Animal)")
#
# # Check inheritance
# print(f"Is Dog instance of Animal? {isinstance(fido, Animal)}")

print()

# ============================================
# PASO 2: Usar super()
# ============================================
print('--- Paso 2: Usar super() ---')

# super() permite llamar al constructor del padre.
# Esto es útil para inicializar atributos heredados y agregar nuevos.

# Descomenta las siguientes líneas:
# class Animal:
#     def __init__(self, name: str):
#         self.name = name
#
# class Dog(Animal):
#     def __init__(self, name: str, breed: str):
#         # Call parent's __init__ first
#         super().__init__(name)
#         # Then add child-specific attributes
#         self.breed = breed
#
# # Create Dog with both attributes
# fido = Dog("Fido", "Labrador")
# print(f"{fido.name} is a {fido.breed}")

print()

# ============================================
# PASO 3: Sobrescribir Métodos
# ============================================
print('--- Paso 3: Sobrescribir Métodos ---')

# La clase hija puede redefinir (override) métodos del padre.
# El método de la hija reemplaza completamente al del padre.

# Descomenta las siguientes líneas:
# class Animal:
#     def __init__(self, name: str):
#         self.name = name
#
#     def speak(self) -> str:
#         return "Generic animal sound"
#
# class Dog(Animal):
#     def speak(self) -> str:
#         return "Woof!"  # Overrides Animal.speak()
#
# class Cat(Animal):
#     def speak(self) -> str:
#         return "Meow!"  # Overrides Animal.speak()
#
# # Polymorphism: same method, different behavior
# animal = Animal("Generic")
# dog = Dog("Fido")
# cat = Cat("Whiskers")
#
# print(animal.speak())  # Generic animal sound
# print(dog.speak())     # Woof!
# print(cat.speak())     # Meow!

print()

# ============================================
# PASO 4: Extender Métodos
# ============================================
print('--- Paso 4: Extender Métodos ---')

# Puedes llamar al método del padre y agregar funcionalidad.
# Usa super().method() para obtener el resultado del padre.

# Descomenta las siguientes líneas:
# class Animal:
#     def __init__(self, name: str):
#         self.name = name
#
#     def speak(self) -> str:
#         return "Generic sound"
#
# class Dog(Animal):
#     def __init__(self, name: str, happy: bool = True):
#         super().__init__(name)
#         self.happy = happy
#
#     def speak(self) -> str:
#         # Start with parent's implementation
#         base_sound = "Woof!"
#         # Extend with child-specific behavior
#         if self.happy:
#             return f"{base_sound} (I'm a happy dog!)"
#         return base_sound
#
# dog = Dog("Fido", happy=True)
# print(dog.speak())

print()

# ============================================
# PASO 5: Jerarquía de Vehículos
# ============================================
print('--- Paso 5: Jerarquía de Vehículos ---')

# Crea una jerarquía: Vehicle → Car → ElectricCar
# Cada nivel agrega atributos y extiende métodos.

# Descomenta las siguientes líneas:
# class Vehicle:
#     """Base class for all vehicles."""
#
#     def __init__(self, brand: str, model: str):
#         self.brand = brand
#         self.model = model
#
#     def start(self) -> str:
#         return f"{self.brand} {self.model} starting..."
#
# class Car(Vehicle):
#     """Car extends Vehicle."""
#
#     def __init__(self, brand: str, model: str, doors: int = 4):
#         super().__init__(brand, model)
#         self.doors = doors
#
#     def start(self) -> str:
#         # Extend parent's start method
#         return f"{super().start()} Engine running!"
#
# class ElectricCar(Car):
#     """ElectricCar extends Car."""
#
#     def __init__(self, brand: str, model: str, battery_kwh: int):
#         super().__init__(brand, model, doors=4)
#         self.battery_kwh = battery_kwh
#
#     def start(self) -> str:
#         # Extend Car's start method
#         return f"{super().start()} Battery: {self.battery_kwh} kWh"
#
# # Create vehicles at different levels
# vehicle = Vehicle("Toyota", "Corolla")
# car = Car("Toyota", "Corolla", 4)
# tesla = ElectricCar("Tesla", "Model 3", 75)
#
# print(vehicle.start())
# print(car.start())
# print(tesla.start())

print()

# ============================================
# 🎯 DESAFÍO EXTRA (Opcional)
# ============================================
# Crea una jerarquía de empleados:
# - Employee (base): name, salary, get_info()
# - Manager (extends Employee): department, get_info() incluye department
# - Developer (extends Employee): language, get_info() incluye language
#
# Descomenta y completa:
# class Employee:
#     def __init__(self, name: str, salary: float):
#         pass  # Tu código aquí
#
#     def get_info(self) -> str:
#         pass  # Tu código aquí
#
# class Manager(Employee):
#     def __init__(self, name: str, salary: float, department: str):
#         pass  # Tu código aquí
#
#     def get_info(self) -> str:
#         pass  # Tu código aquí
