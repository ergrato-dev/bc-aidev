"""
Ejercicio 02: Herencia
======================
Aprende a implementar herencia y usar super() en Python.

Instrucciones:
1. Lee cada paso en el README.md
2. Descomenta el código correspondiente
3. Ejecuta para ver los resultados
"""

# ============================================
# PASO 1: Herencia Simple
# ============================================
print('=== Paso 1: Herencia Simple ===')

# La clase hija hereda atributos y métodos del padre
# Puede hacer override (sobrescribir) métodos
# Descomenta las siguientes líneas:

# class Animal:
#     def __init__(self, name: str):
#         self.name = name
#
#     def speak(self) -> str:
#         return "Some generic sound"
#
#
# class Dog(Animal):  # Dog hereda de Animal
#     def speak(self) -> str:  # Override del método
#         return f"{self.name} says: Woof!"
#
#
# class Cat(Animal):
#     def speak(self) -> str:
#         return f"{self.name} says: Meow!"
#
#
# # Crear instancias
# buddy = Dog("Buddy")
# whiskers = Cat("Whiskers")
#
# # Ambos tienen .name (heredado) y .speak() (override)
# print(buddy.speak())     # Buddy says: Woof!
# print(whiskers.speak())  # Whiskers says: Meow!

print()

# ============================================
# PASO 2: Usar super()
# ============================================
print('=== Paso 2: Usar super() ===')

# super() permite llamar métodos de la clase padre
# Importante para extender __init__ sin perder atributos
# Descomenta las siguientes líneas:

# class Animal:
#     def __init__(self, name: str, age: int):
#         self.name = name
#         self.age = age
#
#
# class Dog(Animal):
#     def __init__(self, name: str, age: int, breed: str):
#         # Llamar al __init__ del padre
#         super().__init__(name, age)
#         # Añadir atributo propio
#         self.breed = breed
#
#     def __str__(self) -> str:
#         return f"{self.name}, {self.age} years old, {self.breed}"
#
#
# max_dog = Dog("Max", 5, "Golden Retriever")
# print(max_dog)  # Max, 5 years old, Golden Retriever
#
# # Verificar que tiene todos los atributos
# print(f"Name: {max_dog.name}")    # Heredado
# print(f"Age: {max_dog.age}")      # Heredado
# print(f"Breed: {max_dog.breed}")  # Propio

print()

# ============================================
# PASO 3: isinstance e issubclass
# ============================================
print('=== Paso 3: isinstance e issubclass ===')

# isinstance() verifica si un objeto es de cierta clase
# issubclass() verifica si una clase hereda de otra
# Descomenta las siguientes líneas:

# class Animal:
#     pass
#
# class Dog(Animal):
#     pass
#
# class Cat(Animal):
#     pass
#
# buddy = Dog()
#
# # isinstance - verifica objeto
# print(f"Is buddy a Dog? {isinstance(buddy, Dog)}")        # True
# print(f"Is buddy an Animal? {isinstance(buddy, Animal)}") # True
# print(f"Is buddy a Cat? {isinstance(buddy, Cat)}")        # False
#
# # issubclass - verifica clases
# print(f"Is Dog subclass of Animal? {issubclass(Dog, Animal)}")  # True
# print(f"Is Cat subclass of Dog? {issubclass(Cat, Dog)}")        # False

print()

# ============================================
# PASO 4: Extender Métodos
# ============================================
print('=== Paso 4: Extender Métodos ===')

# Puedes llamar al método del padre y añadir funcionalidad
# Descomenta las siguientes líneas:

# class Animal:
#     def __init__(self, name: str):
#         self.name = name
#
#     def describe(self) -> str:
#         return f"I am {self.name}"
#
#
# class Dog(Animal):
#     def describe(self) -> str:
#         # Llamar al método del padre
#         parent_description = super().describe()
#         # Añadir más información
#         return f"{parent_description} and I bark!"
#
#
# rex = Dog("Rex")
# print(rex.describe())  # I am Rex and I bark!

print()

# ============================================
# PASO 5: Herencia Múltiple
# ============================================
print('=== Paso 5: Herencia Múltiple ===')

# Python permite heredar de múltiples clases
# El MRO (Method Resolution Order) define el orden de búsqueda
# Descomenta las siguientes líneas:

# class Flyer:
#     def fly(self) -> str:
#         return "Flying!"
#
#
# class Swimmer:
#     def swim(self) -> str:
#         return "Swimming!"
#
#
# class Duck(Flyer, Swimmer):
#     def quack(self) -> str:
#         return "Quack!"
#
#
# donald = Duck()
#
# # Duck tiene métodos de Flyer Y Swimmer
# print(donald.fly())    # Flying!
# print(donald.swim())   # Swimming!
# print(donald.quack())  # Quack!
#
# # Ver el MRO
# mro = " -> ".join(c.__name__ for c in Duck.__mro__)
# print(f"MRO: {mro}")

print()

# ============================================
# PASO 6: Sistema de Empleados
# ============================================
print('=== Paso 6: Sistema de Empleados ===')

# Ejemplo práctico combinando herencia y super()
# Descomenta las siguientes líneas:

# class Employee:
#     """Base class for all employees."""
#
#     def __init__(self, name: str, salary: float):
#         self.name = name
#         self.salary = salary
#
#     def get_annual_salary(self) -> float:
#         """Calculate annual salary."""
#         return self.salary * 12
#
#     def __str__(self) -> str:
#         return f"{self.name}: ${self.get_annual_salary():,.2f}/year"
#
#
# class Manager(Employee):
#     """Manager with department and bonus."""
#
#     def __init__(self, name: str, salary: float, department: str):
#         super().__init__(name, salary)
#         self.department = department
#
#     def get_annual_salary(self) -> float:
#         """Managers get 20% bonus."""
#         base = super().get_annual_salary()
#         return base * 1.2
#
#     def __str__(self) -> str:
#         return f"{self.name} ({self.department}): ${self.get_annual_salary():,.2f}/year"
#
#
# # Crear empleados
# manager = Manager("Alice", 8000, "Engineering")
# employee = Employee("Bob", 5000)
#
# print(manager)   # Alice (Engineering): $115,200.00/year
# print(employee)  # Bob: $60,000.00/year

print()
print('✅ Ejercicio completado!')
