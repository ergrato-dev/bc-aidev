"""
Ejercicio 04: Sistema Integrador
================================
Combina OOP: clases, herencia, properties, polimorfismo.

Instrucciones:
1. Lee cada paso en el README.md
2. Descomenta el código correspondiente
3. Ejecuta para ver los resultados
"""

from __future__ import annotations
from datetime import date

# ============================================
# PASO 1: Clase Base Vehicle
# ============================================
print('=== Paso 1: Clase Base Vehicle ===')

# Clase base con property computada
# Descomenta las siguientes líneas:

# class Vehicle:
#     """Base class for all vehicles."""
#
#     def __init__(self, brand: str, model: str, year: int):
#         self.brand = brand
#         self.model = model
#         self._year = year
#
#     @property
#     def year(self) -> int:
#         return self._year
#
#     @year.setter
#     def year(self, value: int) -> None:
#         current_year = date.today().year
#         if value < 1886 or value > current_year + 1:
#             raise ValueError(f"Year must be between 1886 and {current_year + 1}")
#         self._year = value
#
#     @property
#     def age(self) -> int:
#         """Calculate vehicle age."""
#         return date.today().year - self._year
#
#     def __str__(self) -> str:
#         return f"{self._year} {self.brand} {self.model}"
#
#
# vehicle = Vehicle("Toyota", "Camry", 2020)
# print(vehicle)
# print(f"Age: {vehicle.age} years")

print()

# ============================================
# PASO 2: Car y Motorcycle
# ============================================
print('=== Paso 2: Car y Motorcycle ===')

# Clases hijas con atributos específicos
# Descomenta las siguientes líneas:

# class Vehicle:
#     def __init__(self, brand: str, model: str, year: int):
#         self.brand = brand
#         self.model = model
#         self.year = year
#
#
# class Car(Vehicle):
#     """A car with doors."""
#
#     def __init__(self, brand: str, model: str, year: int, num_doors: int = 4):
#         super().__init__(brand, model, year)
#         self.num_doors = num_doors
#
#     def __str__(self) -> str:
#         return f"{self.brand} {self.model} ({self.num_doors} doors)"
#
#
# class Motorcycle(Vehicle):
#     """A motorcycle with engine size."""
#
#     def __init__(self, brand: str, model: str, year: int, engine_cc: int):
#         super().__init__(brand, model, year)
#         self.engine_cc = engine_cc
#
#     def __str__(self) -> str:
#         return f"{self.brand} {self.model} ({self.engine_cc}cc)"
#
#
# car = Car("Toyota", "Camry", 2020, 4)
# moto = Motorcycle("Harley-Davidson", "Street 750", 2019, 750)
#
# print(car)   # Toyota Camry (4 doors)
# print(moto)  # Harley-Davidson Street 750 (750cc)

print()

# ============================================
# PASO 3: Polimorfismo - describe()
# ============================================
print('=== Paso 3: Polimorfismo ===')

# Mismo método, diferentes implementaciones
# Descomenta las siguientes líneas:

# class Vehicle:
#     def __init__(self, brand: str, model: str, year: int):
#         self.brand = brand
#         self.model = model
#         self.year = year
#
#     def describe(self) -> str:
#         return f"{self.year} {self.brand} {self.model}"
#
#
# class Car(Vehicle):
#     def __init__(self, brand: str, model: str, year: int, num_doors: int = 4):
#         super().__init__(brand, model, year)
#         self.num_doors = num_doors
#
#     def describe(self) -> str:
#         base = super().describe()
#         return f"{base} ({self.num_doors} doors)"
#
#
# class Motorcycle(Vehicle):
#     def __init__(self, brand: str, model: str, year: int, engine_cc: int):
#         super().__init__(brand, model, year)
#         self.engine_cc = engine_cc
#
#     def describe(self) -> str:
#         base = super().describe()
#         return f"{base} ({self.engine_cc}cc)"
#
#
# # Polimorfismo: misma interfaz, diferente comportamiento
# vehicles: list[Vehicle] = [
#     Car("Toyota", "Camry", 2020, 4),
#     Motorcycle("Harley-Davidson", "Street 750", 2019, 750),
# ]
#
# for v in vehicles:
#     print(v.describe())  # Cada uno usa su implementación

print()

# ============================================
# PASO 4: Garage - Composición
# ============================================
print('=== Paso 4: Garage ===')

# Clase que contiene otras clases (composición)
# Descomenta las siguientes líneas:

# class Vehicle:
#     def __init__(self, brand: str, model: str, year: int):
#         self.brand = brand
#         self.model = model
#         self.year = year
#
#     def describe(self) -> str:
#         return f"{self.year} {self.brand} {self.model}"
#
#
# class Car(Vehicle):
#     def __init__(self, brand: str, model: str, year: int, num_doors: int = 4):
#         super().__init__(brand, model, year)
#         self.num_doors = num_doors
#
#     def describe(self) -> str:
#         return f"{super().describe()} ({self.num_doors} doors)"
#
#
# class Motorcycle(Vehicle):
#     def __init__(self, brand: str, model: str, year: int, engine_cc: int):
#         super().__init__(brand, model, year)
#         self.engine_cc = engine_cc
#
#     def describe(self) -> str:
#         return f"{super().describe()} ({self.engine_cc}cc)"
#
#
# class Garage:
#     """A garage that holds vehicles."""
#
#     def __init__(self, capacity: int):
#         self._capacity = capacity
#         self._vehicles: list[Vehicle] = []
#
#     @property
#     def capacity(self) -> int:
#         return self._capacity
#
#     @property
#     def count(self) -> int:
#         return len(self._vehicles)
#
#     @property
#     def is_full(self) -> bool:
#         return self.count >= self._capacity
#
#     def add_vehicle(self, vehicle: Vehicle) -> bool:
#         """Add vehicle if space available."""
#         if self.is_full:
#             return False
#         self._vehicles.append(vehicle)
#         return True
#
#     def list_vehicles(self) -> list[str]:
#         """Get descriptions of all vehicles."""
#         return [v.describe() for v in self._vehicles]
#
#     def __str__(self) -> str:
#         return f"Garage: {self.count}/{self._capacity} vehicles"
#
#
# # Crear garage y vehículos
# garage = Garage(3)
# car = Car("Toyota", "Camry", 2020, 4)
# moto = Motorcycle("Harley-Davidson", "Street 750", 2019, 750)
#
# # Agregar vehículos
# print(f"Added {car.brand} {car.model}: {garage.add_vehicle(car)}")
# print(f"Added {moto.brand} {moto.model}: {garage.add_vehicle(moto)}")
# print(garage)
#
# # Listar vehículos
# print("All vehicles:")
# for desc in garage.list_vehicles():
#     print(f"  - {desc}")

print()

# ============================================
# PASO 5: Factory Method
# ============================================
print('=== Paso 5: Factory Method ===')

# @classmethod como constructor alternativo
# Descomenta las siguientes líneas:

# class Vehicle:
#     def __init__(self, brand: str, model: str, year: int):
#         self.brand = brand
#         self.model = model
#         self.year = year
#
#
# class Car(Vehicle):
#     def __init__(self, brand: str, model: str, year: int, num_doors: int = 4):
#         super().__init__(brand, model, year)
#         self.num_doors = num_doors
#
#     @classmethod
#     def from_string(cls, data: str) -> Car:
#         """Create car from string 'brand-model-year-doors'."""
#         parts = data.split("-")
#         if len(parts) != 4:
#             raise ValueError("Format: brand-model-year-doors")
#         brand, model, year, doors = parts
#         return cls(brand, model, int(year), int(doors))
#
#     def describe(self) -> str:
#         return f"{self.year} {self.brand} {self.model} ({self.num_doors} doors)"
#
#
# # Crear desde string
# car = Car.from_string("Honda-Civic-2022-4")
# print(f"Created: {car.describe()}")

print()

# ============================================
# PASO 6: Sistema Completo
# ============================================
print('=== Paso 6: Sistema Completo ===')

# Sistema integrado con dataclass
# Descomenta las siguientes líneas:

# from dataclasses import dataclass, field
#
#
# @dataclass
# class Owner:
#     """Vehicle owner using dataclass."""
#     name: str
#     license_number: str
#     vehicles: list = field(default_factory=list)
#
#     def add_vehicle(self, brand: str, model: str, value: float) -> None:
#         self.vehicles.append({
#             "brand": brand,
#             "model": model,
#             "value": value
#         })
#
#     @property
#     def total_value(self) -> float:
#         return sum(v["value"] for v in self.vehicles)
#
#     def __str__(self) -> str:
#         return f"{self.name} ({self.license_number})"
#
#
# # Crear owner
# owner = Owner("Alice", "DL-12345")
# owner.add_vehicle("Toyota", "Camry", 25000)
# owner.add_vehicle("Honda", "CBR500R", 3500)
#
# print(f"Owner: {owner}")
# print(f"Vehicles: {len(owner.vehicles)}")
# print(f"Total value: ${owner.total_value:,.2f}")

print()
print('✅ Ejercicio completado!')
