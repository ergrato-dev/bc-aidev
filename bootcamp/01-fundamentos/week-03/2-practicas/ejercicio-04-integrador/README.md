# 🔗 Ejercicio 04: Sistema Integrador

## 🎯 Objetivo

Combinar clases, herencia, properties y polimorfismo en un sistema completo.

---

## 📋 Conceptos Cubiertos

- Múltiples clases relacionadas
- Herencia y composición
- Properties con validación
- Polimorfismo (duck typing)
- `@classmethod` y `@staticmethod`
- Dataclasses

---

## 🚀 Instrucciones

### Paso 1: Clase Base con Property

Crear clase base `Vehicle`:

```python
class Vehicle:
    def __init__(self, brand: str, model: str, year: int):
        self.brand = brand
        self.model = model
        self.year = year

    @property
    def age(self) -> int:
        from datetime import date
        return date.today().year - self.year
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Herencia - Car y Motorcycle

Clases hijas con atributos propios:

```python
class Car(Vehicle):
    def __init__(self, brand, model, year, num_doors: int):
        super().__init__(brand, model, year)
        self.num_doors = num_doors

class Motorcycle(Vehicle):
    def __init__(self, brand, model, year, engine_cc: int):
        super().__init__(brand, model, year)
        self.engine_cc = engine_cc
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Polimorfismo - describe()

Método común con diferentes implementaciones:

```python
class Vehicle:
    def describe(self) -> str:
        return f"{self.year} {self.brand} {self.model}"

class Car(Vehicle):
    def describe(self) -> str:
        base = super().describe()
        return f"{base} ({self.num_doors} doors)"
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Garage - Composición

Clase que contiene otros objetos:

```python
class Garage:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._vehicles: list[Vehicle] = []

    def add_vehicle(self, vehicle: Vehicle) -> bool:
        if len(self._vehicles) >= self.capacity:
            return False
        self._vehicles.append(vehicle)
        return True
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: @classmethod Factory

Constructor alternativo:

```python
class Car(Vehicle):
    @classmethod
    def from_string(cls, data: str) -> "Car":
        """Create car from string 'brand-model-year-doors'."""
        brand, model, year, doors = data.split("-")
        return cls(brand, model, int(year), int(doors))
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Sistema Completo

Uso de dataclass y sistema integrado:

```python
from dataclasses import dataclass

@dataclass
class Owner:
    name: str
    license_number: str

    def can_drive(self, vehicle: Vehicle) -> bool:
        return True  # Simplified
```

**Descomenta** la sección del Paso 6.

---

## ✅ Resultado Esperado

```
=== Paso 1: Clase Base Vehicle ===
2020 Toyota Camry
Age: 4 years

=== Paso 2: Car y Motorcycle ===
Toyota Camry (4 doors)
Harley-Davidson Street 750 (750cc)

=== Paso 3: Polimorfismo ===
2020 Toyota Camry (4 doors)
2019 Harley-Davidson Street 750 (750cc)

=== Paso 4: Garage ===
Added Toyota Camry: True
Added Harley-Davidson Street 750: True
Garage: 2/3 vehicles
All vehicles:
  - 2020 Toyota Camry (4 doors)
  - 2019 Harley-Davidson Street 750 (750cc)

=== Paso 5: Factory Method ===
Created: 2022 Honda Civic (4 doors)

=== Paso 6: Sistema Completo ===
Owner: Alice (DL-12345)
Vehicles: 2
Total value: $28,500.00
```

---

## 🔗 Recursos

- [Composition vs Inheritance](https://realpython.com/inheritance-composition-python/)
- [Dataclasses](https://docs.python.org/3/library/dataclasses.html)

---

_Anterior: [Ejercicio 03](../ejercicio-03-propiedades/)_
