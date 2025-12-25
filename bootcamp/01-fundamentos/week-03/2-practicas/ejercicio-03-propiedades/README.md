# 🔐 Ejercicio 03: Properties y Encapsulamiento

## 🎯 Objetivo

Aprender a usar `@property` para encapsular atributos y validar datos.

---

## 📋 Conceptos Cubiertos

- Convención de atributos "privados" (`_`)
- Decorador `@property`
- Getters y setters pythónicos
- Validación de datos
- Properties computadas (read-only)

---

## 🚀 Instrucciones

### Paso 1: Atributos Protegidos

Convención de `_` para atributos internos:

```python
class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance  # Convención: no modificar directamente

    def get_balance(self) -> float:
        return self._balance
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Property Básica

Convertir getter en propiedad:

```python
class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance

    @property
    def balance(self) -> float:
        """Get the balance."""
        return self._balance

account = BankAccount(100)
print(account.balance)  # Acceso como atributo, no método
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Setter con Validación

Añadir setter que valida datos:

```python
class Circle:
    def __init__(self, radius: float):
        self.radius = radius  # Llama al setter

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Radius must be positive")
        self._radius = value
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Property Read-Only

Propiedades que solo se pueden leer:

```python
class Circle:
    @property
    def area(self) -> float:
        """Calculated property - read only."""
        import math
        return math.pi * self._radius ** 2

# circle.area = 100  # Error! No hay setter
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Clase User con Validaciones

Múltiples properties con validación:

```python
class User:
    @property
    def email(self) -> str:
        return self._email

    @email.setter
    def email(self, value: str) -> None:
        if "@" not in value:
            raise ValueError("Invalid email")
        self._email = value.lower()
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Producto con Stock

Ejemplo completo con properties:

```python
class Product:
    @property
    def value(self) -> float:
        """Total value (computed)."""
        return self._price * self._stock

    @property
    def is_available(self) -> bool:
        return self._stock > 0
```

**Descomenta** la sección del Paso 6.

---

## ✅ Resultado Esperado

```
=== Paso 1: Atributos Protegidos ===
Balance: 100.0

=== Paso 2: Property Básica ===
Balance: 100.0

=== Paso 3: Setter con Validación ===
Radius: 5.0
New radius: 10.0
Error: Radius must be positive

=== Paso 4: Property Read-Only ===
Radius: 5, Area: 78.54
Diameter: 10.0
Cannot set area (read-only)

=== Paso 5: User con Validaciones ===
Name: Alice
Email: alice@example.com
Error: Invalid email format

=== Paso 6: Producto con Stock ===
Laptop: $999.99 (10 units) - In Stock
Total value: $9,999.90
Sold 3 for $2,999.97
```

---

## 🔗 Recursos

- [property()](https://docs.python.org/3/library/functions.html#property)
- [Real Python - Properties](https://realpython.com/python-property/)

---

_Anterior: [Ejercicio 02](../ejercicio-02-herencia/) | Siguiente: [Ejercicio 04](../ejercicio-04-integrador/)_
