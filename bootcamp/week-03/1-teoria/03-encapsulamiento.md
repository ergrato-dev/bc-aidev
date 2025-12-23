# 🔐 Encapsulamiento en Python

## 🎯 Objetivos

- Comprender el encapsulamiento y su propósito
- Usar convenciones de acceso (`_` y `__`)
- Implementar propiedades con `@property`
- Crear getters y setters pythónicos
- Validar datos en la asignación

---

## 📋 Contenido

### 1. ¿Qué es el Encapsulamiento?

El **encapsulamiento** es ocultar los detalles internos de un objeto y exponer solo lo necesario. En Python es por **convención**, no por restricción del lenguaje.

![Encapsulamiento](../0-assets/03-encapsulamiento.svg)

#### Niveles de Acceso (Convención)

| Prefijo   | Significado | Ejemplo         | Acceso                  |
| --------- | ----------- | --------------- | ----------------------- |
| (ninguno) | Público     | `self.name`     | Libre acceso            |
| `_`       | Protegido   | `self._balance` | "No tocar" (convención) |
| `__`      | Privado     | `self.__secret` | Name mangling aplicado  |

---

### 2. Atributos "Protegidos" con `_`

Un guion bajo indica "uso interno, no modificar directamente":

```python
class BankAccount:
    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner      # Público
        self._balance = balance  # "Protegido" - no modificar directamente

    def deposit(self, amount: float) -> None:
        if amount > 0:
            self._balance += amount

    def get_balance(self) -> float:
        return self._balance

account = BankAccount("Alice", 100)

# Funciona, pero NO se recomienda
account._balance = 1000000  # ⚠️ Mal estilo, pero Python lo permite

# La forma correcta
account.deposit(500)
print(account.get_balance())
```

---

### 3. Name Mangling con `__`

Doble guion bajo activa "name mangling" para evitar colisiones en herencia:

```python
class Parent:
    def __init__(self):
        self.__secret = "Parent's secret"
        self._protected = "Protected"

    def reveal(self) -> str:
        return self.__secret

class Child(Parent):
    def __init__(self):
        super().__init__()
        self.__secret = "Child's secret"  # Diferente atributo!

    def child_reveal(self) -> str:
        return self.__secret

obj = Child()
print(obj.reveal())       # Parent's secret
print(obj.child_reveal()) # Child's secret

# Name mangling renombra __secret
print(obj._Parent__secret)  # Parent's secret
print(obj._Child__secret)   # Child's secret

# ❌ Esto falla
# print(obj.__secret)  # AttributeError
```

#### ⚠️ `__` NO es para seguridad

```python
# El name mangling NO es seguridad real
class Secret:
    def __init__(self):
        self.__password = "12345"

s = Secret()
# print(s.__password)  # AttributeError

# Pero se puede acceder así:
print(s._Secret__password)  # 12345 (¡expuesto!)
```

---

### 4. Properties: La Forma Pythónica

`@property` permite acceso controlado con sintaxis de atributo:

```python
class Circle:
    def __init__(self, radius: float):
        self._radius = radius

    @property
    def radius(self) -> float:
        """Get the radius."""
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        """Set the radius with validation."""
        if value <= 0:
            raise ValueError("Radius must be positive")
        self._radius = value

    @property
    def area(self) -> float:
        """Calculate area (read-only property)."""
        import math
        return math.pi * self._radius ** 2

    @property
    def diameter(self) -> float:
        """Get diameter."""
        return self._radius * 2

    @diameter.setter
    def diameter(self, value: float) -> None:
        """Set diameter (updates radius)."""
        self.radius = value / 2  # Usa el setter de radius

# Uso - sintaxis limpia de atributo
circle = Circle(5)

print(circle.radius)    # 5 (getter)
print(circle.area)      # 78.54... (computed property)
print(circle.diameter)  # 10

circle.radius = 10      # Setter con validación
print(circle.area)      # 314.16...

circle.diameter = 6     # Setter que actualiza radius
print(circle.radius)    # 3

# circle.radius = -5    # ValueError: Radius must be positive
# circle.area = 100     # AttributeError: can't set (read-only)
```

---

### 5. Property para Validación

```python
class User:
    def __init__(self, name: str, email: str, age: int):
        # Los setters se llaman automáticamente
        self.name = name
        self.email = email
        self.age = age

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not value or not value.strip():
            raise ValueError("Name cannot be empty")
        self._name = value.strip()

    @property
    def email(self) -> str:
        return self._email

    @email.setter
    def email(self, value: str) -> None:
        if "@" not in value:
            raise ValueError("Invalid email format")
        self._email = value.lower()

    @property
    def age(self) -> int:
        return self._age

    @age.setter
    def age(self, value: int) -> None:
        if not isinstance(value, int) or value < 0 or value > 150:
            raise ValueError("Age must be between 0 and 150")
        self._age = value

# Uso
user = User("Alice", "Alice@Example.com", 25)
print(user.name)   # Alice
print(user.email)  # alice@example.com (normalizado)
print(user.age)    # 25

user.age = 30      # OK
# user.age = -5    # ValueError
# user.email = "invalid"  # ValueError
```

---

### 6. Property Read-Only

```python
class Employee:
    _id_counter = 0

    def __init__(self, name: str, salary: float):
        Employee._id_counter += 1
        self._id = Employee._id_counter  # Asignado una vez
        self._name = name
        self._salary = salary
        self._hire_date = self._get_current_date()

    def _get_current_date(self) -> str:
        from datetime import date
        return date.today().isoformat()

    @property
    def id(self) -> int:
        """Employee ID (read-only)."""
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def hire_date(self) -> str:
        """Hire date (read-only)."""
        return self._hire_date

    @property
    def salary(self) -> float:
        return self._salary

    @salary.setter
    def salary(self, value: float) -> None:
        if value < 0:
            raise ValueError("Salary cannot be negative")
        self._salary = value

emp = Employee("Bob", 50000)
print(emp.id)         # 1 (read-only)
print(emp.hire_date)  # 2024-12-01 (read-only)

emp.name = "Robert"   # OK - tiene setter
emp.salary = 55000    # OK - tiene setter

# emp.id = 999        # AttributeError - no setter
# emp.hire_date = "2020-01-01"  # AttributeError - no setter
```

---

### 7. Deleter (Opcional)

```python
class Document:
    def __init__(self, content: str):
        self._content = content

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        self._content = value

    @content.deleter
    def content(self) -> None:
        print("Deleting content...")
        self._content = ""

doc = Document("Hello World")
print(doc.content)  # Hello World

del doc.content     # Deleting content...
print(doc.content)  # (vacío)
```

---

### 8. Ejemplo Completo: Producto con Stock

```python
class Product:
    """A product with price and stock management."""

    def __init__(self, name: str, price: float, stock: int = 0):
        self._name = name
        self.price = price  # Usa setter para validación
        self.stock = stock  # Usa setter para validación

    @property
    def name(self) -> str:
        """Product name (read-only after creation)."""
        return self._name

    @property
    def price(self) -> float:
        """Product price."""
        return self._price

    @price.setter
    def price(self, value: float) -> None:
        if value < 0:
            raise ValueError("Price cannot be negative")
        self._price = round(value, 2)

    @property
    def stock(self) -> int:
        """Current stock quantity."""
        return self._stock

    @stock.setter
    def stock(self, value: int) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError("Stock must be a non-negative integer")
        self._stock = value

    @property
    def value(self) -> float:
        """Total value of stock (read-only, computed)."""
        return self._price * self._stock

    @property
    def is_available(self) -> bool:
        """Check if product is in stock."""
        return self._stock > 0

    def add_stock(self, quantity: int) -> None:
        """Add to stock."""
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        self.stock += quantity

    def sell(self, quantity: int = 1) -> float:
        """Sell product and return total price."""
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if quantity > self._stock:
            raise ValueError(f"Not enough stock. Available: {self._stock}")
        self.stock -= quantity
        return self._price * quantity

    def __str__(self) -> str:
        status = "In Stock" if self.is_available else "Out of Stock"
        return f"{self._name}: ${self._price:.2f} ({self._stock} units) - {status}"


# Uso
laptop = Product("Laptop", 999.99, 10)
print(laptop)  # Laptop: $999.99 (10 units) - In Stock

print(f"Total value: ${laptop.value:,.2f}")  # $9,999.90

# Vender
total = laptop.sell(3)
print(f"Sold 3 for ${total:,.2f}")  # Sold 3 for $2,999.97
print(laptop)  # Laptop: $999.99 (7 units) - In Stock

# Agregar stock
laptop.add_stock(5)
print(laptop)  # Laptop: $999.99 (12 units) - In Stock

# Cambiar precio
laptop.price = 899.99
print(laptop)  # Laptop: $899.99 (12 units) - In Stock

# Validaciones
# laptop.price = -100  # ValueError
# laptop.stock = -5    # ValueError
# laptop.sell(100)     # ValueError: Not enough stock
```

---

## 💡 Buenas Prácticas

1. **Usar `_` para atributos internos**: Indica "no modificar directamente"
2. **Preferir `@property` sobre getters/setters explícitos**
3. **Validar en setters**: Mantener integridad de datos
4. **Properties computadas**: Para valores derivados
5. **Read-only para datos inmutables**: Sin setter

---

## ⚠️ Errores Comunes

```python
# ❌ Recursión infinita
class Bad:
    @property
    def value(self):
        return self.value  # ¡Llama a sí mismo!

    @value.setter
    def value(self, v):
        self.value = v  # ¡Recursión infinita!

# ✅ Correcto - usar atributo con _
class Good:
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v

# ❌ No inicializar atributo
class Bad2:
    @property
    def name(self):
        return self._name  # ¡_name no existe!

# ✅ Inicializar en __init__ o setter
class Good2:
    def __init__(self, name: str):
        self.name = name  # Llama al setter

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
```

---

## ✅ Checklist de Verificación

- [ ] Entiendo la convención `_` y `__`
- [ ] Puedo crear properties con `@property`
- [ ] Sé implementar setters con validación
- [ ] Puedo crear properties read-only
- [ ] Entiendo properties computadas

---

## 🔗 Recursos

- [Python Docs - Property](https://docs.python.org/3/library/functions.html#property)
- [Real Python - Properties](https://realpython.com/python-property/)
- [Descriptor Protocol](https://docs.python.org/3/howto/descriptor.html)

---

_Anterior: [Herencia](02-herencia.md) | Siguiente: [Polimorfismo](04-polimorfismo.md)_
