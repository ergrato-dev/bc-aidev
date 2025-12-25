# 🌳 Ejercicio 02: Herencia

## 🎯 Objetivo

Aprender a implementar herencia, usar `super()` y entender el MRO.

---

## 📋 Conceptos Cubiertos

- Herencia simple
- Método `super()`
- Override de métodos
- `isinstance()` y `issubclass()`
- Herencia múltiple y MRO

---

## 🚀 Instrucciones

### Paso 1: Herencia Simple

Una clase que hereda de otra:

```python
class Animal:
    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        return "Some sound"

class Dog(Animal):  # Dog hereda de Animal
    def speak(self) -> str:  # Override
        return f"{self.name} says: Woof!"
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Usar `super()`

Llamar al constructor de la clase padre:

```python
class Animal:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

class Dog(Animal):
    def __init__(self, name: str, age: int, breed: str):
        super().__init__(name, age)  # Llama a Animal.__init__
        self.breed = breed  # Atributo adicional
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: isinstance() e issubclass()

Verificar tipos y relaciones de herencia:

```python
dog = Dog("Buddy", 3, "Labrador")

# isinstance - verifica si objeto es de cierta clase
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True (hereda de Animal)

# issubclass - verifica herencia entre clases
print(issubclass(Dog, Animal))  # True
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Extender Métodos

Añadir funcionalidad al método del padre:

```python
class Animal:
    def describe(self) -> str:
        return f"I am {self.name}"

class Dog(Animal):
    def describe(self) -> str:
        parent_desc = super().describe()  # Llamar al padre
        return f"{parent_desc} and I bark!"
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Herencia Múltiple

Heredar de varias clases:

```python
class Flyer:
    def fly(self) -> str:
        return "Flying!"

class Swimmer:
    def swim(self) -> str:
        return "Swimming!"

class Duck(Flyer, Swimmer):  # Hereda de ambas
    def quack(self) -> str:
        return "Quack!"
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Sistema de Empleados

Ejemplo completo con herencia:

```python
class Employee:
    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary

    def get_annual_salary(self) -> float:
        return self.salary * 12

class Manager(Employee):
    def __init__(self, name: str, salary: float, department: str):
        super().__init__(name, salary)
        self.department = department

    def get_annual_salary(self) -> float:
        return super().get_annual_salary() * 1.2  # 20% bonus
```

**Descomenta** la sección del Paso 6.

---

## ✅ Resultado Esperado

```
=== Paso 1: Herencia Simple ===
Buddy says: Woof!
Whiskers says: Meow!

=== Paso 2: Usar super() ===
Max, 5 years old, Golden Retriever

=== Paso 3: isinstance e issubclass ===
Is buddy a Dog? True
Is buddy an Animal? True
Is Dog subclass of Animal? True

=== Paso 4: Extender Métodos ===
I am Rex and I bark!

=== Paso 5: Herencia Múltiple ===
Flying!
Swimming!
Quack!
MRO: Duck -> Flyer -> Swimmer -> object

=== Paso 6: Sistema de Empleados ===
Alice (Engineering): $115,200.00/year
Bob: $60,000.00/year
```

---

## 🔗 Recursos

- [Python Inheritance](https://docs.python.org/3/tutorial/classes.html#inheritance)
- [super()](https://docs.python.org/3/library/functions.html#super)
- [MRO](https://www.python.org/download/releases/2.3/mro/)

---

_Anterior: [Ejercicio 01](../ejercicio-01-clases/) | Siguiente: [Ejercicio 03](../ejercicio-03-propiedades/)_
