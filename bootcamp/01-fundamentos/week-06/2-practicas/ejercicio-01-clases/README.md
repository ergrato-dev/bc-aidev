# 🏋️ Ejercicio 01: Clases Básicas

## 🎯 Objetivo

Aprender a definir clases, crear instancias y trabajar con atributos y métodos.

---

## 📋 Conceptos

- Definición de clases con `class`
- Método constructor `__init__`
- Atributos de instancia con `self`
- Métodos de instancia
- Atributos de clase

---

## 🚀 Pasos

### Paso 1: Definir una Clase Básica

Una clase es una plantilla para crear objetos. El método `__init__` es el constructor.

```python
class Dog:
    def __init__(self, name: str, age: int):
        self.name = name  # Instance attribute
        self.age = age
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Crear Instancias

Las instancias son objetos concretos creados a partir de la clase.

```python
fido = Dog("Fido", 3)
print(fido.name)  # Fido
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Agregar Métodos

Los métodos son funciones que pertenecen a la clase y operan sobre `self`.

```python
def bark(self) -> str:
    return f"{self.name} says Woof!"
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Atributos de Clase

Los atributos de clase son compartidos por todas las instancias.

```python
class Dog:
    species = "Canis familiaris"  # Class attribute
    count = 0
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Clase Completa - BankAccount

Aplica lo aprendido creando una clase `BankAccount` con métodos para depositar y retirar.

**Descomenta** la sección del Paso 5.

---

## ✅ Resultado Esperado

```
--- Paso 1: Definir Clase ---
Dog class defined

--- Paso 2: Crear Instancias ---
Name: Fido, Age: 3
Name: Rex, Age: 5

--- Paso 3: Métodos ---
Fido says Woof!
Rex is now 6 years old

--- Paso 4: Atributos de Clase ---
Species: Canis familiaris
Total dogs created: 2

--- Paso 5: BankAccount ---
Alice: $1000.00
After deposit: $1500.00
Withdrawal successful: True
Final balance: $1300.00
Withdrawal failed: False
```

---

## 🔗 Navegación

| Anterior                    | Índice                     | Siguiente                                        |
| --------------------------- | -------------------------- | ------------------------------------------------ |
| [← Prácticas](../README.md) | [Ejercicios](../README.md) | [Herencia →](../ejercicio-02-herencia/README.md) |
