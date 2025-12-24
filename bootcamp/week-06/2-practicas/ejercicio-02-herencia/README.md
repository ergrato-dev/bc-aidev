# 🏋️ Ejercicio 02: Herencia

## 🎯 Objetivo

Aprender a crear jerarquías de clases usando herencia y el método `super()`.

---

## 📋 Conceptos

- Herencia simple: `class Child(Parent)`
- Método `super()` para llamar al padre
- Sobrescritura (override) de métodos
- Extensión de métodos del padre

---

## 🚀 Pasos

### Paso 1: Herencia Básica

La clase hija hereda todos los atributos y métodos de la clase padre.

```python
class Animal:
    def __init__(self, name: str):
        self.name = name

class Dog(Animal):  # Dog hereda de Animal
    pass
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Usar super()

`super()` permite llamar al constructor y métodos del padre.

```python
class Dog(Animal):
    def __init__(self, name: str, breed: str):
        super().__init__(name)  # Llama a Animal.__init__
        self.breed = breed
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Sobrescribir Métodos

La clase hija puede redefinir métodos del padre.

```python
class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"  # Override del método speak
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Extender Métodos

Puedes llamar al método del padre y agregar funcionalidad.

```python
def speak(self) -> str:
    parent_msg = super().speak()
    return f"{parent_msg} (from a dog)"
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Jerarquía de Vehículos

Crea una jerarquía completa: Vehicle → Car → ElectricCar.

**Descomenta** la sección del Paso 5.

---

## ✅ Resultado Esperado

```
--- Paso 1: Herencia Básica ---
Fido (inherited from Animal)
Is Dog instance of Animal? True

--- Paso 2: Usar super() ---
Fido is a Labrador

--- Paso 3: Sobrescribir Métodos ---
Generic animal sound
Woof!
Meow!

--- Paso 4: Extender Métodos ---
Woof! (I'm a happy dog!)

--- Paso 5: Jerarquía de Vehículos ---
Toyota Corolla starting...
Toyota Corolla starting... Engine running!
Tesla Model 3 starting... Engine running! Battery: 75 kWh
```

---

## 🔗 Navegación

| Anterior                                     | Índice                     | Siguiente                                              |
| -------------------------------------------- | -------------------------- | ------------------------------------------------------ |
| [← Clases](../ejercicio-01-clases/README.md) | [Ejercicios](../README.md) | [Propiedades →](../ejercicio-03-propiedades/README.md) |
