# 🏋️ Ejercicio 03: Propiedades

## 🎯 Objetivo

Aprender a usar `@property` para crear getters y setters con validación.

---

## 📋 Conceptos

- Decorador `@property` para getters
- Decorador `@nombre.setter` para setters
- Validación de datos en setters
- Atributos calculados (computed properties)

---

## 🚀 Pasos

### Paso 1: Problema sin Propiedades

Sin propiedades, cualquiera puede asignar valores inválidos.

```python
person.age = -5  # ¡Esto no debería ser válido!
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Crear una Propiedad Básica

`@property` convierte un método en un atributo de solo lectura.

```python
@property
def age(self) -> int:
    return self._age
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Agregar un Setter

`@nombre.setter` permite asignar valores con validación.

```python
@age.setter
def age(self, value: int) -> None:
    if value < 0:
        raise ValueError("Age cannot be negative")
    self._age = value
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Propiedades Calculadas

Las propiedades pueden calcular valores dinámicamente.

```python
@property
def full_name(self) -> str:
    return f"{self.first_name} {self.last_name}"
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Clase Temperature Completa

Implementa una clase con conversión automática entre Celsius y Fahrenheit.

**Descomenta** la sección del Paso 5.

---

## ✅ Resultado Esperado

```
--- Paso 1: Problema sin Propiedades ---
Age: -5 (¡Esto no debería ser válido!)

--- Paso 2: Propiedad Básica ---
Age (read-only): 25

--- Paso 3: Setter con Validación ---
Age set to: 30
Error caught: Age cannot be negative

--- Paso 4: Propiedades Calculadas ---
Full name: John Doe
Email: john.doe@example.com

--- Paso 5: Temperature ---
Celsius: 25.0
Fahrenheit: 77.0
After setting Fahrenheit to 32:
Celsius: 0.0
```

---

## 🔗 Navegación

| Anterior                                         | Índice                     | Siguiente                                            |
| ------------------------------------------------ | -------------------------- | ---------------------------------------------------- |
| [← Herencia](../ejercicio-02-herencia/README.md) | [Ejercicios](../README.md) | [Dunder Methods →](../ejercicio-04-dunder/README.md) |
