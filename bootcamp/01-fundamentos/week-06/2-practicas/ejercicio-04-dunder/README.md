# 🏋️ Ejercicio 04: Dunder Methods

## 🎯 Objetivo

Aprender a implementar métodos especiales (dunder methods) para personalizar el comportamiento de tus clases.

---

## 📋 Conceptos

- `__str__` y `__repr__` para representación
- `__eq__` para comparar igualdad
- `__lt__`, `__le__`, `__gt__`, `__ge__` para ordenamiento
- `__len__`, `__getitem__` para comportamiento de colecciones

---

## 🚀 Pasos

### Paso 1: `__str__` y `__repr__`

`__str__` es para usuarios, `__repr__` es para desarrolladores.

```python
def __str__(self) -> str:
    return f"Point({self.x}, {self.y})"

def __repr__(self) -> str:
    return f"Point(x={self.x}, y={self.y})"
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: `__eq__` para Igualdad

Define cuándo dos objetos son "iguales".

```python
def __eq__(self, other: object) -> bool:
    if not isinstance(other, Point):
        return NotImplemented
    return self.x == other.x and self.y == other.y
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Operadores de Comparación

Implementa `__lt__` para habilitar ordenamiento con `sorted()`.

```python
def __lt__(self, other: "Point") -> bool:
    # Compare by distance from origin
    return self.distance < other.distance
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Operadores Aritméticos

Implementa `__add__`, `__sub__` para operaciones matemáticas.

```python
def __add__(self, other: "Vector") -> "Vector":
    return Vector(self.x + other.x, self.y + other.y)
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Colecciones Personalizadas

Implementa `__len__`, `__getitem__`, `__iter__` para comportamiento de lista.

**Descomenta** la sección del Paso 5.

---

## ✅ Resultado Esperado

```
--- Paso 1: __str__ y __repr__ ---
str: Point at (3, 4)
repr: Point(x=3, y=4)

--- Paso 2: __eq__ ---
p1 == p2: True
p1 == p3: False
p1 is p2: False

--- Paso 3: Ordenamiento ---
Original: [Point(3, 4), Point(1, 1), Point(0, 5)]
Sorted: [Point(1, 1), Point(3, 4), Point(0, 5)]

--- Paso 4: Operadores Aritméticos ---
v1 + v2 = Vector(4, 6)
v1 * 3 = Vector(3, 6)

--- Paso 5: Colección Personalizada ---
Playlist length: 3
First song: Bohemian Rhapsody
Songs: Bohemian Rhapsody, Stairway to Heaven, Hotel California
```

---

## 🔗 Navegación

| Anterior                                               | Índice                     | Siguiente                                |
| ------------------------------------------------------ | -------------------------- | ---------------------------------------- |
| [← Propiedades](../ejercicio-03-propiedades/README.md) | [Ejercicios](../README.md) | [Proyecto →](../../3-proyecto/README.md) |
