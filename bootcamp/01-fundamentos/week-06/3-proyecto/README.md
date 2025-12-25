# 🎮 Proyecto: Sistema RPG

## Sistema de Personajes para Juego de Rol

---

## 📋 Descripción

Desarrolla un sistema de personajes para un juego RPG aplicando los cuatro pilares de la Programación Orientada a Objetos:

- **Clases y Objetos**: Modelar personajes con atributos y métodos
- **Herencia**: Crear tipos especializados (Warrior, Mage, Archer)
- **Encapsulamiento**: Proteger atributos con validación
- **Polimorfismo**: Comportamiento diferente para cada tipo

---

## 🎯 Objetivos

Al completar este proyecto serás capaz de:

- ✅ Diseñar jerarquías de clases
- ✅ Implementar herencia con `super()`
- ✅ Usar `@property` para validación de datos
- ✅ Sobrescribir métodos especiales (dunder methods)
- ✅ Aplicar polimorfismo en sistemas reales

---

## 📁 Estructura del Proyecto

```
3-proyecto/
├── README.md           # Este archivo
├── 0-assets/           # Diagramas del sistema
├── starter/            # Archivos a completar
│   ├── main.py         # Programa principal
│   ├── character.py    # Clase base Character
│   ├── classes.py      # Warrior, Mage, Archer
│   ├── inventory.py    # Sistema de inventario
│   └── combat.py       # Sistema de combate
└── .solution/          # Solución de referencia
```

---

## 🏗️ Arquitectura

### Diagrama de Clases

```
                    ┌─────────────┐
                    │  Character  │
                    │─────────────│
                    │ - _name     │
                    │ - _health   │
                    │ - _level    │
                    │ - inventory │
                    │─────────────│
                    │ + attack()  │
                    │ + defend()  │
                    │ + level_up()│
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
    │ Warrior │      │  Mage   │      │ Archer  │
    │─────────│      │─────────│      │─────────│
    │ + rage  │      │ + mana  │      │ + arrows│
    │─────────│      │─────────│      │─────────│
    │ attack()│      │ attack()│      │ attack()│
    │ defend()│      │cast_sp()│      │ shoot() │
    └─────────┘      └─────────┘      └─────────┘
```

---

## 📝 Requisitos

### 1. Clase Base `Character` (character.py)

```python
class Character:
    """Base class for all RPG characters."""

    def __init__(self, name: str, health: int = 100, level: int = 1):
        # Atributos con validación usando @property
        pass

    @property
    def name(self) -> str: ...

    @property
    def health(self) -> int: ...

    @health.setter
    def health(self, value: int) -> None:
        # Validar: 0 <= health <= max_health
        pass

    def attack(self, target: "Character") -> int:
        """Base attack - returns damage dealt."""
        pass

    def defend(self) -> None:
        """Reduce incoming damage next turn."""
        pass

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other) -> bool: ...
```

### 2. Clases Especializadas (classes.py)

#### Warrior

- **Atributo extra**: `rage` (0-100)
- **Attack**: Daño base + bonus por rage
- **Habilidad especial**: `berserk()` - consume rage, aumenta daño

#### Mage

- **Atributo extra**: `mana` (0-100)
- **Attack**: Daño mágico basado en level
- **Habilidad especial**: `cast_spell(spell_name)` - consume mana

#### Archer

- **Atributo extra**: `arrows` (cantidad)
- **Attack**: Daño a distancia, consume flechas
- **Habilidad especial**: `shoot()` - ataque crítico

### 3. Sistema de Inventario (inventory.py)

```python
class Item:
    """Base class for items."""
    name: str
    value: int

class Weapon(Item):
    damage: int

class Potion(Item):
    healing: int

class Inventory:
    """Character inventory with collection behavior."""

    def __len__(self) -> int: ...
    def __getitem__(self, index) -> Item: ...
    def __iter__(self) -> Iterator[Item]: ...
    def __contains__(self, item) -> bool: ...
```

### 4. Sistema de Combate (combat.py)

```python
class Combat:
    """Manages combat between characters."""

    def __init__(self, char1: Character, char2: Character): ...

    def execute_turn(self, attacker: Character,
                     defender: Character) -> str: ...

    def is_over(self) -> bool: ...

    def get_winner(self) -> Character | None: ...
```

---

## ⚙️ Funcionalidades

### Obligatorias

1. **Crear personajes** de cada tipo
2. **Sistema de atributos** con validación
3. **Herencia** correcta con `super()`
4. **Polimorfismo** en método `attack()`
5. **Inventario** con comportamiento de colección
6. **Combate básico** entre personajes

### Opcionales (Bonus)

- [ ] Guardar/cargar partida (JSON)
- [ ] Sistema de experiencia y niveles
- [ ] Equipar armas del inventario
- [ ] Múltiples hechizos para Mage
- [ ] Sistema de críticos aleatorios

---

## 🚀 Cómo Empezar

1. **Lee los archivos** en `starter/` para entender la estructura
2. **Implementa `Character`** primero (es la base de todo)
3. **Crea las clases hijas** una por una
4. **Implementa `Inventory`** y `Item`
5. **Implementa `Combat`** al final
6. **Prueba con `main.py`**

---

## ✅ Ejemplo de Uso

```python
# Create characters
warrior = Warrior("Conan", health=120)
mage = Mage("Gandalf", health=80, mana=100)

# Check attributes
print(warrior)  # Warrior: Conan (HP: 120, Level: 1, Rage: 0)
print(mage)     # Mage: Gandalf (HP: 80, Level: 1, Mana: 100)

# Combat
damage = warrior.attack(mage)
print(f"Warrior dealt {damage} damage!")
print(f"Mage health: {mage.health}")

# Special abilities
mage.cast_spell("Fireball", warrior)
warrior.berserk()

# Inventory
warrior.inventory.add(Weapon("Sword", damage=15))
warrior.inventory.add(Potion("Health Potion", healing=30))
print(f"Inventory size: {len(warrior.inventory)}")
```

---

## 📊 Criterios de Evaluación

| Criterio                             | Puntos  |
| ------------------------------------ | ------- |
| Clase `Character` con propiedades    | 20      |
| Herencia correcta (3 clases)         | 25      |
| Dunder methods (`__str__`, `__eq__`) | 15      |
| Sistema de inventario                | 20      |
| Sistema de combate                   | 15      |
| Código limpio y documentado          | 5       |
| **Total**                            | **100** |

---

## ⏱️ Tiempo Estimado

| Tarea                 | Tiempo   |
| --------------------- | -------- |
| Character base        | 30 min   |
| Clases especializadas | 40 min   |
| Inventory y Items     | 30 min   |
| Combat system         | 20 min   |
| **Total**             | **~2 h** |

---

## 🔗 Navegación

| Anterior                                | Inicio                    | Siguiente                             |
| --------------------------------------- | ------------------------- | ------------------------------------- |
| [← Prácticas](../2-practicas/README.md) | [Semana 06](../README.md) | [Recursos →](../4-recursos/README.md) |
