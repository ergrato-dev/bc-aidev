# 📅 Semana 06: Programación Orientada a Objetos (POO)

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender los conceptos fundamentales de la POO
- ✅ Crear clases con atributos y métodos en Python
- ✅ Implementar herencia simple y múltiple
- ✅ Aplicar encapsulamiento con propiedades
- ✅ Utilizar polimorfismo y métodos especiales (dunder methods)
- ✅ Diseñar sistemas modulares con clases colaborativas

---

## 📚 Requisitos Previos

- ✅ Semana 01-05 completadas
- ✅ Funciones y módulos
- ✅ Estructuras de datos (listas, diccionarios)
- ✅ Manejo de archivos y excepciones

---

## 🗂️ Estructura de la Semana

```
week-06/
├── README.md                 # Este archivo
├── rubrica-evaluacion.md     # Criterios de evaluación
├── 0-assets/                 # Diagramas SVG
│   ├── 01-clases-objetos.svg
│   ├── 02-herencia.svg
│   ├── 03-polimorfismo.svg
│   └── 04-sistema-rpg.svg
├── 1-teoria/                 # Material teórico
│   ├── 01-clases-objetos.md
│   ├── 02-herencia.md
│   ├── 03-encapsulamiento.md
│   └── 04-polimorfismo.md
├── 2-practicas/              # Ejercicios guiados
│   ├── README.md
│   ├── ejercicio-01-clases/
│   ├── ejercicio-02-herencia/
│   ├── ejercicio-03-propiedades/
│   └── ejercicio-04-dunder/
├── 3-proyecto/               # Proyecto integrador
│   ├── README.md
│   ├── starter/
│   └── .solution/
├── 4-recursos/               # Material complementario
│   ├── README.md
│   ├── ebooks-free/
│   ├── videografia/
│   └── webgrafia/
└── 5-glosario/               # Términos clave
    └── README.md
```

---

## 📝 Contenidos

### 1️⃣ Teoría

| Archivo                                                 | Tema             | Conceptos                                             |
| ------------------------------------------------------- | ---------------- | ----------------------------------------------------- |
| [01-clases-objetos.md](1-teoria/01-clases-objetos.md)   | Clases y Objetos | `class`, `__init__`, atributos, métodos, `self`       |
| [02-herencia.md](1-teoria/02-herencia.md)               | Herencia         | Simple, múltiple, `super()`, MRO                      |
| [03-encapsulamiento.md](1-teoria/03-encapsulamiento.md) | Encapsulamiento  | Propiedades, `@property`, getters/setters, `_` y `__` |
| [04-polimorfismo.md](1-teoria/04-polimorfismo.md)       | Polimorfismo     | Duck typing, dunder methods, ABC                      |

### 2️⃣ Prácticas

| Ejercicio                                             | Tema           | Habilidades                               |
| ----------------------------------------------------- | -------------- | ----------------------------------------- |
| [Ejercicio 01](2-practicas/ejercicio-01-clases/)      | Clases Básicas | Definir clases, crear instancias          |
| [Ejercicio 02](2-practicas/ejercicio-02-herencia/)    | Herencia       | Extender clases, super()                  |
| [Ejercicio 03](2-practicas/ejercicio-03-propiedades/) | Propiedades    | @property, validación                     |
| [Ejercicio 04](2-practicas/ejercicio-04-dunder/)      | Dunder Methods | `__str__`, `__repr__`, `__eq__`, `__lt__` |

### 3️⃣ Proyecto: Sistema RPG

Desarrollar un sistema de personajes para un juego RPG con:

- Clases base y especializadas (Warrior, Mage, Archer)
- Sistema de inventario con items
- Combate entre personajes
- Guardado/carga con JSON

[Ver instrucciones completas →](3-proyecto/README.md)

---

## ⏱️ Distribución del Tiempo

| Actividad    | Tiempo  | Descripción                  |
| ------------ | ------- | ---------------------------- |
| 📖 Teoría    | 1.5 h   | Leer material y ejemplos     |
| 💻 Prácticas | 2.5 h   | Completar ejercicios guiados |
| 🚀 Proyecto  | 2 h     | Implementar Sistema RPG      |
| **Total**    | **6 h** |                              |

---

## 📌 Entregables

1. **Ejercicios completados** (4 ejercicios en `2-practicas/`)
2. **Proyecto funcional** (Sistema RPG en `3-proyecto/`)
3. **Código documentado** con docstrings

---

## 🎯 Criterios de Evaluación

| Criterio        | Peso | Descripción                            |
| --------------- | ---- | -------------------------------------- |
| 🧠 Conocimiento | 30%  | Comprensión de conceptos POO           |
| 💪 Desempeño    | 40%  | Ejercicios completados correctamente   |
| 📦 Producto     | 30%  | Proyecto funcional y bien estructurado |

[Ver rúbrica detallada →](rubrica-evaluacion.md)

---

## 💡 Conceptos Clave

```python
# Clase con atributos y métodos
class Character:
    def __init__(self, name: str, health: int = 100):
        self.name = name
        self.health = health

    def take_damage(self, amount: int) -> None:
        self.health = max(0, self.health - amount)

# Herencia
class Warrior(Character):
    def __init__(self, name: str):
        super().__init__(name, health=150)
        self.armor = 20

# Instanciación
hero = Warrior("Aragorn")
hero.take_damage(30)
```

---

## 🔗 Navegación

| Anterior                                                    | Inicio                         | Siguiente                           |
| ----------------------------------------------------------- | ------------------------------ | ----------------------------------- |
| [← Semana 05: Archivos y Excepciones](../week-05/README.md) | [🏠 Bootcamp](../../README.md) | [Semana 07 →](../week-07/README.md) |

---

## 📚 Recursos Adicionales

- [Python OOP Tutorial](https://realpython.com/python3-object-oriented-programming/)
- [Python Data Model](https://docs.python.org/3/reference/datamodel.html)
- [Documentación oficial: Classes](https://docs.python.org/3/tutorial/classes.html)

---

_Semana 06 de 36 | Módulo: Fundamentos de Python_
