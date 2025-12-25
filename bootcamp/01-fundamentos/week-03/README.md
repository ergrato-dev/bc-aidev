# 🐍 Semana 03: Programación Orientada a Objetos

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OOP](https://img.shields.io/badge/OOP-Classes-FF6F00?style=for-the-badge)
![Nivel](https://img.shields.io/badge/Nivel-Principiante-4ecca3?style=for-the-badge)

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender los principios fundamentales de OOP
- ✅ Crear clases con atributos y métodos
- ✅ Utilizar `__init__` y otros métodos especiales (dunder methods)
- ✅ Implementar herencia simple y múltiple
- ✅ Aplicar encapsulamiento con propiedades
- ✅ Entender polimorfismo y duck typing
- ✅ Usar decoradores `@property`, `@classmethod`, `@staticmethod`
- ✅ Crear dataclasses para simplificar código

---

## 📚 Requisitos Previos

- ✅ Semana 01: Variables, tipos de datos, control de flujo
- ✅ Semana 02: Funciones y estructuras de datos

---

## 🗂️ Estructura de la Semana

```
week-03/
├── README.md                    # Este archivo
├── rubrica-evaluacion.md        # Criterios de evaluación
├── 0-assets/                    # Diagramas y recursos visuales
├── 1-teoria/                    # Material teórico
│   ├── 01-clases-objetos.md     # Fundamentos de clases
│   ├── 02-herencia.md           # Herencia y composición
│   ├── 03-encapsulamiento.md    # Properties y acceso
│   └── 04-polimorfismo.md       # Duck typing y protocolos
├── 2-practicas/                 # Ejercicios guiados
│   ├── ejercicio-01-clases/     # Crear clases básicas
│   ├── ejercicio-02-herencia/   # Jerarquías de clases
│   ├── ejercicio-03-propiedades/# Encapsulamiento
│   └── ejercicio-04-integrador/ # Sistema completo
├── 3-proyecto/                  # Proyecto semanal
│   ├── README.md                # Sistema de Gestión de Biblioteca
│   ├── starter/                 # Plantilla inicial
│   └── .solution/               # Solución de referencia
├── 4-recursos/                  # Material complementario
│   ├── ebooks-free/
│   ├── videografia/
│   └── webgrafia/
└── 5-glosario/                  # Términos clave
```

---

## 📝 Contenidos

### 1️⃣ Teoría (1.5 horas)

| Archivo                                                 | Tema                                 | Duración |
| ------------------------------------------------------- | ------------------------------------ | -------- |
| [01-clases-objetos.md](1-teoria/01-clases-objetos.md)   | Clases, objetos, `__init__`, métodos | 25 min   |
| [02-herencia.md](1-teoria/02-herencia.md)               | Herencia, `super()`, MRO             | 25 min   |
| [03-encapsulamiento.md](1-teoria/03-encapsulamiento.md) | Properties, getters/setters          | 20 min   |
| [04-polimorfismo.md](1-teoria/04-polimorfismo.md)       | Duck typing, protocolos, ABC         | 20 min   |

### 2️⃣ Prácticas (2.5 horas)

| Ejercicio                                             | Tema                         | Duración |
| ----------------------------------------------------- | ---------------------------- | -------- |
| [Ejercicio 01](2-practicas/ejercicio-01-clases/)      | Clases básicas y métodos     | 35 min   |
| [Ejercicio 02](2-practicas/ejercicio-02-herencia/)    | Herencia y jerarquías        | 35 min   |
| [Ejercicio 03](2-practicas/ejercicio-03-propiedades/) | Properties y validación      | 35 min   |
| [Ejercicio 04](2-practicas/ejercicio-04-integrador/)  | Sistema con múltiples clases | 45 min   |

### 3️⃣ Proyecto (2 horas)

| Proyecto                             | Descripción                                        |
| ------------------------------------ | -------------------------------------------------- |
| [Sistema de Biblioteca](3-proyecto/) | Gestión de libros, usuarios y préstamos usando OOP |

---

## ⏱️ Distribución del Tiempo

| Actividad    | Tiempo  | Porcentaje |
| ------------ | ------- | ---------- |
| 📖 Teoría    | 1.5 h   | 25%        |
| 💻 Prácticas | 2.5 h   | 42%        |
| 🏗️ Proyecto  | 2.0 h   | 33%        |
| **Total**    | **6 h** | **100%**   |

---

## 📌 Entregables

1. **Ejercicios completados** (4 ejercicios en `2-practicas/`)
2. **Proyecto funcional** (`3-proyecto/starter/`)
3. **Autoevaluación** del glosario

---

## 🎓 Conceptos Clave de la Semana

```python
# Clase con atributos y métodos
class Book:
    def __init__(self, title: str, author: str):
        self.title = title
        self.author = author

    def __str__(self) -> str:
        return f"{self.title} by {self.author}"

# Herencia
class EBook(Book):
    def __init__(self, title: str, author: str, file_size: int):
        super().__init__(title, author)
        self.file_size = file_size

# Property para encapsulamiento
class User:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not value.strip():
            raise ValueError("Name cannot be empty")
        self._name = value
```

---

## 🔗 Navegación

| ⬅️ Anterior                         | 📋 Índice       | Siguiente ➡️                       |
| ----------------------------------- | --------------- | ---------------------------------- |
| [Semana 02: Funciones](../week-02/) | [Bootcamp](../) | [Semana 04: Archivos](../week-04/) |

---

## 📚 Recursos Rápidos

- 📖 [Documentación oficial - Classes](https://docs.python.org/3/tutorial/classes.html)
- 🎥 [Corey Schafer - OOP Playlist](https://www.youtube.com/playlist?list=PL-osiE80TeTsqhIuOqKhwlXsIBIdSeYtc)
- 📝 [Real Python - OOP](https://realpython.com/python3-object-oriented-programming/)

---

_Última actualización: Diciembre 2024_
