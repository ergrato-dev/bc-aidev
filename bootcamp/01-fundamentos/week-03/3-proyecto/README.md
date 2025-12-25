# 📚 Proyecto: Sistema de Gestión de Biblioteca

## 🎯 Objetivo

Crear un sistema de gestión de biblioteca usando OOP que permita administrar libros, usuarios y préstamos.

---

## 📋 Descripción

Desarrollarás un sistema con las siguientes clases:

![Pipeline del Sistema](../0-assets/04-biblioteca-sistema.svg)

- **Book**: Representa un libro con título, autor, ISBN
- **User**: Usuario que puede tomar libros prestados
- **Library**: Gestiona libros, usuarios y préstamos

---

## 🏗️ Arquitectura

```
Library
├── books: list[Book]
├── users: list[User]
└── loans: dict[str, str]  # ISBN -> user_id

Book
├── title: str
├── author: str
├── isbn: str
└── available: bool

User
├── name: str
├── user_id: str
└── borrowed_books: list[str]  # ISBNs
```

---

## 📝 Requisitos

### Clase `Book`

| Atributo/Método | Descripción                        |
| --------------- | ---------------------------------- |
| `title`         | Título del libro                   |
| `author`        | Autor del libro                    |
| `isbn`          | Código ISBN único                  |
| `available`     | Property: disponible para préstamo |
| `__str__`       | Representación legible             |
| `__repr__`      | Representación para debugging      |

### Clase `User`

| Atributo/Método  | Descripción                                      |
| ---------------- | ------------------------------------------------ |
| `name`           | Nombre del usuario                               |
| `user_id`        | ID único del usuario                             |
| `borrowed_books` | Lista de ISBNs prestados                         |
| `borrow_count`   | Property: cantidad de libros prestados           |
| `can_borrow`     | Property: True si puede tomar más libros (máx 3) |
| `__str__`        | Representación legible                           |

### Clase `Library`

| Atributo/Método              | Descripción                   |
| ---------------------------- | ----------------------------- |
| `name`                       | Nombre de la biblioteca       |
| `add_book(book)`             | Agregar libro al catálogo     |
| `register_user(user)`        | Registrar nuevo usuario       |
| `find_book(isbn)`            | Buscar libro por ISBN         |
| `find_user(user_id)`         | Buscar usuario por ID         |
| `loan_book(isbn, user_id)`   | Prestar libro a usuario       |
| `return_book(isbn, user_id)` | Devolver libro                |
| `get_available_books()`      | Lista de libros disponibles   |
| `get_user_loans(user_id)`    | Libros prestados a un usuario |

---

## 🚀 Instrucciones

1. **Abre `starter/main.py`**
2. **Implementa cada clase** siguiendo los TODOs
3. **Ejecuta** para probar tu implementación
4. **Compara** con `.solution/main.py` si necesitas ayuda

---

## ✅ Resultado Esperado

```
=== Library System Demo ===

Added books:
  - 1984 by George Orwell (978-0-452-28423-4) - Available

Registered users:
  - Alice (U001) - 0 books borrowed

Loan operations:
  Alice borrowed '1984': True
  1984 by George Orwell - Not Available
  Alice's books: ['978-0-452-28423-4']

Return operations:
  Alice returned '1984': True
  1984 by George Orwell - Available
  Alice's books: []

Available books: 3
```

---

## 📊 Criterios de Evaluación

| Criterio                    | Puntos |
| --------------------------- | ------ |
| Clase `Book` completa       | 8      |
| Clase `User` con properties | 8      |
| Clase `Library` funcional   | 10     |
| Validaciones y errores      | 4      |
| **Total**                   | **30** |

---

## 💡 Tips

- Usa `@property` para `available`, `borrow_count`, `can_borrow`
- Valida que el libro exista antes de prestar
- Valida que el usuario pueda tomar más libros
- Usa `__str__` para mostrar información legible

---

## 🔗 Recursos

- [Python Classes](https://docs.python.org/3/tutorial/classes.html)
- [Properties](https://docs.python.org/3/library/functions.html#property)

---

_Volver a: [Semana 03](../README.md)_
