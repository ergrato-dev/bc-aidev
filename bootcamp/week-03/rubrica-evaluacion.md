# 📋 Rúbrica de Evaluación - Semana 03

## 🎯 Programación Orientada a Objetos

---

## 📊 Distribución de Evidencias

| Tipo de Evidencia | Peso | Actividades                     |
| ----------------- | ---- | ------------------------------- |
| 🧠 Conocimiento   | 30%  | Cuestionario teórico, glosario  |
| 💪 Desempeño      | 40%  | Ejercicios prácticos (4)        |
| 📦 Producto       | 30%  | Proyecto: Sistema de Biblioteca |

---

## 🧠 Evidencia de Conocimiento (30%)

### Cuestionario Teórico

| Criterio       | Puntos | Descripción                                        |
| -------------- | ------ | -------------------------------------------------- |
| Conceptos OOP  | 10     | Definir clase, objeto, atributo, método            |
| Pilares OOP    | 10     | Explicar encapsulamiento, herencia, polimorfismo   |
| Dunder methods | 5      | Conocer `__init__`, `__str__`, `__repr__`          |
| Decoradores    | 5      | Diferenciar @property, @classmethod, @staticmethod |
| **Total**      | **30** |                                                    |

### Preguntas de Ejemplo

1. ¿Cuál es la diferencia entre una clase y un objeto?
2. ¿Para qué sirve el método `__init__`?
3. ¿Qué es el MRO (Method Resolution Order)?
4. ¿Cuándo usar `@classmethod` vs `@staticmethod`?
5. ¿Qué significa "duck typing" en Python?

---

## 💪 Evidencia de Desempeño (40%)

### Ejercicio 01: Clases Básicas (10 pts)

| Criterio               | Puntos | Descripción                  |
| ---------------------- | ------ | ---------------------------- |
| Definición de clase    | 3      | Sintaxis correcta de `class` |
| Constructor `__init__` | 3      | Inicialización de atributos  |
| Métodos de instancia   | 2      | Uso correcto de `self`       |
| Método `__str__`       | 2      | Representación legible       |

### Ejercicio 02: Herencia (10 pts)

| Criterio            | Puntos | Descripción                           |
| ------------------- | ------ | ------------------------------------- |
| Herencia simple     | 3      | Clase hija hereda de padre            |
| Uso de `super()`    | 3      | Llamada correcta al constructor padre |
| Override de métodos | 2      | Sobrescritura apropiada               |
| Herencia múltiple   | 2      | Entender MRO                          |

### Ejercicio 03: Properties (10 pts)

| Criterio           | Puntos | Descripción                   |
| ------------------ | ------ | ----------------------------- |
| `@property` getter | 3      | Acceso controlado a atributos |
| `@name.setter`     | 3      | Validación en asignación      |
| Atributos privados | 2      | Convención `_nombre`          |
| Validaciones       | 2      | Manejo de errores             |

### Ejercicio 04: Integrador (10 pts)

| Criterio              | Puntos | Descripción                            |
| --------------------- | ------ | -------------------------------------- |
| Múltiples clases      | 3      | Sistema con varias clases relacionadas |
| Composición           | 3      | Objetos que contienen otros objetos    |
| Métodos colaborativos | 2      | Interacción entre objetos              |
| Type hints            | 2      | Anotaciones de tipos                   |

---

## 📦 Evidencia de Producto (30%)

### Proyecto: Sistema de Gestión de Biblioteca

| Criterio                 | Puntos | Descripción                                |
| ------------------------ | ------ | ------------------------------------------ |
| **Estructura de Clases** | 8      |                                            |
| Clase `Book`             | 2      | Atributos: título, autor, ISBN, disponible |
| Clase `User`             | 2      | Atributos: nombre, ID, libros prestados    |
| Clase `Library`          | 4      | Gestión de libros y usuarios               |
| **Funcionalidad**        | 12     |                                            |
| Agregar libros           | 2      | Método para añadir al catálogo             |
| Registrar usuarios       | 2      | Método para crear usuarios                 |
| Préstamo de libros       | 4      | Validar disponibilidad, actualizar estado  |
| Devolución de libros     | 4      | Actualizar disponibilidad y usuario        |
| **Calidad de Código**    | 10     |                                            |
| Encapsulamiento          | 3      | Uso de properties donde aplique            |
| Docstrings               | 2      | Documentación de clases y métodos          |
| Type hints               | 2      | Anotaciones de tipos                       |
| Métodos especiales       | 3      | `__str__`, `__repr__` implementados        |
| **Total**                | **30** |                                            |

---

## 📈 Escala de Calificación

| Rango  | Calificación | Descripción                   |
| ------ | ------------ | ----------------------------- |
| 90-100 | Excelente    | Dominio completo de OOP       |
| 80-89  | Muy Bien     | Buen manejo, detalles menores |
| 70-79  | Bien         | Cumple requisitos básicos     |
| 60-69  | Suficiente   | Necesita refuerzo             |
| < 60   | Insuficiente | No cumple objetivos mínimos   |

---

## ✅ Criterios de Aprobación

- [ ] Mínimo **70%** en cada tipo de evidencia
- [ ] Todos los ejercicios completados
- [ ] Proyecto funcional con clases requeridas
- [ ] Código ejecutable sin errores

---

## 🚀 Criterios de Excelencia (Bonus)

| Criterio      | Bonus | Descripción                            |
| ------------- | ----- | -------------------------------------- |
| Dataclasses   | +5    | Usar `@dataclass` para clases de datos |
| ABC/Protocols | +5    | Implementar clases abstractas          |
| Testing       | +5    | Incluir tests unitarios                |
| Documentación | +3    | README detallado del proyecto          |

---

## 📝 Retroalimentación

### Fortalezas Comunes

- Buena comprensión de clases básicas
- Uso correcto de herencia simple

### Áreas de Mejora Frecuentes

- Confusión entre `@classmethod` y `@staticmethod`
- Olvidar `super().__init__()` en herencia
- No usar `@property` para encapsular

---

## 🔗 Referencias

- [PEP 8 - Style Guide](https://peps.python.org/pep-0008/)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)

---

_Volver a: [Semana 03](README.md)_
