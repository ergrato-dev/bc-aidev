# 📋 Rúbrica de Evaluación - Semana 06

## Programación Orientada a Objetos (POO)

---

## 📊 Distribución de Puntaje

| Tipo de Evidencia | Peso     | Puntos      |
| ----------------- | -------- | ----------- |
| 🧠 Conocimiento   | 30%      | 30 pts      |
| 💪 Desempeño      | 40%      | 40 pts      |
| 📦 Producto       | 30%      | 30 pts      |
| **Total**         | **100%** | **100 pts** |

---

## 🧠 Conocimiento (30 pts)

### Conceptos Evaluados

| Concepto            | Puntos | Criterio                                   |
| ------------------- | ------ | ------------------------------------------ |
| Clases y objetos    | 6 pts  | Define correctamente clases con `__init__` |
| Atributos y métodos | 6 pts  | Distingue atributos de instancia y clase   |
| Herencia            | 6 pts  | Implementa herencia con `super()`          |
| Encapsulamiento     | 6 pts  | Usa propiedades y convenciones `_`/`__`    |
| Polimorfismo        | 6 pts  | Aplica duck typing y dunder methods        |

### Niveles de Desempeño

| Nivel        | Puntos | Descripción                                              |
| ------------ | ------ | -------------------------------------------------------- |
| Excelente    | 27-30  | Domina todos los conceptos, explica con ejemplos propios |
| Bueno        | 21-26  | Comprende conceptos, implementa correctamente            |
| Suficiente   | 15-20  | Comprende lo básico, necesita práctica                   |
| Insuficiente | 0-14   | No demuestra comprensión de POO                          |

---

## 💪 Desempeño (40 pts)

### Ejercicios Prácticos

| Ejercicio           | Puntos | Criterios                                  |
| ------------------- | ------ | ------------------------------------------ |
| 01 - Clases Básicas | 10 pts | Crea clases, instancias, llama métodos     |
| 02 - Herencia       | 10 pts | Extiende clases, usa super() correctamente |
| 03 - Propiedades    | 10 pts | Implementa @property con validación        |
| 04 - Dunder Methods | 10 pts | Implementa `__str__`, `__repr__`, `__eq__` |

### Criterios por Ejercicio

#### Ejercicio 01 - Clases Básicas (10 pts)

| Criterio                         | Puntos |
| -------------------------------- | ------ |
| Define clase con `__init__`      | 3 pts  |
| Crea atributos de instancia      | 2 pts  |
| Implementa métodos               | 3 pts  |
| Crea e interactúa con instancias | 2 pts  |

#### Ejercicio 02 - Herencia (10 pts)

| Criterio                        | Puntos |
| ------------------------------- | ------ |
| Crea clase hija                 | 3 pts  |
| Usa `super().__init__()`        | 3 pts  |
| Sobrescribe métodos             | 2 pts  |
| Añade atributos/métodos propios | 2 pts  |

#### Ejercicio 03 - Propiedades (10 pts)

| Criterio                         | Puntos |
| -------------------------------- | ------ |
| Usa `@property` correctamente    | 3 pts  |
| Implementa getter                | 2 pts  |
| Implementa setter con validación | 3 pts  |
| Maneja errores de validación     | 2 pts  |

#### Ejercicio 04 - Dunder Methods (10 pts)

| Criterio                      | Puntos |
| ----------------------------- | ------ |
| Implementa `__str__`          | 2 pts  |
| Implementa `__repr__`         | 2 pts  |
| Implementa `__eq__`           | 3 pts  |
| Implementa `__lt__` o similar | 3 pts  |

---

## 📦 Producto (30 pts)

### Proyecto: Sistema RPG

| Criterio                 | Puntos | Descripción                       |
| ------------------------ | ------ | --------------------------------- |
| **Estructura de clases** | 8 pts  | Jerarquía bien diseñada           |
| **Funcionalidad**        | 8 pts  | El sistema funciona correctamente |
| **Código limpio**        | 6 pts  | Legible, bien organizado          |
| **Documentación**        | 4 pts  | Docstrings en clases y métodos    |
| **Persistencia**         | 4 pts  | Guarda/carga con JSON             |

### Desglose Estructura de Clases (8 pts)

| Aspecto                                  | Puntos |
| ---------------------------------------- | ------ |
| Clase base `Character`                   | 2 pts  |
| Clases derivadas (Warrior, Mage, Archer) | 3 pts  |
| Clase `Item` e inventario                | 2 pts  |
| Relaciones entre clases                  | 1 pt   |

### Desglose Funcionalidad (8 pts)

| Aspecto               | Puntos |
| --------------------- | ------ |
| Crear personajes      | 2 pts  |
| Sistema de combate    | 3 pts  |
| Gestión de inventario | 2 pts  |
| Uso de items          | 1 pt   |

### Desglose Código Limpio (6 pts)

| Aspecto                                          | Puntos |
| ------------------------------------------------ | ------ |
| Nomenclatura consistente (PascalCase/snake_case) | 2 pts  |
| Type hints en métodos                            | 2 pts  |
| Sin código duplicado                             | 2 pts  |

### Desglose Documentación (4 pts)

| Aspecto                        | Puntos |
| ------------------------------ | ------ |
| Docstrings en clases           | 2 pts  |
| Docstrings en métodos públicos | 2 pts  |

### Desglose Persistencia (4 pts)

| Aspecto                         | Puntos |
| ------------------------------- | ------ |
| Método `to_dict()`              | 2 pts  |
| Método `from_dict()` o `load()` | 2 pts  |

---

## 📈 Escala de Calificación

| Calificación | Rango  | Descripción                               |
| ------------ | ------ | ----------------------------------------- |
| A            | 90-100 | Excelente - Dominio completo              |
| B            | 80-89  | Bueno - Comprensión sólida                |
| C            | 70-79  | Satisfactorio - Cumple requisitos mínimos |
| D            | 60-69  | Necesita mejorar                          |
| F            | 0-59   | No aprobado                               |

---

## ✅ Checklist de Entrega

### Ejercicios

- [ ] Ejercicio 01 completado y funcional
- [ ] Ejercicio 02 completado y funcional
- [ ] Ejercicio 03 completado y funcional
- [ ] Ejercicio 04 completado y funcional

### Proyecto

- [ ] Clase base `Character` implementada
- [ ] Al menos 3 clases derivadas
- [ ] Sistema de items/inventario
- [ ] Combate funcional
- [ ] Guardado/carga JSON
- [ ] Docstrings en todas las clases

### Calidad

- [ ] Código ejecuta sin errores
- [ ] Type hints en funciones/métodos
- [ ] Nomenclatura consistente
- [ ] Sin código comentado innecesario

---

## 🔗 Navegación

| Anterior                 | Inicio                         |
| ------------------------ | ------------------------------ |
| [← Semana 06](README.md) | [🏠 Bootcamp](../../README.md) |
