# 📦 Semana 04: Módulos, Paquetes y Entornos Virtuales

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Crear y organizar módulos Python propios
- ✅ Estructurar paquetes con `__init__.py`
- ✅ Dominar el sistema de imports (absolutos y relativos)
- ✅ Crear y gestionar entornos virtuales con `venv`
- ✅ Manejar dependencias con `pip` y `requirements.txt`
- ✅ Entender el Python Path y la resolución de módulos
- ✅ Publicar paquetes básicos (estructura para PyPI)

---

## 📚 Requisitos Previos

- ✅ Semana 01: Fundamentos de Python
- ✅ Semana 02: Estructuras de Datos
- ✅ Semana 03: Programación Orientada a Objetos

---

## 🗂️ Estructura de la Semana

```
week-04/
├── README.md                    # Este archivo
├── rubrica-evaluacion.md        # Criterios de evaluación
├── 0-assets/                    # Diagramas SVG
│   ├── 01-estructura-paquete.svg
│   ├── 02-python-path.svg
│   ├── 03-entorno-virtual.svg
│   └── 04-flujo-dependencias.svg
├── 1-teoria/                    # Contenido teórico
│   ├── 01-modulos.md
│   ├── 02-paquetes.md
│   ├── 03-imports.md
│   └── 04-entornos-virtuales.md
├── 2-practicas/                 # Ejercicios guiados
│   ├── README.md
│   ├── ejercicio-01-modulos/
│   ├── ejercicio-02-paquetes/
│   ├── ejercicio-03-imports/
│   └── ejercicio-04-entornos/
├── 3-proyecto/                  # Proyecto integrador
│   ├── README.md
│   ├── starter/
│   └── .solution/
├── 4-recursos/                  # Material complementario
│   ├── ebooks-free/
│   ├── videografia/
│   └── webgrafia/
└── 5-glosario/                  # Términos clave
    └── README.md
```

---

## 📝 Contenidos

### 1. Teoría (1.5 horas)

| Archivo                                                       | Tema                     | Duración |
| ------------------------------------------------------------- | ------------------------ | -------- |
| [01-modulos.md](1-teoria/01-modulos.md)                       | Módulos Python           | 20 min   |
| [02-paquetes.md](1-teoria/02-paquetes.md)                     | Paquetes y `__init__.py` | 25 min   |
| [03-imports.md](1-teoria/03-imports.md)                       | Sistema de Imports       | 25 min   |
| [04-entornos-virtuales.md](1-teoria/04-entornos-virtuales.md) | Entornos y Dependencias  | 20 min   |

### 2. Prácticas (2.5 horas)

| Ejercicio                                          | Tema                          | Duración |
| -------------------------------------------------- | ----------------------------- | -------- |
| [ejercicio-01](2-practicas/ejercicio-01-modulos/)  | Crear módulos propios         | 30 min   |
| [ejercicio-02](2-practicas/ejercicio-02-paquetes/) | Estructurar paquetes          | 40 min   |
| [ejercicio-03](2-practicas/ejercicio-03-imports/)  | Imports absolutos y relativos | 35 min   |
| [ejercicio-04](2-practicas/ejercicio-04-entornos/) | Entornos virtuales y pip      | 45 min   |

### 3. Proyecto (2 horas)

| Proyecto                         | Descripción                                   |
| -------------------------------- | --------------------------------------------- |
| [CLI Utils Package](3-proyecto/) | Crear un paquete de utilidades CLI instalable |

---

## ⏱️ Distribución del Tiempo

| Actividad    | Tiempo      |
| ------------ | ----------- |
| 📖 Teoría    | 1.5 horas   |
| 💻 Prácticas | 2.5 horas   |
| 🏗️ Proyecto  | 2 horas     |
| **Total**    | **6 horas** |

---

## 📌 Entregables

1. **Ejercicios completados** (4 ejercicios)
2. **Proyecto**: Paquete `cli_utils` funcional e instalable
3. **Cuestionario**: Conceptos de módulos y entornos

---

## 🎓 Evaluación

| Tipo            | Peso | Descripción          |
| --------------- | ---- | -------------------- |
| 🧠 Conocimiento | 30%  | Cuestionario teórico |
| 💪 Desempeño    | 40%  | Ejercicios prácticos |
| 📦 Producto     | 30%  | Proyecto CLI Utils   |

Ver [rubrica-evaluacion.md](rubrica-evaluacion.md) para detalles.

---

## 🔑 Conceptos Clave

```python
# Módulo = archivo .py
import mymodule
from mymodule import function

# Paquete = carpeta con __init__.py
from mypackage import submodule
from mypackage.submodule import Class

# Imports relativos (dentro de paquetes)
from . import sibling_module
from ..parent import something

# Entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

---

## 💡 Tips de la Semana

> 🎯 **Siempre usa entornos virtuales** - Evita contaminar el Python del sistema

> 📁 **Un paquete = una responsabilidad** - Mantén tus paquetes enfocados

> 🔧 **requirements.txt con versiones** - `package==1.2.3` para reproducibilidad

> 🐍 **`if __name__ == '__main__':`** - Permite que un módulo sea importable y ejecutable

---

## 🔗 Navegación

| Anterior                                 | Índice                   | Siguiente                                  |
| ---------------------------------------- | ------------------------ | ------------------------------------------ |
| [← Semana 03: OOP](../week-03/README.md) | [Bootcamp](../README.md) | [Semana 05: NumPy →](../week-05/README.md) |

---

_Semana 04 de 36 · Módulo: Fundamentos (4/8)_
