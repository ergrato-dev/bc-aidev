# 📁 Paquetes en Python

![Estructura de Paquete](../0-assets/01-estructura-paquete.svg)

## 🎯 Objetivos

- Entender la diferencia entre módulo y paquete
- Crear paquetes con `__init__.py`
- Usar `__all__` para controlar exports
- Organizar código en subpaquetes

---

## 📖 ¿Qué es un Paquete?

Un **paquete** es una carpeta que contiene módulos Python y un archivo especial `__init__.py`.

```
mypackage/              ← Paquete (carpeta)
├── __init__.py         ← Hace que sea un paquete
├── module_a.py         ← Módulo
├── module_b.py         ← Módulo
└── subpackage/         ← Subpaquete (carpeta)
    ├── __init__.py
    └── module_c.py
```

### Módulo vs Paquete

| Concepto       | Descripción               | Ejemplo                |
| -------------- | ------------------------- | ---------------------- |
| **Módulo**     | Un archivo `.py`          | `calculator.py`        |
| **Paquete**    | Carpeta con `__init__.py` | `math_tools/`          |
| **Subpaquete** | Paquete dentro de paquete | `math_tools/geometry/` |

---

## 🔧 El Archivo `__init__.py`

El archivo `__init__.py` se ejecuta cuando importas el paquete. Puede estar vacío o contener código de inicialización.

### `__init__.py` Vacío

```python
# mypackage/__init__.py
# (archivo vacío - el paquete existe pero no expone nada directamente)
```

```python
# Uso
from mypackage import module_a  # Funciona
import mypackage.module_a       # Funciona
```

### `__init__.py` con Imports

```python
# mypackage/__init__.py
"""Paquete de utilidades matemáticas."""

from .module_a import function_a
from .module_b import ClassB

__version__ = "1.0.0"
```

```python
# Uso - Ahora puedes importar directamente del paquete
from mypackage import function_a, ClassB
print(mypackage.__version__)
```

---

## 📦 Ejemplo Práctico: Paquete `data_tools`

### Estructura

```
data_tools/
├── __init__.py
├── readers.py
├── writers.py
├── validators.py
└── transformers/
    ├── __init__.py
    ├── text.py
    └── numeric.py
```

### Implementación

```python
# data_tools/readers.py
"""Módulo para lectura de datos."""

import json
import csv
from pathlib import Path


def read_json(filepath: str) -> dict:
    """Lee un archivo JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def read_csv(filepath: str) -> list[dict]:
    """Lee un archivo CSV como lista de diccionarios."""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def read_text(filepath: str) -> str:
    """Lee un archivo de texto."""
    return Path(filepath).read_text()
```

```python
# data_tools/writers.py
"""Módulo para escritura de datos."""

import json
import csv
from pathlib import Path


def write_json(data: dict, filepath: str, indent: int = 2) -> None:
    """Escribe datos a un archivo JSON."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def write_csv(data: list[dict], filepath: str) -> None:
    """Escribe una lista de diccionarios a CSV."""
    if not data:
        return

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
```

```python
# data_tools/validators.py
"""Módulo para validación de datos."""

from pathlib import Path


def file_exists(filepath: str) -> bool:
    """Verifica si un archivo existe."""
    return Path(filepath).exists()


def is_valid_json(filepath: str) -> bool:
    """Verifica si un archivo es JSON válido."""
    import json
    try:
        with open(filepath) as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, FileNotFoundError):
        return False


def validate_required_fields(data: dict, required: list[str]) -> bool:
    """Valida que un diccionario tenga campos requeridos."""
    return all(field in data for field in required)
```

```python
# data_tools/__init__.py
"""
Data Tools - Paquete para manejo de datos.

Proporciona utilidades para:
- Leer archivos (JSON, CSV, texto)
- Escribir archivos
- Validar datos

Example:
    >>> from data_tools import read_json, write_json
    >>> data = read_json("config.json")
    >>> write_json(data, "backup.json")
"""

__version__ = "1.0.0"
__author__ = "AI Bootcamp"

# Exponer funciones principales en el nivel del paquete
from .readers import read_json, read_csv, read_text
from .writers import write_json, write_csv
from .validators import file_exists, is_valid_json

# Definir qué se exporta con "from data_tools import *"
__all__ = [
    # Readers
    "read_json",
    "read_csv",
    "read_text",
    # Writers
    "write_json",
    "write_csv",
    # Validators
    "file_exists",
    "is_valid_json",
]
```

### Uso del Paquete

```python
# Opción 1: Import directo del paquete
from data_tools import read_json, write_json

data = read_json("config.json")
write_json(data, "backup.json")

# Opción 2: Import de módulos específicos
from data_tools.readers import read_csv
from data_tools.validators import file_exists

if file_exists("data.csv"):
    records = read_csv("data.csv")

# Opción 3: Import del paquete completo
import data_tools

print(data_tools.__version__)  # "1.0.0"
data = data_tools.read_json("config.json")
```

---

## 🎛️ `__all__` - Controlando Exports

`__all__` es una lista que define qué se exporta cuando alguien hace `from package import *`.

```python
# mypackage/__init__.py

from .module_a import func1, func2, _private_func
from .module_b import ClassA, ClassB

# Solo estas se exportan con "import *"
__all__ = ["func1", "ClassA"]

# func2, ClassB, _private_func NO se exportan con *
# pero SÍ se pueden importar explícitamente
```

```python
# Uso
from mypackage import *  # Solo importa func1 y ClassA

from mypackage import func2  # Esto SÍ funciona (import explícito)
```

### Buenas Prácticas con `__all__`

```python
# ✅ BIEN - __all__ explícito y organizado
__all__ = [
    # Funciones principales
    "load_data",
    "save_data",
    # Clases
    "DataLoader",
    "DataWriter",
    # Constantes
    "DEFAULT_FORMAT",
]

# ❌ MAL - No definir __all__ en paquetes públicos
# (cualquier cosa con from x import * será confusa)
```

---

## 📁 Subpaquetes

Los paquetes pueden contener otros paquetes (subpaquetes).

```python
# data_tools/transformers/__init__.py
"""Subpaquete de transformadores."""

from .text import clean_text, normalize
from .numeric import scale, normalize_values

__all__ = ["clean_text", "normalize", "scale", "normalize_values"]
```

```python
# data_tools/transformers/text.py
"""Transformadores de texto."""

import re


def clean_text(text: str) -> str:
    """Limpia un texto eliminando caracteres especiales."""
    return re.sub(r'[^\w\s]', '', text)


def normalize(text: str) -> str:
    """Normaliza texto a minúsculas sin espacios extra."""
    return ' '.join(text.lower().split())
```

```python
# data_tools/transformers/numeric.py
"""Transformadores numéricos."""


def scale(values: list[float], factor: float) -> list[float]:
    """Escala valores por un factor."""
    return [v * factor for v in values]


def normalize_values(values: list[float]) -> list[float]:
    """Normaliza valores al rango [0, 1]."""
    min_val, max_val = min(values), max(values)
    range_val = max_val - min_val
    if range_val == 0:
        return [0.0] * len(values)
    return [(v - min_val) / range_val for v in values]
```

### Uso de Subpaquetes

```python
# Desde el subpaquete directamente
from data_tools.transformers import clean_text, scale

# Desde módulos específicos
from data_tools.transformers.text import normalize
from data_tools.transformers.numeric import normalize_values
```

---

## 🆕 Namespace Packages (Python 3.3+)

Desde Python 3.3, los paquetes **no requieren** `__init__.py` (namespace packages). Sin embargo, se recomienda seguir usándolo para:

- Código de inicialización
- Exponer API pública
- Definir `__all__`
- Mantener compatibilidad

```
# Namespace package (sin __init__.py)
mypackage/
├── module_a.py
└── module_b.py

# Regular package (con __init__.py) - RECOMENDADO
mypackage/
├── __init__.py  ← Recomendado
├── module_a.py
└── module_b.py
```

---

## 🏗️ Estructura de Paquete Profesional

```
myproject/
├── pyproject.toml          # Configuración del proyecto
├── README.md               # Documentación
├── LICENSE                 # Licencia
├── src/                    # Código fuente (src layout)
│   └── mypackage/
│       ├── __init__.py
│       ├── core.py
│       ├── utils.py
│       └── cli.py
├── tests/                  # Tests
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
└── docs/                   # Documentación adicional
    └── usage.md
```

### `pyproject.toml` Básico

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "mypackage"
version = "1.0.0"
description = "Mi paquete de ejemplo"
readme = "README.md"
requires-python = ">=3.10"
authors = [
    {name = "Tu Nombre", email = "tu@email.com"}
]
dependencies = []

[project.optional-dependencies]
dev = ["pytest", "black", "mypy"]

[project.scripts]
mycommand = "mypackage.cli:main"
```

---

## ⚠️ Errores Comunes

### 1. Olvidar `__init__.py`

```
# ❌ Sin __init__.py - imports pueden fallar
mypackage/
├── module_a.py
└── module_b.py

# ✅ Con __init__.py
mypackage/
├── __init__.py  # Aunque esté vacío
├── module_a.py
└── module_b.py
```

### 2. Imports circulares en `__init__.py`

```python
# ❌ MAL - Si module_a importa algo de __init__.py
# __init__.py
from .module_a import func  # module_a aún no cargó

# ✅ BIEN - Lazy imports o reorganizar
```

### 3. Nombres de paquetes inválidos

```
# ❌ MAL
my-package/      # Guiones no válidos
123package/      # No empezar con número
my package/      # Espacios no válidos

# ✅ BIEN
my_package/
package123/
mypackage/
```

---

## ✅ Buenas Prácticas

1. **Siempre crear `__init__.py`** aunque esté vacío
2. **Exponer API pública** en `__init__.py`
3. **Definir `__all__`** para paquetes públicos
4. **Usar nombres snake_case** para paquetes y módulos
5. **Un paquete = una responsabilidad** cohesiva
6. **Documentar** el paquete con docstring en `__init__.py`
7. **Versionar** con `__version__` en `__init__.py`

---

## 📚 Resumen

| Concepto              | Descripción                             |
| --------------------- | --------------------------------------- |
| **Paquete**           | Carpeta con `__init__.py`               |
| **`__init__.py`**     | Inicialización y API pública            |
| **`__all__`**         | Control de `from pkg import *`          |
| **Subpaquete**        | Paquete dentro de paquete               |
| **Namespace package** | Paquete sin `__init__.py` (Python 3.3+) |

---

## 🔗 Siguiente

Continúa con [03-imports.md](03-imports.md) para dominar el sistema de imports de Python.

---

_Volver a: [Semana 04](../README.md)_
