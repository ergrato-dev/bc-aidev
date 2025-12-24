# 📦 Módulos en Python

![Estructura de Módulo](../0-assets/01-estructura-paquete.svg)

## 🎯 Objetivos

- Entender qué es un módulo en Python
- Crear módulos propios reutilizables
- Usar `__name__` y `__file__`
- Documentar módulos con docstrings

---

## 📖 ¿Qué es un Módulo?

Un **módulo** es simplemente un archivo `.py` que contiene código Python (funciones, clases, variables) que puede ser importado y reutilizado.

```python
# math_utils.py - Este archivo ES un módulo
"""Utilidades matemáticas para el bootcamp."""

PI = 3.14159

def circle_area(radius: float) -> float:
    """Calcula el área de un círculo."""
    return PI * radius ** 2

def circle_perimeter(radius: float) -> float:
    """Calcula el perímetro de un círculo."""
    return 2 * PI * radius
```

### Usar el Módulo

```python
# main.py
import math_utils

area = math_utils.circle_area(5)
print(f"Área: {area}")  # Área: 78.53975
```

---

## 🔧 Formas de Importar

### 1. Import Completo

```python
import math_utils

# Acceso con prefijo
result = math_utils.circle_area(5)
print(math_utils.PI)
```

### 2. Import Selectivo

```python
from math_utils import circle_area, PI

# Acceso directo (sin prefijo)
result = circle_area(5)
print(PI)
```

### 3. Import con Alias

```python
import math_utils as mu

result = mu.circle_area(5)
```

```python
from math_utils import circle_area as area

result = area(5)
```

### 4. Import Todo (⚠️ Evitar)

```python
from math_utils import *  # Importa todo - NO recomendado

# Contamina el namespace
result = circle_area(5)
```

> ⚠️ **Evita `import *`** - Hace difícil saber de dónde vienen las funciones y puede causar colisiones de nombres.

---

## 🎭 `__name__` - El Guardián del Módulo

La variable especial `__name__` contiene:

- `"__main__"` → si el archivo se ejecuta directamente
- `"nombre_modulo"` → si el archivo se importa

### Patrón Fundamental

```python
# calculator.py
"""Módulo de calculadora con funciones básicas."""

def add(a: float, b: float) -> float:
    """Suma dos números."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Resta dos números."""
    return a - b

def multiply(a: float, b: float) -> float:
    """Multiplica dos números."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide dos números."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


# Este bloque SOLO se ejecuta si corres: python calculator.py
# NO se ejecuta si haces: import calculator
if __name__ == "__main__":
    # Código de prueba / demostración
    print("=== Calculator Demo ===")
    print(f"5 + 3 = {add(5, 3)}")
    print(f"5 - 3 = {subtract(5, 3)}")
    print(f"5 * 3 = {multiply(5, 3)}")
    print(f"5 / 3 = {divide(5, 3):.2f}")
```

### ¿Por qué es Importante?

```python
# Si ejecutas: python calculator.py
# Output:
# === Calculator Demo ===
# 5 + 3 = 8
# ...

# Si importas desde otro archivo:
import calculator
result = calculator.add(10, 20)  # Funciona sin ejecutar el demo
```

---

## 📂 `__file__` - Ubicación del Módulo

La variable `__file__` contiene la ruta al archivo del módulo.

```python
# file_utils.py
"""Utilidades para manejo de archivos."""

from pathlib import Path

# Obtener directorio del módulo (útil para rutas relativas)
MODULE_DIR = Path(__file__).parent
DATA_DIR = MODULE_DIR / "data"

def read_data_file(filename: str) -> str:
    """Lee un archivo del directorio data/."""
    file_path = DATA_DIR / filename
    return file_path.read_text()

def get_module_info() -> dict:
    """Retorna información del módulo."""
    return {
        "file": __file__,
        "name": __name__,
        "dir": str(MODULE_DIR),
    }


if __name__ == "__main__":
    info = get_module_info()
    for key, value in info.items():
        print(f"{key}: {value}")
```

---

## 📝 Documentación de Módulos

### Module Docstring

El docstring del módulo va al inicio del archivo:

```python
"""
Módulo de utilidades para procesamiento de texto.

Este módulo proporciona funciones para:
- Limpiar y normalizar texto
- Contar palabras y caracteres
- Formatear strings

Example:
    >>> from text_utils import word_count
    >>> word_count("Hola mundo")
    2

Author: Tu Nombre
Version: 1.0.0
"""

def word_count(text: str) -> int:
    """Cuenta las palabras en un texto."""
    return len(text.split())
```

### Acceder a la Documentación

```python
import text_utils

# Ver docstring del módulo
print(text_utils.__doc__)

# Ver docstring de una función
print(text_utils.word_count.__doc__)

# Usar help()
help(text_utils)
help(text_utils.word_count)
```

---

## 🏗️ Estructura de un Buen Módulo

```python
"""
module_name.py - Descripción breve del módulo.

Descripción más detallada si es necesario.

Example:
    >>> from module_name import main_function
    >>> main_function()
"""

# =============================================================================
# IMPORTS
# =============================================================================
# 1. Standard library
import os
from pathlib import Path

# 2. Third party (si aplica)
# import numpy as np

# 3. Local imports (si aplica)
# from . import helper

# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_VALUE = 100
MODULE_DIR = Path(__file__).parent

# =============================================================================
# CLASSES
# =============================================================================
class MyClass:
    """Descripción de la clase."""
    pass

# =============================================================================
# FUNCTIONS
# =============================================================================
def public_function(param: str) -> str:
    """
    Función pública del módulo.

    Args:
        param: Descripción del parámetro.

    Returns:
        Descripción del valor retornado.
    """
    return _helper_function(param)

def _helper_function(param: str) -> str:
    """Función privada (convención con _)."""
    return param.upper()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Código de demostración o testing
    result = public_function("test")
    print(f"Result: {result}")
```

---

## 🔍 Módulos de la Biblioteca Estándar

Python incluye una rica biblioteca de módulos listos para usar:

```python
# Módulos comunes
import os           # Interacción con el sistema operativo
import sys          # Configuración del intérprete
import json         # Manejo de JSON
import csv          # Manejo de CSV
import datetime     # Fechas y horas
import random       # Números aleatorios
import re           # Expresiones regulares
import pathlib      # Manejo de rutas (moderno)
import collections  # Estructuras de datos adicionales
import itertools    # Herramientas de iteración
import functools    # Herramientas funcionales
```

### Ejemplo: Módulo `pathlib`

```python
from pathlib import Path

# Crear rutas de forma multiplataforma
data_dir = Path("data")
file_path = data_dir / "users.json"

# Verificar existencia
if file_path.exists():
    content = file_path.read_text()

# Crear directorio
data_dir.mkdir(exist_ok=True)

# Listar archivos
for py_file in Path(".").glob("*.py"):
    print(py_file.name)
```

---

## ⚠️ Errores Comunes

### 1. Nombre de módulo = nombre de stdlib

```python
# ❌ MAL - No nombres tu archivo "random.py"
# random.py
import random  # ¡Importa tu archivo, no el módulo estándar!
```

### 2. Import circular

```python
# ❌ MAL - Dependencia circular
# a.py
from b import func_b  # b necesita a, a necesita b

# b.py
from a import func_a  # Error!
```

### 3. Olvidar `if __name__`

```python
# ❌ MAL - Código que se ejecuta al importar
# bad_module.py
print("Cargando módulo...")  # Se imprime al hacer import
result = heavy_computation()  # Se ejecuta innecesariamente
```

---

## ✅ Buenas Prácticas

1. **Nombres descriptivos**: `file_utils.py` mejor que `utils.py`
2. **Un módulo, una responsabilidad**: No mezclar lógica no relacionada
3. **Siempre usar `if __name__ == "__main__":`** para código ejecutable
4. **Documentar con docstrings**: Módulo y funciones públicas
5. **Usar `_` para funciones privadas**: `_helper()` indica uso interno
6. **Imports al inicio**: Organizados por tipo
7. **Evitar `import *`**: Siempre importar explícitamente

---

## 📚 Resumen

| Concepto            | Descripción                                      |
| ------------------- | ------------------------------------------------ |
| **Módulo**          | Archivo `.py` con código reutilizable            |
| **`import`**        | Cargar un módulo completo                        |
| **`from...import`** | Cargar elementos específicos                     |
| **`__name__`**      | `"__main__"` si se ejecuta, nombre si se importa |
| **`__file__`**      | Ruta al archivo del módulo                       |
| **Docstring**       | Documentación al inicio del módulo/función       |

---

## 🔗 Siguiente

Continúa con [02-paquetes.md](02-paquetes.md) para aprender a organizar múltiples módulos en paquetes.

---

_Volver a: [Semana 04](../README.md)_
