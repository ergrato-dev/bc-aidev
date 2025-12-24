# 🔗 Sistema de Imports en Python

![Python Path](../0-assets/02-python-path.svg)

## 🎯 Objetivos

- Dominar imports absolutos y relativos
- Entender `sys.path` y la resolución de módulos
- Organizar imports correctamente
- Evitar errores comunes de importación

---

## 📖 ¿Cómo Funciona el Import?

Cuando escribes `import module`, Python busca en este orden:

1. **Módulos built-in** (sys, os, etc.)
2. **Directorio actual** del script
3. **`PYTHONPATH`** (variable de entorno)
4. **Site-packages** (paquetes instalados con pip)

```python
import sys
print(sys.path)  # Lista de directorios donde Python busca módulos
```

### Ejemplo de `sys.path`

```python
[
    '/home/user/myproject',        # Directorio del script
    '/usr/lib/python3.11',         # Biblioteca estándar
    '/usr/lib/python3.11/lib-dynload',
    '/home/user/.local/lib/python3.11/site-packages',  # Paquetes usuario
    '/usr/lib/python3.11/site-packages',  # Paquetes sistema
]
```

---

## 📍 Imports Absolutos

Los **imports absolutos** usan la ruta completa desde la raíz del proyecto.

### Estructura de Ejemplo

```
myproject/
├── main.py
└── mypackage/
    ├── __init__.py
    ├── module_a.py
    ├── module_b.py
    └── subpkg/
        ├── __init__.py
        └── module_c.py
```

### Sintaxis

```python
# main.py (en la raíz del proyecto)

# Import completo del módulo
import mypackage.module_a

# Import selectivo
from mypackage.module_a import function_a

# Import de subpaquete
from mypackage.subpkg.module_c import ClassC

# Import con alias
import mypackage.module_a as mod_a
```

### Dentro de un Paquete

```python
# mypackage/module_b.py

# ✅ BIEN - Import absoluto (funciona desde cualquier lugar)
from mypackage.module_a import function_a
from mypackage.subpkg.module_c import ClassC
```

---

## 📍 Imports Relativos

Los **imports relativos** usan puntos (`.`) para indicar la ubicación relativa al módulo actual.

| Sintaxis | Significado    |
| -------- | -------------- |
| `.`      | Paquete actual |
| `..`     | Paquete padre  |
| `...`    | Paquete abuelo |

### Sintaxis

```python
# mypackage/module_b.py

# Import del mismo paquete
from . import module_a
from .module_a import function_a

# Import del paquete (desde __init__.py)
from . import some_function_from_init
```

```python
# mypackage/subpkg/module_c.py

# Import del paquete padre
from .. import module_a
from ..module_a import function_a

# Import de otro submódulo del padre
from ..module_b import function_b
```

### ⚠️ Restricciones Importantes

```python
# ❌ Los imports relativos NO funcionan en scripts ejecutados directamente
# Si ejecutas: python mypackage/module_b.py
# from .module_a import func  # Error: ImportError

# ✅ Funcionan cuando el módulo se importa como parte de un paquete
# python -m mypackage.module_b  # Funciona
# O desde main.py: from mypackage.module_b import ...
```

---

## 🆚 Absolutos vs Relativos

### Recomendación General

```python
# ✅ RECOMENDADO: Imports absolutos para código público
from mypackage.module_a import function_a
from mypackage.utils.helpers import helper

# ✅ ACEPTABLE: Imports relativos dentro de un paquete
from .module_a import function_a  # Mismo paquete
from ..utils import helper        # Paquete hermano
```

### Comparación

| Aspecto             | Absolutos                       | Relativos                      |
| ------------------- | ------------------------------- | ------------------------------ |
| **Claridad**        | ✅ Ruta completa visible        | ⚠️ Requiere conocer estructura |
| **Refactoring**     | ⚠️ Cambiar si mueves el paquete | ✅ No cambian si se mueve todo |
| **Ejecutar script** | ✅ Funciona siempre             | ❌ No funciona directamente    |
| **PEP 8**           | ✅ Recomendado                  | ⚠️ Solo dentro de paquetes     |

---

## 📋 Organización de Imports (PEP 8)

Los imports deben estar al inicio del archivo, organizados en grupos:

```python
"""Módulo de ejemplo con imports organizados."""

# =============================================================================
# 1. STANDARD LIBRARY (biblioteca estándar de Python)
# =============================================================================
import os
import sys
from pathlib import Path
from typing import Optional, List

# =============================================================================
# 2. THIRD PARTY (paquetes instalados con pip)
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# =============================================================================
# 3. LOCAL / FIRST PARTY (tu propio código)
# =============================================================================
from mypackage import config
from mypackage.utils import helpers
from .module_a import function_a

# Línea en blanco antes del código
def my_function():
    pass
```

### Herramienta: `isort`

```bash
# Instalar
pip install isort

# Ordenar imports automáticamente
isort my_file.py
isort mypackage/

# Verificar sin modificar
isort --check-only my_file.py
```

---

## 🔧 Modificar `sys.path`

A veces necesitas agregar rutas personalizadas:

```python
import sys
from pathlib import Path

# Agregar directorio al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ahora puedes importar desde project_root
from mypackage import module
```

### Variable de Entorno `PYTHONPATH`

```bash
# Linux/Mac
export PYTHONPATH="/path/to/myproject:$PYTHONPATH"

# Windows
set PYTHONPATH=C:\path\to\myproject;%PYTHONPATH%
```

> ⚠️ **Evita modificar `sys.path`** en código de producción. Es mejor instalar el paquete correctamente o usar `pip install -e .`

---

## 🔄 Imports Circulares

Los imports circulares ocurren cuando dos módulos se importan mutuamente.

### Problema

```python
# module_a.py
from module_b import func_b  # module_b necesita module_a

def func_a():
    return func_b() + 1

# module_b.py
from module_a import func_a  # module_a necesita module_b - ¡CIRCULAR!

def func_b():
    return func_a() + 1
```

### Soluciones

#### 1. Import dentro de la función (lazy import)

```python
# module_a.py
def func_a():
    from module_b import func_b  # Import cuando se necesita
    return func_b() + 1
```

#### 2. Reorganizar código

```python
# Mover código compartido a un tercer módulo
# shared.py
def shared_func():
    pass

# module_a.py
from shared import shared_func

# module_b.py
from shared import shared_func
```

#### 3. Import del módulo, no de la función

```python
# module_a.py
import module_b  # Solo importa el módulo

def func_a():
    return module_b.func_b() + 1  # Accede cuando se necesita
```

---

## 🎭 `if TYPE_CHECKING`

Para imports que solo se necesitan para type hints:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

# Este import solo se ejecuta durante el chequeo de tipos
# No causa imports circulares en runtime
if TYPE_CHECKING:
    from mypackage.heavy_module import HeavyClass

def process(obj: HeavyClass) -> None:
    """Procesa un objeto HeavyClass."""
    pass
```

---

## 📦 Import de Paquetes Instalados

### Con `pip install`

```bash
pip install requests
```

```python
import requests

response = requests.get("https://api.example.com")
```

### Con `pip install -e .` (Editable)

```bash
# Desde la raíz del proyecto con pyproject.toml
pip install -e .
```

Esto permite importar tu paquete desde cualquier lugar:

```python
# Funciona desde cualquier directorio
from mypackage import module_a
```

---

## ⚠️ Errores Comunes

### 1. `ModuleNotFoundError`

```python
# Error: ModuleNotFoundError: No module named 'mypackage'

# Causas:
# - El paquete no está en sys.path
# - Falta __init__.py
# - Nombre incorrecto
# - Entorno virtual no activado
```

### 2. `ImportError: attempted relative import`

```python
# Error al ejecutar directamente un script con imports relativos
# python mypackage/module.py

# Solución: Ejecutar como módulo
# python -m mypackage.module
```

### 3. `ImportError: cannot import name`

```python
# Error: ImportError: cannot import name 'func' from 'module'

# Causas:
# - La función no existe en el módulo
# - Import circular
# - Typo en el nombre
```

### 4. Shadowing de módulos

```python
# ❌ Nombrar archivo igual que módulo estándar
# random.py
import random  # ¡Importa tu archivo, no el módulo estándar!

# ✅ Usar nombres únicos
# my_random.py
import random  # Funciona correctamente
```

---

## 🧪 Depurar Imports

```python
# Ver de dónde viene un módulo
import mymodule
print(mymodule.__file__)

# Ver todos los módulos cargados
import sys
print(list(sys.modules.keys()))

# Ver el path de búsqueda
print(sys.path)

# Recargar un módulo (para desarrollo)
import importlib
importlib.reload(mymodule)
```

---

## ✅ Buenas Prácticas

1. **Prefiere imports absolutos** para claridad
2. **Organiza imports** según PEP 8 (stdlib, third-party, local)
3. **Usa `isort`** para mantener orden automáticamente
4. **Evita `import *`** excepto en casos específicos
5. **Evita modificar `sys.path`** en producción
6. **Usa `TYPE_CHECKING`** para imports de tipos
7. **Instala tu paquete** con `pip install -e .` para desarrollo
8. **No nombres archivos** igual que módulos de stdlib

---

## 📚 Resumen

| Concepto            | Descripción                             |
| ------------------- | --------------------------------------- |
| **Import absoluto** | `from package.module import func`       |
| **Import relativo** | `from .module import func`              |
| **`sys.path`**      | Lista de directorios donde Python busca |
| **PEP 8**           | stdlib → third-party → local            |
| **Import circular** | Dos módulos se importan mutuamente      |
| **`TYPE_CHECKING`** | Imports solo para type hints            |

---

## 🔗 Siguiente

Continúa con [04-entornos-virtuales.md](04-entornos-virtuales.md) para aprender a aislar dependencias.

---

_Volver a: [Semana 04](../README.md)_
