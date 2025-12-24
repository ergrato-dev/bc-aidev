# 📖 Glosario - Semana 04

Términos técnicos clave de módulos, paquetes y entornos virtuales, ordenados alfabéticamente.

---

## A

### Absolute Import

Import que usa la ruta completa desde la raíz del proyecto.

```python
from mypackage.submodule import function
```

### Activate

Comando para activar un entorno virtual y usar su Python/pip.

```bash
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### `__all__`

Lista que define qué se exporta cuando se usa `from module import *`.

```python
__all__ = ["public_func", "PublicClass"]
```

### argparse

Módulo de la biblioteca estándar para crear interfaces de línea de comandos.

```python
import argparse
parser = argparse.ArgumentParser()
```

---

## B

### Built-in Module

Módulo compilado directamente en el intérprete de Python (ej: `sys`, `os`).

---

## D

### Deactivate

Comando para salir de un entorno virtual activo.

```bash
deactivate
```

### Dependency

Paquete externo que tu código necesita para funcionar (ej: `requests`).

---

## E

### Editable Install

Instalación de un paquete en modo desarrollo que refleja cambios sin reinstalar.

```bash
pip install -e .
```

### Entry Point

Punto de entrada que permite ejecutar código como comando de terminal.

```toml
[project.scripts]
mycommand = "mypackage.cli:main"
```

---

## F

### `__file__`

Variable especial que contiene la ruta al archivo del módulo actual.

```python
from pathlib import Path
MODULE_DIR = Path(__file__).parent
```

### Freeze

Comando pip que lista todos los paquetes instalados con sus versiones.

```bash
pip freeze > requirements.txt
```

---

## I

### Import

Sentencia que carga código de otro módulo para usarlo.

```python
import module
from module import function
```

### Import Circular

Error cuando dos módulos intentan importarse mutuamente.

### `__init__.py`

Archivo que convierte una carpeta en un paquete Python.

---

## M

### Module

Archivo `.py` que contiene código Python reutilizable.

### MRO (Method Resolution Order)

En contexto de imports, el orden en que Python busca módulos en `sys.path`.

---

## N

### `__name__`

Variable especial: `"__main__"` si se ejecuta directamente, nombre del módulo si se importa.

```python
if __name__ == "__main__":
    main()
```

### Namespace Package

Paquete sin `__init__.py` que puede distribuirse en múltiples directorios (Python 3.3+).

---

## P

### Package

Carpeta que contiene módulos Python y (normalmente) un `__init__.py`.

### pip

Gestor de paquetes de Python para instalar dependencias.

```bash
pip install requests
pip install -r requirements.txt
```

### PyPI (Python Package Index)

Repositorio oficial de paquetes Python: https://pypi.org/

### `pyproject.toml`

Archivo de configuración moderno para proyectos Python (reemplaza setup.py).

```toml
[project]
name = "mypackage"
version = "1.0.0"
```

### PYTHONPATH

Variable de entorno que agrega directorios a `sys.path`.

```bash
export PYTHONPATH="/my/path:$PYTHONPATH"
```

---

## R

### Relative Import

Import usando puntos para indicar posición relativa dentro de un paquete.

```python
from . import sibling        # Mismo nivel
from ..parent import func    # Nivel superior
```

### requirements.txt

Archivo que lista las dependencias de un proyecto.

```
requests==2.31.0
pandas>=2.0.0
numpy
```

---

## S

### Script

Archivo Python diseñado para ser ejecutado directamente.

### setuptools

Biblioteca para empaquetar y distribuir proyectos Python.

### Site-packages

Directorio donde pip instala los paquetes.

```
.venv/lib/python3.11/site-packages/
```

### Subpackage

Paquete dentro de otro paquete.

```
package/
└── subpackage/
    └── __init__.py
```

### `sys.path`

Lista de directorios donde Python busca módulos al importar.

```python
import sys
print(sys.path)
```

---

## V

### venv

Módulo de la biblioteca estándar para crear entornos virtuales.

```bash
python -m venv .venv
```

### Virtual Environment

Entorno Python aislado con su propio conjunto de paquetes instalados.

### `__version__`

Convención para definir la versión de un paquete en `__init__.py`.

```python
__version__ = "1.0.0"
```

---

## W

### Wheel

Formato de distribución de paquetes Python (archivos `.whl`).

---

## Tabla de Comandos pip

| Comando                           | Descripción                     |
| --------------------------------- | ------------------------------- |
| `pip install pkg`                 | Instalar paquete                |
| `pip install pkg==1.0.0`          | Instalar versión específica     |
| `pip install -r requirements.txt` | Instalar desde archivo          |
| `pip install -e .`                | Instalar en modo editable       |
| `pip uninstall pkg`               | Desinstalar paquete             |
| `pip list`                        | Listar paquetes instalados      |
| `pip freeze`                      | Listar con formato requirements |
| `pip show pkg`                    | Info de un paquete              |
| `pip search`                      | (Deshabilitado en PyPI)         |

---

## Tabla de Import Styles

| Estilo               | Ejemplo                 | Uso                   |
| -------------------- | ----------------------- | --------------------- |
| Import completo      | `import os`             | Acceso como `os.path` |
| Import selectivo     | `from os import path`   | Acceso directo `path` |
| Import con alias     | `import numpy as np`    | Acceso como `np`      |
| Relativo mismo nivel | `from . import module`  | Dentro de paquete     |
| Relativo padre       | `from .. import module` | Subir nivel           |

---

## Estructura de Paquete

```
mypackage/
├── __init__.py      # Hace que sea paquete
├── module.py        # Módulo principal
├── _private.py      # Módulo privado (convención)
└── subpkg/          # Subpaquete
    ├── __init__.py
    └── helper.py
```

---

_Volver a: [Semana 04](../README.md)_
