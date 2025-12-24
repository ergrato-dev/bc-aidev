"""
Módulo de información del sistema.

Proporciona funciones para obtener información sobre:
- Sistema operativo
- Entorno Python
"""

import os
import sys
import platform
from pathlib import Path


def get_system_info() -> dict:
    """
    Obtiene información del sistema operativo.

    Returns:
        Diccionario con:
        - os: nombre del sistema operativo
        - os_version: versión del SO
        - machine: arquitectura
        - processor: tipo de procesador
        - hostname: nombre del host
    """
    # TODO: Implementar
    # Hint: Usar platform.system(), platform.version(), etc.
    pass


def get_python_info() -> dict:
    """
    Obtiene información del entorno Python.

    Returns:
        Diccionario con:
        - version: versión de Python
        - executable: ruta al ejecutable
        - prefix: directorio base
        - is_venv: si está en entorno virtual
        - packages_dir: directorio de paquetes
    """
    # TODO: Implementar
    # Hint: Usar sys.version, sys.executable, sys.prefix
    pass


def is_virtual_env() -> bool:
    """
    Verifica si se está ejecutando en un entorno virtual.

    Returns:
        True si está en venv, False si no.
    """
    # TODO: Implementar
    # Hint: Comparar sys.prefix con sys.base_prefix
    pass


if __name__ == "__main__":
    print("=== System Module Test ===")
    # TODO: Agregar pruebas
