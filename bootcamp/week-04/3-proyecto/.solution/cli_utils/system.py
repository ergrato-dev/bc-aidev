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
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "N/A",
        "hostname": platform.node(),
    }


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
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "is_venv": is_virtual_env(),
        "path_count": len(sys.path),
    }


def is_virtual_env() -> bool:
    """
    Verifica si se está ejecutando en un entorno virtual.

    Returns:
        True si está en venv, False si no.
    """
    # En un venv, sys.prefix != sys.base_prefix
    return sys.prefix != sys.base_prefix


def get_env_variables(prefix: str | None = None) -> dict:
    """
    Obtiene variables de entorno.

    Args:
        prefix: Filtrar por prefijo (ej: "PYTHON").

    Returns:
        Diccionario de variables de entorno.
    """
    env_vars = dict(os.environ)
    
    if prefix:
        env_vars = {
            k: v for k, v in env_vars.items()
            if k.upper().startswith(prefix.upper())
        }
    
    return env_vars


if __name__ == "__main__":
    print("=== System Module Test ===")
    
    print("\nInformación del Sistema:")
    for key, value in get_system_info().items():
        print(f"  {key}: {value}")
    
    print("\nInformación de Python:")
    for key, value in get_python_info().items():
        print(f"  {key}: {value}")
    
    print("\nVariables PYTHON_*:")
    python_vars = get_env_variables("PYTHON")
    if python_vars:
        for key, value in python_vars.items():
            print(f"  {key}: {value}")
    else:
        print("  (ninguna encontrada)")
