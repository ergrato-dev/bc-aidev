"""
CLI Utils - Paquete de utilidades de línea de comandos.

Proporciona herramientas para:
- Operaciones con archivos
- Procesamiento de texto
- Información del sistema

Example:
    >>> from cli_utils import count_lines, clean_text
    >>> count_lines("archivo.py")
    42
"""

__version__ = "1.0.0"
__author__ = "AI Bootcamp Student"

from .files import count_lines, search_in_file, list_files
from .text import clean_text, word_count, to_slug
from .system import get_system_info, get_python_info, is_virtual_env

__all__ = [
    "count_lines",
    "search_in_file",
    "list_files",
    "clean_text",
    "word_count",
    "to_slug",
    "get_system_info",
    "get_python_info",
    "is_virtual_env",
]
