"""
Módulo de utilidades para texto.

Proporciona funciones para:
- Limpiar texto
- Contar palabras
- Generar slugs
"""

import re


def clean_text(text: str) -> str:
    """
    Limpia un texto eliminando espacios extra.

    Args:
        text: Texto a limpiar.

    Returns:
        Texto limpio sin espacios extra.

    Example:
        >>> clean_text("  hola   mundo  ")
        "hola mundo"
    """
    # TODO: Implementar
    # Hint: Usar split() y join()
    pass


def word_count(text: str) -> dict:
    """
    Cuenta palabras y genera estadísticas.

    Args:
        text: Texto a analizar.

    Returns:
        Diccionario con estadísticas:
        - words: número de palabras
        - chars: número de caracteres
        - lines: número de líneas
    """
    # TODO: Implementar
    pass


def to_slug(text: str) -> str:
    """
    Convierte texto a formato slug (URL-friendly).

    Args:
        text: Texto a convertir.

    Returns:
        Slug en minúsculas con guiones.

    Example:
        >>> to_slug("Mi Título de Artículo!")
        "mi-titulo-de-articulo"
    """
    # TODO: Implementar
    # Hint: lower(), re.sub() para caracteres especiales
    pass


if __name__ == "__main__":
    print("=== Text Module Test ===")
    # TODO: Agregar pruebas
