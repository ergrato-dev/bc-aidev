"""
Módulo de utilidades para texto.

Proporciona funciones para:
- Limpiar texto
- Contar palabras
- Generar slugs
"""

import re
import unicodedata


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
    return " ".join(text.split())


def word_count(text: str) -> dict:
    """
    Cuenta palabras y genera estadísticas.

    Args:
        text: Texto a analizar.

    Returns:
        Diccionario con estadísticas:
        - words: número de palabras
        - chars: número de caracteres
        - chars_no_spaces: caracteres sin espacios
        - lines: número de líneas
    """
    lines = text.splitlines()
    words = text.split()
    
    return {
        "words": len(words),
        "chars": len(text),
        "chars_no_spaces": len(text.replace(" ", "").replace("\n", "")),
        "lines": len(lines) if text else 0,
    }


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
    # Normalizar unicode (convertir acentos)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Reemplazar espacios y caracteres especiales con guiones
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    
    # Eliminar guiones al inicio y final
    return text.strip("-")


def truncate(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Trunca texto a una longitud máxima.

    Args:
        text: Texto a truncar.
        max_length: Longitud máxima.
        suffix: Sufijo a agregar si se trunca.

    Returns:
        Texto truncado con sufijo si excede max_length.
    """
    if len(text) <= max_length:
        return text
    
    return text[: max_length - len(suffix)].rstrip() + suffix


if __name__ == "__main__":
    print("=== Text Module Test ===")
    
    # Test clean_text
    dirty = "  hola    mundo   con   espacios  "
    print(f"Original: '{dirty}'")
    print(f"Limpio: '{clean_text(dirty)}'")
    
    # Test word_count
    sample = "Hola mundo.\nEsta es una prueba."
    stats = word_count(sample)
    print(f"\nEstadísticas de '{sample}':")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test to_slug
    titles = [
        "Mi Título de Artículo!",
        "¿Cómo crear un SLUG?",
        "Python 3.11 -- Nuevas características",
    ]
    print("\nSlugs:")
    for title in titles:
        print(f"  '{title}' -> '{to_slug(title)}'")
    
    # Test truncate
    long_text = "Este es un texto muy largo que necesita ser truncado"
    print(f"\nTruncar: '{truncate(long_text, 30)}'")
