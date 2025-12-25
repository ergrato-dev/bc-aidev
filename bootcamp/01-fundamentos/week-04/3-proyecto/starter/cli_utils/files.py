"""
Módulo de utilidades para archivos.

Proporciona funciones para:
- Contar líneas
- Buscar texto
- Listar archivos
"""

from pathlib import Path


def count_lines(filepath: str) -> int:
    """
    Cuenta las líneas de un archivo.

    Args:
        filepath: Ruta al archivo.

    Returns:
        Número de líneas en el archivo.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    # TODO: Implementar
    # Hint: Usar Path(filepath).read_text().splitlines()
    pass


def search_in_file(filepath: str, pattern: str) -> list[tuple[int, str]]:
    """
    Busca un patrón en un archivo.

    Args:
        filepath: Ruta al archivo.
        pattern: Texto a buscar.

    Returns:
        Lista de tuplas (número_línea, contenido_línea).
    """
    # TODO: Implementar
    # Hint: Enumerar líneas y filtrar las que contienen el patrón
    pass


def list_files(directory: str, extension: str | None = None) -> list[str]:
    """
    Lista archivos en un directorio.

    Args:
        directory: Ruta al directorio.
        extension: Filtrar por extensión (ej: ".py").

    Returns:
        Lista de rutas de archivos.
    """
    # TODO: Implementar
    # Hint: Usar Path(directory).iterdir() o .glob()
    pass


if __name__ == "__main__":
    # Código de prueba
    print("=== Files Module Test ===")
    # TODO: Agregar pruebas
