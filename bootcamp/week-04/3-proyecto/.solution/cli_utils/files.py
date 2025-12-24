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
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
    
    content = path.read_text(encoding="utf-8")
    return len(content.splitlines())


def search_in_file(filepath: str, pattern: str) -> list[tuple[int, str]]:
    """
    Busca un patrón en un archivo.

    Args:
        filepath: Ruta al archivo.
        pattern: Texto a buscar.

    Returns:
        Lista de tuplas (número_línea, contenido_línea).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
    
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if pattern in line:
                results.append((line_num, line.rstrip()))
    
    return results


def list_files(directory: str, extension: str | None = None) -> list[str]:
    """
    Lista archivos en un directorio.

    Args:
        directory: Ruta al directorio.
        extension: Filtrar por extensión (ej: ".py").

    Returns:
        Lista de rutas de archivos.
    """
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {directory}")
    
    if not path.is_dir():
        raise NotADirectoryError(f"No es un directorio: {directory}")
    
    if extension:
        # Asegurar que la extensión empiece con punto
        if not extension.startswith("."):
            extension = f".{extension}"
        files = path.glob(f"*{extension}")
    else:
        files = (f for f in path.iterdir() if f.is_file())
    
    return sorted(str(f) for f in files)


if __name__ == "__main__":
    print("=== Files Module Test ===")
    
    # Test count_lines (usando este mismo archivo)
    lines = count_lines(__file__)
    print(f"Este archivo tiene {lines} líneas")
    
    # Test search_in_file
    results = search_in_file(__file__, "def ")
    print(f"\nFunciones encontradas:")
    for num, line in results:
        print(f"  L{num}: {line.strip()}")
    
    # Test list_files
    from pathlib import Path
    parent = str(Path(__file__).parent)
    py_files = list_files(parent, ".py")
    print(f"\nArchivos .py en {parent}:")
    for f in py_files:
        print(f"  {f}")
