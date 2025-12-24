"""
CLI - Entry point para línea de comandos.

Uso:
    cli-utils <comando> [opciones]

Comandos disponibles:
    count-lines <archivo>     Cuenta líneas de un archivo
    search <patron> <archivo> Busca patrón en archivo
    list-files <directorio>   Lista archivos
    system-info               Muestra info del sistema
    python-info               Muestra info de Python
    clean-text                Limpia texto de stdin
    slug <texto>              Convierte a slug
    word-count <archivo>      Cuenta palabras en archivo
"""

import argparse
import sys

from .files import count_lines, search_in_file, list_files
from .text import clean_text, to_slug, word_count
from .system import get_system_info, get_python_info


def create_parser() -> argparse.ArgumentParser:
    """Crea y configura el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="cli-utils",
        description="Utilidades de línea de comandos para desarrollo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  cli-utils count-lines script.py
  cli-utils search "def " script.py
  cli-utils list-files . --ext .py
  cli-utils system-info
  cli-utils slug "Mi Título de Artículo"
  echo "texto  con  espacios" | cli-utils clean-text
        """,
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 1.0.0",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")
    
    # Comando: count-lines
    count_parser = subparsers.add_parser(
        "count-lines",
        help="Cuenta las líneas de un archivo",
    )
    count_parser.add_argument("file", help="Archivo a procesar")
    
    # Comando: search
    search_parser = subparsers.add_parser(
        "search",
        help="Busca un patrón en un archivo",
    )
    search_parser.add_argument("pattern", help="Patrón a buscar")
    search_parser.add_argument("file", help="Archivo donde buscar")
    
    # Comando: list-files
    list_parser = subparsers.add_parser(
        "list-files",
        help="Lista archivos en un directorio",
    )
    list_parser.add_argument("directory", help="Directorio a listar")
    list_parser.add_argument(
        "--ext", "-e",
        help="Filtrar por extensión (ej: .py)",
        default=None,
    )
    
    # Comando: system-info
    subparsers.add_parser(
        "system-info",
        help="Muestra información del sistema",
    )
    
    # Comando: python-info
    subparsers.add_parser(
        "python-info",
        help="Muestra información de Python",
    )
    
    # Comando: clean-text
    subparsers.add_parser(
        "clean-text",
        help="Limpia texto de stdin (elimina espacios extra)",
    )
    
    # Comando: slug
    slug_parser = subparsers.add_parser(
        "slug",
        help="Convierte texto a formato slug",
    )
    slug_parser.add_argument("text", nargs="+", help="Texto a convertir")
    
    # Comando: word-count
    wc_parser = subparsers.add_parser(
        "word-count",
        help="Cuenta palabras en un archivo o stdin",
    )
    wc_parser.add_argument(
        "file",
        nargs="?",
        help="Archivo a procesar (o stdin si no se especifica)",
    )
    
    return parser


def main() -> None:
    """Entry point principal."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        if args.command == "count-lines":
            result = count_lines(args.file)
            print(f"{result}")
        
        elif args.command == "search":
            results = search_in_file(args.file, args.pattern)
            if results:
                for line_num, line in results:
                    print(f"{line_num}: {line}")
            else:
                print(f"No se encontró '{args.pattern}' en {args.file}")
        
        elif args.command == "list-files":
            files = list_files(args.directory, args.ext)
            for f in files:
                print(f)
        
        elif args.command == "system-info":
            info = get_system_info()
            for key, value in info.items():
                print(f"{key}: {value}")
        
        elif args.command == "python-info":
            info = get_python_info()
            for key, value in info.items():
                print(f"{key}: {value}")
        
        elif args.command == "clean-text":
            # Leer de stdin
            text = sys.stdin.read()
            print(clean_text(text))
        
        elif args.command == "slug":
            text = " ".join(args.text)
            print(to_slug(text))
        
        elif args.command == "word-count":
            if args.file:
                with open(args.file, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                text = sys.stdin.read()
            
            stats = word_count(text)
            for key, value in stats.items():
                print(f"{key}: {value}")
        
        else:
            parser.print_help()
    
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"Error de permisos: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperación cancelada", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
