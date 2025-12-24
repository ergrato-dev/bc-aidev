"""
CLI - Entry point para línea de comandos.

Uso:
    cli-utils <comando> [opciones]

Comandos disponibles:
    count-lines <archivo>     Cuenta líneas de un archivo
    search <patron> <archivo> Busca patrón en archivo
    system-info               Muestra info del sistema
    python-info               Muestra info de Python
    clean-text                Limpia texto de stdin
    slug <texto>              Convierte a slug
"""

import argparse
import sys

# TODO: Importar funciones de los módulos
# from .files import count_lines, search_in_file
# from .text import clean_text, to_slug
# from .system import get_system_info, get_python_info


def create_parser() -> argparse.ArgumentParser:
    """Crea y configura el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="cli-utils",
        description="Utilidades de línea de comandos",
    )
    
    # TODO: Agregar subparsers para cada comando
    # subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")
    
    # Ejemplo de subparser:
    # count_parser = subparsers.add_parser("count-lines", help="Cuenta líneas")
    # count_parser.add_argument("file", help="Archivo a procesar")
    
    return parser


def main() -> None:
    """Entry point principal."""
    parser = create_parser()
    args = parser.parse_args()
    
    # TODO: Implementar lógica según el comando
    # if args.command == "count-lines":
    #     result = count_lines(args.file)
    #     print(f"Líneas: {result}")
    # elif args.command == "system-info":
    #     info = get_system_info()
    #     for key, value in info.items():
    #         print(f"{key}: {value}")
    # else:
    #     parser.print_help()
    
    print("CLI Utils - TODO: Implementar comandos")
    parser.print_help()


if __name__ == "__main__":
    main()
