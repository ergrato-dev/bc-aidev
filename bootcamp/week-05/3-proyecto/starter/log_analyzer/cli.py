"""
Log Analyzer - CLI
==================
"""

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .parser import LogParser
from .filters import LogFilter
from .stats import StatsAnalyzer
from .exporters import export_to_json, export_to_csv, export_stats_to_json

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configura logging para la CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def create_parser() -> argparse.ArgumentParser:
    """Crea el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog='log_analyzer',
        description='Herramienta para analizar archivos de log'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Mostrar información de debug'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando: analyze
    analyze_parser = subparsers.add_parser('analyze', help='Analizar archivo de log')
    analyze_parser.add_argument('file', type=Path, help='Archivo de log a analizar')
    analyze_parser.add_argument(
        '--level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Filtrar por nivel de log'
    )
    analyze_parser.add_argument(
        '--min-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Nivel mínimo a mostrar'
    )
    analyze_parser.add_argument(
        '--pattern', '-p',
        help='Filtrar por patrón en mensaje'
    )
    analyze_parser.add_argument(
        '--limit', '-n',
        type=int,
        default=0,
        help='Limitar número de resultados (0 = todos)'
    )
    
    # Comando: stats
    stats_parser = subparsers.add_parser('stats', help='Mostrar estadísticas')
    stats_parser.add_argument('file', type=Path, help='Archivo de log')
    stats_parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Exportar estadísticas a archivo JSON'
    )
    
    # Comando: export
    export_parser = subparsers.add_parser('export', help='Exportar logs')
    export_parser.add_argument('file', type=Path, help='Archivo de log')
    export_parser.add_argument(
        '--format', '-f',
        choices=['json', 'csv'],
        default='json',
        help='Formato de exportación'
    )
    export_parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Archivo de salida'
    )
    export_parser.add_argument(
        '--level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Filtrar por nivel antes de exportar'
    )
    
    return parser


def cmd_analyze(args: argparse.Namespace) -> int:
    """Comando analyze."""
    # TODO: Implementar
    # 1. Parsear archivo
    # 2. Aplicar filtros si se especificaron
    # 3. Mostrar resultados
    pass


def cmd_stats(args: argparse.Namespace) -> int:
    """Comando stats."""
    # TODO: Implementar
    # 1. Parsear archivo
    # 2. Generar estadísticas
    # 3. Mostrar o exportar
    pass


def cmd_export(args: argparse.Namespace) -> int:
    """Comando export."""
    # TODO: Implementar
    # 1. Parsear archivo
    # 2. Aplicar filtros si se especificaron
    # 3. Exportar en formato elegido
    pass


def main() -> int:
    """Punto de entrada principal."""
    parser = create_parser()
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    commands = {
        'analyze': cmd_analyze,
        'stats': cmd_stats,
        'export': cmd_export,
    }
    
    try:
        return commands[args.command](args)
    except KeyboardInterrupt:
        print("\nOperación cancelada")
        return 130
    except Exception as e:
        logger.exception("Error inesperado")
        return 1


if __name__ == '__main__':
    sys.exit(main())
