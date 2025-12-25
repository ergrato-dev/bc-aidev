"""
Log Analyzer - Parser de logs
=============================
"""

import re
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .exceptions import ParseError, FileFormatError

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Representa una entrada de log parseada."""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    raw_line: str
    line_number: int


class LogParser:
    """Parser para archivos de log."""
    
    # Patrón: 2025-12-23 10:30:45 | INFO | app.main | Mensaje
    PATTERN = re.compile(
        r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*\|\s*'
        r'(\w+)\s*\|\s*'
        r'([\w.]+)\s*\|\s*'
        r'(.+)$'
    )
    
    VALID_LEVELS = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    
    def __init__(self, strict: bool = False):
        """
        Inicializa el parser.
        
        Args:
            strict: Si True, lanza excepción en líneas inválidas.
                   Si False, las ignora con warning.
        """
        self.strict = strict
        self._errors: list[ParseError] = []
    
    @property
    def errors(self) -> list[ParseError]:
        """Errores encontrados durante el parsing."""
        return self._errors.copy()
    
    def parse_file(self, filepath: Path | str) -> list[LogEntry]:
        """
        Parsea un archivo de log completo.
        
        Args:
            filepath: Ruta al archivo de log.
            
        Returns:
            Lista de entradas parseadas.
            
        Raises:
            FileNotFoundError: Si el archivo no existe.
            FileFormatError: Si el archivo está vacío o no tiene entradas válidas.
        """
        # TODO: Implementar
        # 1. Convertir filepath a Path
        # 2. Verificar que existe
        # 3. Leer y parsear cada línea
        # 4. Retornar lista de LogEntry
        pass
    
    def parse_file_streaming(self, filepath: Path | str) -> Iterator[LogEntry]:
        """
        Parsea un archivo de log en modo streaming (para archivos grandes).
        
        Args:
            filepath: Ruta al archivo de log.
            
        Yields:
            LogEntry por cada línea válida.
        """
        # TODO: Implementar
        # Similar a parse_file pero usando yield
        pass
    
    def parse_line(self, line: str, line_number: int) -> LogEntry | None:
        """
        Parsea una línea de log individual.
        
        Args:
            line: Línea de texto a parsear.
            line_number: Número de línea (para reportar errores).
            
        Returns:
            LogEntry si la línea es válida, None si no.
            
        Raises:
            ParseError: Si strict=True y la línea es inválida.
        """
        # TODO: Implementar
        # 1. Hacer strip de la línea
        # 2. Ignorar líneas vacías
        # 3. Aplicar regex PATTERN
        # 4. Validar level
        # 5. Parsear timestamp
        # 6. Crear y retornar LogEntry
        pass
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Convierte string de timestamp a datetime."""
        # TODO: Implementar
        pass
