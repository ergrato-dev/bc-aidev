"""
Log Analyzer - Parser de logs (Solución)
========================================
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
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Log file not found: {filepath}")
        
        self._errors = []
        entries = []
        
        logger.info(f"Parsing log file: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                entry = self.parse_line(line, line_number)
                if entry:
                    entries.append(entry)
        
        if not entries:
            raise FileFormatError(f"No valid log entries found in: {filepath}")
        
        logger.info(f"Parsed {len(entries)} entries ({len(self._errors)} errors)")
        
        return entries
    
    def parse_file_streaming(self, filepath: Path | str) -> Iterator[LogEntry]:
        """
        Parsea un archivo de log en modo streaming.
        
        Args:
            filepath: Ruta al archivo de log.
            
        Yields:
            LogEntry por cada línea válida.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Log file not found: {filepath}")
        
        self._errors = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                entry = self.parse_line(line, line_number)
                if entry:
                    yield entry
    
    def parse_line(self, line: str, line_number: int) -> LogEntry | None:
        """
        Parsea una línea de log individual.
        
        Args:
            line: Línea de texto a parsear.
            line_number: Número de línea.
            
        Returns:
            LogEntry si la línea es válida, None si no.
        """
        line = line.strip()
        
        if not line:
            return None
        
        match = self.PATTERN.match(line)
        
        if not match:
            error = ParseError(line_number, line, "Line doesn't match expected format")
            self._errors.append(error)
            if self.strict:
                raise error
            logger.debug(f"Skipping invalid line {line_number}")
            return None
        
        timestamp_str, level, logger_name, message = match.groups()
        
        level = level.upper()
        if level not in self.VALID_LEVELS:
            error = ParseError(line_number, line, f"Invalid level: {level}")
            self._errors.append(error)
            if self.strict:
                raise error
            return None
        
        try:
            timestamp = self._parse_timestamp(timestamp_str)
        except ValueError as e:
            error = ParseError(line_number, line, f"Invalid timestamp: {e}")
            self._errors.append(error)
            if self.strict:
                raise error
            return None
        
        return LogEntry(
            timestamp=timestamp,
            level=level,
            logger_name=logger_name.strip(),
            message=message.strip(),
            raw_line=line,
            line_number=line_number
        )
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Convierte string de timestamp a datetime."""
        return datetime.strptime(timestamp_str.strip(), '%Y-%m-%d %H:%M:%S')
