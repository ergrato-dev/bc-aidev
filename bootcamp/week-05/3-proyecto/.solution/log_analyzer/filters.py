"""
Log Analyzer - Filtros de logs (Solución)
=========================================
"""

import re
import logging
from datetime import datetime
from typing import Callable

from .parser import LogEntry
from .exceptions import FilterError

logger = logging.getLogger(__name__)


class LogFilter:
    """Filtros para entradas de log."""
    
    LEVEL_ORDER = {
        'DEBUG': 0,
        'INFO': 1,
        'WARNING': 2,
        'ERROR': 3,
        'CRITICAL': 4
    }
    
    @staticmethod
    def by_level(
        entries: list[LogEntry],
        level: str | list[str]
    ) -> list[LogEntry]:
        """
        Filtra entradas por nivel de log.
        
        Args:
            entries: Lista de entradas a filtrar.
            level: Nivel o lista de niveles.
            
        Returns:
            Lista filtrada de entradas.
        """
        if isinstance(level, str):
            levels = [level.upper()]
        else:
            levels = [l.upper() for l in level]
        
        logger.debug(f"Filtering by levels: {levels}")
        
        return [e for e in entries if e.level in levels]
    
    @staticmethod
    def by_level_minimum(
        entries: list[LogEntry],
        min_level: str
    ) -> list[LogEntry]:
        """
        Filtra entradas con nivel igual o superior al especificado.
        
        Args:
            entries: Lista de entradas a filtrar.
            min_level: Nivel mínimo.
            
        Returns:
            Lista filtrada de entradas.
        """
        min_level = min_level.upper()
        
        if min_level not in LogFilter.LEVEL_ORDER:
            raise FilterError(f"Invalid level: {min_level}")
        
        min_order = LogFilter.LEVEL_ORDER[min_level]
        
        logger.debug(f"Filtering by minimum level: {min_level}")
        
        return [
            e for e in entries 
            if LogFilter.LEVEL_ORDER.get(e.level, 0) >= min_order
        ]
    
    @staticmethod
    def by_date_range(
        entries: list[LogEntry],
        start: datetime | None = None,
        end: datetime | None = None
    ) -> list[LogEntry]:
        """
        Filtra entradas por rango de fechas.
        
        Args:
            entries: Lista de entradas a filtrar.
            start: Fecha/hora de inicio (inclusive).
            end: Fecha/hora de fin (inclusive).
            
        Returns:
            Lista filtrada de entradas.
        """
        logger.debug(f"Filtering by date range: {start} - {end}")
        
        result = entries
        
        if start:
            result = [e for e in result if e.timestamp >= start]
        
        if end:
            result = [e for e in result if e.timestamp <= end]
        
        return result
    
    @staticmethod
    def by_logger(
        entries: list[LogEntry],
        logger_name: str
    ) -> list[LogEntry]:
        """
        Filtra entradas por nombre de logger.
        
        Args:
            entries: Lista de entradas a filtrar.
            logger_name: Nombre del logger (soporta prefijos).
            
        Returns:
            Lista filtrada de entradas.
        """
        logger.debug(f"Filtering by logger: {logger_name}")
        
        return [
            e for e in entries 
            if e.logger_name == logger_name or e.logger_name.startswith(f"{logger_name}.")
        ]
    
    @staticmethod
    def by_pattern(
        entries: list[LogEntry],
        pattern: str,
        ignore_case: bool = True
    ) -> list[LogEntry]:
        """
        Filtra entradas donde el mensaje matchea un patrón regex.
        
        Args:
            entries: Lista de entradas a filtrar.
            pattern: Patrón regex a buscar.
            ignore_case: Si True, ignora mayúsculas/minúsculas.
            
        Returns:
            Lista filtrada de entradas.
        """
        flags = re.IGNORECASE if ignore_case else 0
        
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            raise FilterError(f"Invalid regex pattern: {e}")
        
        logger.debug(f"Filtering by pattern: {pattern}")
        
        return [e for e in entries if regex.search(e.message)]
    
    @staticmethod
    def custom(
        entries: list[LogEntry],
        predicate: Callable[[LogEntry], bool]
    ) -> list[LogEntry]:
        """
        Filtra entradas con una función personalizada.
        
        Args:
            entries: Lista de entradas a filtrar.
            predicate: Función que recibe LogEntry y retorna bool.
            
        Returns:
            Lista filtrada de entradas.
        """
        return [e for e in entries if predicate(e)]
