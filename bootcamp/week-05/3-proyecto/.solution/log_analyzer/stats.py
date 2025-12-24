"""
Log Analyzer - Estadísticas (Solución)
======================================
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from .parser import LogEntry

logger = logging.getLogger(__name__)


@dataclass
class LogStats:
    """Estadísticas de un conjunto de logs."""
    
    total: int = 0
    by_level: dict[str, int] = field(default_factory=dict)
    by_logger: dict[str, int] = field(default_factory=dict)
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    most_common_errors: list[tuple[str, int]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convierte las estadísticas a diccionario."""
        return {
            'total': self.total,
            'by_level': self.by_level,
            'by_logger': self.by_logger,
            'time_range': {
                'start': self.first_timestamp.isoformat() if self.first_timestamp else None,
                'end': self.last_timestamp.isoformat() if self.last_timestamp else None,
            },
            'most_common_errors': self.most_common_errors
        }


class StatsAnalyzer:
    """Analizador de estadísticas de logs."""
    
    @staticmethod
    def analyze(entries: list[LogEntry]) -> LogStats:
        """
        Genera estadísticas de una lista de entradas de log.
        
        Args:
            entries: Lista de entradas a analizar.
            
        Returns:
            LogStats con las estadísticas calculadas.
        """
        if not entries:
            return LogStats()
        
        logger.info(f"Analyzing {len(entries)} log entries")
        
        level_counter = Counter(e.level for e in entries)
        logger_counter = Counter(e.logger_name for e in entries)
        
        timestamps = [e.timestamp for e in entries]
        
        common_errors = StatsAnalyzer._find_common_errors(entries)
        
        return LogStats(
            total=len(entries),
            by_level=dict(level_counter),
            by_logger=dict(logger_counter),
            first_timestamp=min(timestamps),
            last_timestamp=max(timestamps),
            most_common_errors=common_errors
        )
    
    @staticmethod
    def _find_common_errors(
        entries: list[LogEntry],
        top_n: int = 5
    ) -> list[tuple[str, int]]:
        """
        Encuentra los mensajes de error más comunes.
        
        Args:
            entries: Lista de entradas.
            top_n: Número de errores a retornar.
            
        Returns:
            Lista de tuplas (mensaje, count).
        """
        error_entries = [e for e in entries if e.level in ('ERROR', 'CRITICAL')]
        
        if not error_entries:
            return []
        
        message_counter = Counter(e.message for e in error_entries)
        
        return message_counter.most_common(top_n)
    
    @staticmethod
    def summary(entries: list[LogEntry]) -> str:
        """
        Genera un resumen en texto de las estadísticas.
        
        Args:
            entries: Lista de entradas a analizar.
            
        Returns:
            String con el resumen formateado.
        """
        stats = StatsAnalyzer.analyze(entries)
        
        lines = [
            "=" * 50,
            "LOG ANALYSIS SUMMARY",
            "=" * 50,
            "",
            f"Total entries: {stats.total}",
            "",
            "By Level:",
        ]
        
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            count = stats.by_level.get(level, 0)
            if count > 0:
                pct = (count / stats.total) * 100
                lines.append(f"  {level:10} {count:5} ({pct:5.1f}%)")
        
        lines.append("")
        lines.append("By Logger:")
        
        sorted_loggers = sorted(
            stats.by_logger.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        for logger_name, count in sorted_loggers:
            lines.append(f"  {logger_name:20} {count:5}")
        
        if stats.first_timestamp and stats.last_timestamp:
            lines.append("")
            lines.append("Time Range:")
            lines.append(f"  Start: {stats.first_timestamp}")
            lines.append(f"  End:   {stats.last_timestamp}")
        
        if stats.most_common_errors:
            lines.append("")
            lines.append("Most Common Errors:")
            for msg, count in stats.most_common_errors:
                truncated = msg[:50] + "..." if len(msg) > 50 else msg
                lines.append(f"  [{count}x] {truncated}")
        
        lines.append("")
        lines.append("=" * 50)
        
        return "\n".join(lines)
