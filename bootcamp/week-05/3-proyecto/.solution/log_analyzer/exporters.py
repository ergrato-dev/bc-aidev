"""
Log Analyzer - Exportadores (Solución)
======================================
"""

import csv
import json
import logging
from pathlib import Path

from .parser import LogEntry
from .stats import LogStats, StatsAnalyzer
from .exceptions import ExportError

logger = logging.getLogger(__name__)


def export_to_json(
    entries: list[LogEntry],
    filepath: Path | str,
    indent: int = 2
) -> None:
    """
    Exporta entradas de log a archivo JSON.
    
    Args:
        entries: Lista de entradas a exportar.
        filepath: Ruta del archivo de salida.
        indent: Indentación del JSON.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting {len(entries)} entries to JSON: {filepath}")
    
    try:
        data = [_entry_to_dict(e) for e in entries]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        logger.info(f"Export completed: {filepath}")
        
    except (IOError, OSError) as e:
        raise ExportError(f"Failed to write JSON file: {e}")


def export_to_csv(
    entries: list[LogEntry],
    filepath: Path | str
) -> None:
    """
    Exporta entradas de log a archivo CSV.
    
    Args:
        entries: Lista de entradas a exportar.
        filepath: Ruta del archivo de salida.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting {len(entries)} entries to CSV: {filepath}")
    
    if not entries:
        logger.warning("No entries to export")
        return
    
    try:
        fieldnames = ['timestamp', 'level', 'logger', 'message', 'line_number']
        
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in entries:
                writer.writerow(_entry_to_dict(entry))
        
        logger.info(f"Export completed: {filepath}")
        
    except (IOError, OSError) as e:
        raise ExportError(f"Failed to write CSV file: {e}")


def export_stats_to_json(
    stats: LogStats,
    filepath: Path | str,
    indent: int = 2
) -> None:
    """
    Exporta estadísticas a archivo JSON.
    
    Args:
        stats: Estadísticas a exportar.
        filepath: Ruta del archivo de salida.
        indent: Indentación del JSON.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting stats to JSON: {filepath}")
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats.to_dict(), f, indent=indent, ensure_ascii=False)
        
    except (IOError, OSError) as e:
        raise ExportError(f"Failed to write stats file: {e}")


def export_summary(
    entries: list[LogEntry],
    filepath: Path | str
) -> None:
    """
    Exporta un resumen en texto plano.
    
    Args:
        entries: Lista de entradas.
        filepath: Ruta del archivo de salida.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting summary to: {filepath}")
    
    try:
        summary = StatsAnalyzer.summary(entries)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary)
        
    except (IOError, OSError) as e:
        raise ExportError(f"Failed to write summary file: {e}")


def _entry_to_dict(entry: LogEntry) -> dict:
    """Convierte LogEntry a diccionario serializable."""
    return {
        'timestamp': entry.timestamp.isoformat(),
        'level': entry.level,
        'logger': entry.logger_name,
        'message': entry.message,
        'line_number': entry.line_number
    }
