"""
Log Analyzer
============

Herramienta para analizar archivos de log.

Uso:
    from log_analyzer import LogParser, LogFilter, StatsAnalyzer
    
    parser = LogParser()
    entries = parser.parse_file('app.log')
    
    errors = LogFilter.by_level(entries, 'ERROR')
    stats = StatsAnalyzer.analyze(entries)
"""

__version__ = '1.0.0'
__author__ = 'AI Bootcamp'

from .parser import LogParser, LogEntry
from .filters import LogFilter
from .stats import StatsAnalyzer, LogStats
from .exporters import export_to_json, export_to_csv, export_stats_to_json
from .exceptions import (
    LogAnalyzerError,
    ParseError,
    FileFormatError,
    FilterError,
    ExportError
)

__all__ = [
    # Parser
    'LogParser',
    'LogEntry',
    # Filters
    'LogFilter',
    # Stats
    'StatsAnalyzer',
    'LogStats',
    # Exporters
    'export_to_json',
    'export_to_csv',
    'export_stats_to_json',
    # Exceptions
    'LogAnalyzerError',
    'ParseError',
    'FileFormatError',
    'FilterError',
    'ExportError',
]
