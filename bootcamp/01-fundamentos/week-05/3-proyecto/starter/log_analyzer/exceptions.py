"""
Log Analyzer - Excepciones personalizadas
=========================================
"""


class LogAnalyzerError(Exception):
    """Base exception for log analyzer."""
    pass


class ParseError(LogAnalyzerError):
    """Error parsing a log entry."""
    
    def __init__(self, line_number: int, line: str, reason: str):
        self.line_number = line_number
        self.line = line
        self.reason = reason
        super().__init__(f"Line {line_number}: {reason}")


class FileFormatError(LogAnalyzerError):
    """Error with log file format."""
    pass


class FilterError(LogAnalyzerError):
    """Error filtering log entries."""
    pass


class ExportError(LogAnalyzerError):
    """Error exporting log data."""
    pass
