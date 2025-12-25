# 🔍 Proyecto: Log Analyzer

## 📋 Descripción

Desarrolla una herramienta CLI para analizar archivos de log. El analizador debe poder:

- Parsear diferentes formatos de log
- Filtrar por nivel (INFO, WARNING, ERROR, etc.)
- Generar estadísticas
- Exportar resultados a JSON/CSV

---

## 🎯 Objetivos de Aprendizaje

- Aplicar manejo de archivos con `pathlib`
- Usar `csv` y `json` para exportación
- Implementar excepciones personalizadas
- Configurar logging profesional
- Crear CLI con `argparse`

---

## 📁 Estructura del Proyecto

```
log_analyzer/
├── __init__.py
├── parser.py          # Parsear entradas de log
├── filters.py         # Filtrar logs
├── stats.py           # Generar estadísticas
├── exporters.py       # Exportar a JSON/CSV
├── exceptions.py      # Excepciones personalizadas
└── cli.py             # Interfaz de línea de comandos
```

---

## 📊 Formato de Log Esperado

El analizador debe soportar el formato estándar:

```
2025-12-23 10:30:45 | INFO     | app.main | Aplicación iniciada
2025-12-23 10:30:46 | DEBUG    | app.db   | Conectando a base de datos
2025-12-23 10:30:47 | WARNING  | app.auth | Token próximo a expirar
2025-12-23 10:30:48 | ERROR    | app.api  | Conexión rechazada: timeout
2025-12-23 10:30:49 | CRITICAL | app.db   | Base de datos no responde
```

Campos separados por `|`:

1. **Timestamp**: `YYYY-MM-DD HH:MM:SS`
2. **Level**: DEBUG, INFO, WARNING, ERROR, CRITICAL
3. **Logger**: Nombre del módulo
4. **Message**: Mensaje del log

---

## 🚀 Funcionalidades Requeridas

### 1. Parsing de Logs

```python
from log_analyzer import LogParser

parser = LogParser()
entries = parser.parse_file('app.log')

for entry in entries:
    print(f"{entry.timestamp} [{entry.level}] {entry.message}")
```

### 2. Filtrado

```python
from log_analyzer import LogFilter

# Filtrar por nivel
errors = LogFilter.by_level(entries, 'ERROR')

# Filtrar por rango de fechas
from datetime import datetime
recent = LogFilter.by_date_range(
    entries,
    start=datetime(2025, 12, 23, 10, 0),
    end=datetime(2025, 12, 23, 11, 0)
)

# Filtrar por patrón en mensaje
db_logs = LogFilter.by_pattern(entries, 'database|db')
```

### 3. Estadísticas

```python
from log_analyzer import LogStats

stats = LogStats.analyze(entries)
print(f"Total: {stats['total']}")
print(f"Por nivel: {stats['by_level']}")
print(f"Por logger: {stats['by_logger']}")
```

### 4. Exportación

```python
from log_analyzer import export_to_json, export_to_csv

export_to_json(entries, 'output/logs.json')
export_to_csv(entries, 'output/logs.csv')
```

### 5. CLI

```bash
# Analizar archivo
python -m log_analyzer analyze app.log

# Filtrar por nivel
python -m log_analyzer analyze app.log --level ERROR

# Estadísticas
python -m log_analyzer stats app.log

# Exportar
python -m log_analyzer export app.log --format json --output results.json
```

---

## 📝 Implementación

### Paso 1: Excepciones Personalizadas

```python
# exceptions.py

class LogAnalyzerError(Exception):
    """Base exception for log analyzer."""
    pass

class ParseError(LogAnalyzerError):
    """Error parsing log entry."""
    def __init__(self, line_number: int, line: str, reason: str):
        self.line_number = line_number
        self.line = line
        self.reason = reason
        super().__init__(f"Line {line_number}: {reason}")

class FileFormatError(LogAnalyzerError):
    """Error with file format."""
    pass
```

### Paso 2: Modelo de Datos

```python
# parser.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LogEntry:
    timestamp: datetime
    level: str
    logger: str
    message: str
    raw_line: str
    line_number: int
```

### Paso 3: Parser

```python
# parser.py
import re
from pathlib import Path

class LogParser:
    PATTERN = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*'
        r'(\w+)\s*\|\s*'
        r'([\w.]+)\s*\|\s*'
        r'(.+)$'
    )

    def parse_file(self, filepath: Path) -> list[LogEntry]:
        # TODO: Implementar
        pass

    def parse_line(self, line: str, line_number: int) -> LogEntry:
        # TODO: Implementar
        pass
```

---

## ✅ Criterios de Evaluación

| Criterio                        | Puntos |
| ------------------------------- | ------ |
| Parsing correcto de logs        | 6      |
| Filtrado por nivel/fecha/patrón | 6      |
| Estadísticas completas          | 6      |
| Exportación JSON/CSV            | 6      |
| Excepciones personalizadas      | 3      |
| Logging interno                 | 3      |
| **Total**                       | **30** |

---

## 📌 Entregables

1. Código fuente en `starter/log_analyzer/`
2. Archivo de prueba `sample.log` con al menos 20 entradas
3. Resultados de análisis exportados

---

## 💡 Tips

1. **Usa dataclasses** para LogEntry
2. **Regex** para parsear líneas
3. **Generator** para archivos grandes
4. **logging** para el propio analyzer
5. **Maneja líneas malformadas** sin crashear

---

## 🔗 Recursos

- [Teoría: Archivos](../1-teoria/01-archivos-texto.md)
- [Teoría: Formatos](../1-teoria/02-formatos-datos.md)
- [Teoría: Excepciones](../1-teoria/03-excepciones.md)
- [Teoría: Logging](../1-teoria/04-logging.md)

---

_Volver a: [Semana 05](../README.md)_
