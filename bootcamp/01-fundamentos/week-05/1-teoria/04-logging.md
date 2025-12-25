# 📝 Sistema de Logging en Python

## 🎯 Objetivos

- Entender por qué usar logging en lugar de print
- Configurar el módulo logging
- Usar niveles de log apropiados
- Crear handlers y formatters personalizados

---

## 1. ¿Por qué Logging?

### print() vs logging

```python
# ❌ MAL - print() para debugging
print("Procesando archivo...")
print(f"Error: {e}")
print("DEBUG: valor =", value)

# ✅ BIEN - logging
import logging
logger = logging.getLogger(__name__)

logger.info("Procesando archivo...")
logger.error(f"Error: {e}")
logger.debug(f"valor = {value}")
```

### Ventajas de logging

| Característica        | print() | logging |
| --------------------- | ------- | ------- |
| Niveles de severidad  | ❌      | ✅      |
| Filtrar mensajes      | ❌      | ✅      |
| Múltiples destinos    | ❌      | ✅      |
| Formato personalizado | ❌      | ✅      |
| Timestamp automático  | ❌      | ✅      |
| Desactivar fácilmente | ❌      | ✅      |
| Thread-safe           | ❌      | ✅      |

---

## 2. Niveles de Log

| Nivel      | Valor | Uso                                   |
| ---------- | ----- | ------------------------------------- |
| `DEBUG`    | 10    | Información detallada para debugging  |
| `INFO`     | 20    | Confirmación de funcionamiento normal |
| `WARNING`  | 30    | Algo inesperado pero no crítico       |
| `ERROR`    | 40    | Error que impide una función          |
| `CRITICAL` | 50    | Error grave, programa puede fallar    |

### Ejemplos de uso

```python
import logging

logger = logging.getLogger(__name__)

# DEBUG: Solo durante desarrollo
logger.debug("Variable x = %s", x)

# INFO: Eventos normales importantes
logger.info("Usuario %s inició sesión", username)

# WARNING: Algo que podría ser problema
logger.warning("Disco al 90%% de capacidad")

# ERROR: Algo falló
logger.error("No se pudo conectar a la base de datos")

# CRITICAL: Sistema en riesgo
logger.critical("Sin memoria disponible")
```

---

## 3. Configuración Básica

### basicConfig (rápido)

```python
import logging

# Configuración mínima
logging.basicConfig(level=logging.INFO)

# Con más opciones
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Aplicación iniciada")
```

### Logger por módulo

```python
import logging

# Crear logger con nombre del módulo
logger = logging.getLogger(__name__)

def process_data(data):
    logger.info("Procesando %d elementos", len(data))
    # ...
```

---

## 4. Handlers

Los handlers determinan a dónde van los logs.

### StreamHandler (consola)

```python
import logging

logger = logging.getLogger('myapp')
logger.setLevel(logging.DEBUG)

# Handler para consola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)
```

### FileHandler (archivo)

```python
import logging

logger = logging.getLogger('myapp')
logger.setLevel(logging.DEBUG)

# Handler para archivo
file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

logger.addHandler(file_handler)
```

### RotatingFileHandler (archivo con rotación)

```python
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger('myapp')

# Rota cuando alcanza 5MB, mantiene 3 backups
handler = RotatingFileHandler(
    'app.log',
    maxBytes=5*1024*1024,  # 5MB
    backupCount=3,
    encoding='utf-8'
)

logger.addHandler(handler)
```

### TimedRotatingFileHandler (rotación por tiempo)

```python
from logging.handlers import TimedRotatingFileHandler

# Rota cada medianoche, mantiene 7 días
handler = TimedRotatingFileHandler(
    'app.log',
    when='midnight',
    interval=1,
    backupCount=7,
    encoding='utf-8'
)
```

---

## 5. Formatters

### Formato personalizado

```python
import logging

formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger('myapp')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.info("Mensaje de prueba")
# 2025-12-23 10:30:45 | INFO     | myapp | Mensaje de prueba
```

### Atributos disponibles

| Atributo        | Descripción               |
| --------------- | ------------------------- |
| `%(asctime)s`   | Timestamp                 |
| `%(name)s`      | Nombre del logger         |
| `%(levelname)s` | Nivel (INFO, ERROR, etc.) |
| `%(message)s`   | El mensaje                |
| `%(filename)s`  | Nombre del archivo        |
| `%(lineno)d`    | Número de línea           |
| `%(funcName)s`  | Nombre de la función      |
| `%(module)s`    | Nombre del módulo         |
| `%(pathname)s`  | Ruta completa             |

---

## 6. Configuración Completa

### Patrón recomendado

```python
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(
    log_dir: Path = Path('logs'),
    level: int = logging.INFO
) -> logging.Logger:
    """Configura logging para la aplicación."""

    # Crear directorio de logs
    log_dir.mkdir(exist_ok=True)

    # Crear logger raíz
    logger = logging.getLogger('myapp')
    logger.setLevel(logging.DEBUG)  # Captura todo, handlers filtran

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler consola (solo INFO+)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Handler archivo (todo, con rotación)
    file_handler = RotatingFileHandler(
        log_dir / 'app.log',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler para errores (archivo separado)
    error_handler = RotatingFileHandler(
        log_dir / 'errors.log',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger


# Uso
logger = setup_logging()
logger.info("Aplicación iniciada")
```

### Configuración con dictConfig

```python
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'filename': 'logs/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
        },
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
```

---

## 7. Logging de Excepciones

### logger.exception()

```python
import logging

logger = logging.getLogger(__name__)

try:
    result = risky_operation()
except Exception as e:
    # Incluye automáticamente el traceback
    logger.exception("Error en operación")

# Output:
# ERROR | Error en operación
# Traceback (most recent call last):
#   File "...", line ...
#     result = risky_operation()
# SomeError: mensaje del error
```

### exc_info=True

```python
try:
    process()
except ValueError:
    logger.error("Valor inválido", exc_info=True)
```

---

## 8. Contexto Adicional

### Usar extra

```python
logger.info(
    "Usuario procesado",
    extra={'user_id': 123, 'action': 'login'}
)
```

### LoggerAdapter

```python
import logging

class ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[{self.extra['request_id']}] {msg}", kwargs


logger = logging.getLogger(__name__)
adapter = ContextAdapter(logger, {'request_id': 'abc-123'})

adapter.info("Procesando request")
# [abc-123] Procesando request
```

---

## 9. Patrones Comunes

### Logger por módulo

```python
# myapp/utils.py
import logging

logger = logging.getLogger(__name__)  # 'myapp.utils'

def helper():
    logger.debug("Helper ejecutado")
```

### Logging en clases

```python
import logging

class DataProcessor:
    def __init__(self):
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    def process(self, data):
        self.logger.info("Iniciando proceso")
        # ...
        self.logger.info("Proceso completado")
```

### Decorador para logging

```python
import logging
import functools
from typing import Callable, TypeVar

T = TypeVar('T')
logger = logging.getLogger(__name__)

def log_calls(func: Callable[..., T]) -> Callable[..., T]:
    """Loguea llamadas a función."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Llamando {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completado")
            return result
        except Exception as e:
            logger.exception(f"Error en {func.__name__}")
            raise
    return wrapper


@log_calls
def process_data(data):
    # ...
    pass
```

---

## 10. Filtros

```python
import logging

class LevelFilter(logging.Filter):
    """Solo permite logs de un nivel específico."""

    def __init__(self, level: int):
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self.level


# Solo INFO (no DEBUG, WARNING, ERROR)
info_handler = logging.StreamHandler()
info_handler.addFilter(LevelFilter(logging.INFO))
```

---

## 📚 Resumen

| Componente | Función                     |
| ---------- | --------------------------- |
| Logger     | Genera mensajes de log      |
| Handler    | Envía logs a destinos       |
| Formatter  | Da formato a los mensajes   |
| Filter     | Filtra qué logs se procesan |

| Nivel    | Cuándo usar             |
| -------- | ----------------------- |
| DEBUG    | Desarrollo, diagnóstico |
| INFO     | Eventos normales        |
| WARNING  | Algo inesperado         |
| ERROR    | Falló una operación     |
| CRITICAL | Sistema comprometido    |

---

## ✅ Checklist

- [ ] Usar logging en lugar de print
- [ ] Crear logger con `__name__`
- [ ] Elegir nivel apropiado para cada mensaje
- [ ] Configurar handler para consola
- [ ] Configurar handler para archivo
- [ ] Usar `.exception()` en bloques except
- [ ] No loguear información sensible

---

_Volver a: [Semana 05](../README.md)_
