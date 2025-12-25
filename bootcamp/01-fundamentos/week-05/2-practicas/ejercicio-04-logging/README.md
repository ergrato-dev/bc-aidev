# 📝 Ejercicio 04: Logging

## 🎯 Objetivos

- Configurar el módulo logging
- Usar diferentes niveles de log
- Crear handlers y formatters
- Implementar logging en aplicaciones

---

## 📋 Instrucciones

1. Abre `starter/main.py`
2. Descomenta cada paso y ejecútalo
3. Observa los logs en consola y archivos

---

## Paso 1: Logging Básico vs print

Por qué usar logging en lugar de print.

```python
import logging

# Configuración mínima
logging.basicConfig(level=logging.DEBUG)

# En lugar de print
logging.debug("Mensaje de debug")
logging.info("Mensaje informativo")
logging.warning("Advertencia")
logging.error("Error")
logging.critical("Error crítico")
```

**Descomenta** el Paso 1 en `starter/main.py`.

---

## Paso 2: Formato Personalizado

Añadir timestamp y más información.

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Mensaje con formato personalizado")
```

**Descomenta** el Paso 2 y observa el formato.

---

## Paso 3: Logger por Módulo

Crear loggers con nombre para mejor organización.

```python
import logging

# Logger con nombre del módulo
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Handler de consola
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)
```

**Descomenta** el Paso 3 en `starter/main.py`.

---

## Paso 4: Logging a Archivo

Guardar logs en archivos.

```python
import logging
from pathlib import Path

logger = logging.getLogger('myapp')
logger.setLevel(logging.DEBUG)

# Handler de archivo
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

file_handler = logging.FileHandler(log_dir / 'app.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
```

**Descomenta** el Paso 4 y revisa `logs/app.log`.

---

## Paso 5: Múltiples Handlers

Consola + archivo con diferentes niveles.

```python
import logging

logger = logging.getLogger('myapp')
logger.setLevel(logging.DEBUG)

# Consola: solo INFO+
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Archivo: todo (DEBUG+)
file_handler = logging.FileHandler('logs/debug.log')
file_handler.setLevel(logging.DEBUG)

# Añadir ambos
logger.addHandler(console)
logger.addHandler(file_handler)
```

**Descomenta** el Paso 5 en `starter/main.py`.

---

## Paso 6: Logging de Excepciones

Usar `logger.exception()` en bloques except.

```python
import logging

logger = logging.getLogger(__name__)

try:
    result = 10 / 0
except ZeroDivisionError:
    logger.exception("Error en cálculo")
    # Incluye automáticamente el traceback
```

**Descomenta** el Paso 6 y observa el traceback.

---

## Paso 7: Configuración Completa

Setup profesional para aplicaciones.

```python
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(log_dir='logs', level=logging.INFO):
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger('myapp')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )

    # Consola
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Archivo con rotación
    file_handler = RotatingFileHandler(
        log_dir / 'app.log',
        maxBytes=1024*1024,  # 1MB
        backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
```

**Descomenta** el Paso 7 en `starter/main.py`.

---

## ✅ Verificación

Al completar, deberías tener:

- [ ] `logs/app.log` creado
- [ ] `logs/debug.log` con más detalle
- [ ] Entendimiento de niveles DEBUG/INFO/WARNING/ERROR
- [ ] Saber usar `logger.exception()`

---

## 🔗 Siguiente

[Proyecto: Log Analyzer](../../3-proyecto/)
