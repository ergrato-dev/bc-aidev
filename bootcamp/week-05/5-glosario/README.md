# 📖 Glosario - Semana 05

## Manejo de Archivos y Excepciones

Términos técnicos clave de esta semana, ordenados alfabéticamente.

---

## A

### Append Mode (`'a'`)

Modo de apertura de archivo que añade contenido al final sin borrar el existente.

```python
with open('log.txt', 'a') as f:
    f.write('Nueva línea\n')
```

### `argparse`

Módulo de la biblioteca estándar para crear interfaces de línea de comandos (CLI).

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true')
```

---

## B

### Binary Mode (`'b'`)

Modo de apertura para leer/escribir archivos en formato binario (bytes).

```python
with open('image.png', 'rb') as f:
    data = f.read()  # bytes, no str
```

### `BufferedReader` / `BufferedWriter`

Clases que proporcionan buffering para operaciones de I/O, mejorando el rendimiento.

---

## C

### Context Manager

Objeto que define `__enter__` y `__exit__` para usar con `with`. Garantiza limpieza de recursos.

```python
with open('file.txt') as f:  # Context manager
    content = f.read()
# f.close() automático al salir
```

### CSV (Comma-Separated Values)

Formato de texto plano para datos tabulares, donde los valores se separan por comas.

```csv
nombre,edad,ciudad
Ana,25,Madrid
```

### `csv.DictReader`

Clase que lee CSV y retorna cada fila como diccionario con las cabeceras como claves.

```python
import csv
with open('data.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['nombre'])
```

### `csv.DictWriter`

Clase que escribe diccionarios a CSV con cabeceras especificadas.

---

## D

### `DEBUG`

Nivel de logging más bajo (10). Para información detallada de diagnóstico.

```python
logging.debug('Variable x = %s', x)
```

---

## E

### Encoding

Esquema de codificación de caracteres (UTF-8, ASCII, Latin-1). Define cómo se representan los caracteres como bytes.

```python
open('file.txt', encoding='utf-8')
```

### `ERROR`

Nivel de logging (40) para errores que no detienen la ejecución.

### Exception

Evento que interrumpe el flujo normal del programa. Se captura con `try/except`.

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("División por cero")
```

### Exception Chaining

Vincular una excepción con otra usando `raise ... from`.

```python
try:
    parse_config()
except KeyError as e:
    raise ConfigError("Campo faltante") from e
```

---

## F

### File Handle / File Object

Objeto retornado por `open()` que representa un archivo abierto.

### `FileHandler`

Handler de logging que escribe mensajes a un archivo.

```python
handler = logging.FileHandler('app.log')
```

### `FileNotFoundError`

Excepción cuando se intenta abrir un archivo que no existe.

### `finally`

Bloque que siempre se ejecuta, haya o no excepción. Ideal para limpieza.

```python
try:
    f = open('file.txt')
finally:
    f.close()  # Siempre se ejecuta
```

### Formatter

Objeto que define el formato de salida de los mensajes de log.

```python
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
```

---

## H

### Handler

Componente de logging que determina dónde enviar los mensajes (consola, archivo, email).

---

## I

### `INFO`

Nivel de logging (20) para información general sobre la ejecución.

### `IOError`

Excepción base para errores de entrada/salida (heredada por otras).

---

## J

### JSON (JavaScript Object Notation)

Formato ligero de intercambio de datos, legible por humanos.

```json
{ "nombre": "Ana", "edad": 25 }
```

### `json.dump()` / `json.dumps()`

Funciones para serializar objetos Python a JSON. `dump()` escribe a archivo, `dumps()` retorna string.

### `json.load()` / `json.loads()`

Funciones para deserializar JSON a objetos Python. `load()` lee de archivo, `loads()` de string.

---

## L

### Logger

Objeto principal del módulo logging. Punto de entrada para emitir mensajes.

```python
logger = logging.getLogger(__name__)
logger.info('Mensaje')
```

### Logging Level

Severidad del mensaje: DEBUG(10) < INFO(20) < WARNING(30) < ERROR(40) < CRITICAL(50).

---

## N

### Newline (`'\n'`)

Carácter de salto de línea. En Windows también existe `'\r\n'`.

---

## O

### `open()`

Función built-in para abrir archivos. Retorna un file object.

```python
f = open('file.txt', 'r', encoding='utf-8')
```

---

## P

### `Path`

Clase de pathlib que representa rutas de archivos de forma orientada a objetos.

```python
from pathlib import Path
p = Path('data') / 'file.txt'
```

### `pathlib`

Módulo moderno (Python 3.4+) para manipular rutas de archivos.

### `PermissionError`

Excepción cuando no hay permisos para acceder a un archivo.

---

## R

### `raise`

Palabra clave para lanzar una excepción explícitamente.

```python
if value < 0:
    raise ValueError("Valor debe ser positivo")
```

### Read Mode (`'r'`)

Modo de apertura predeterminado. Solo lectura, falla si no existe.

### `readline()` / `readlines()`

Métodos para leer una línea o todas las líneas de un archivo.

### `RotatingFileHandler`

Handler que rota archivos de log cuando alcanzan cierto tamaño.

```python
from logging.handlers import RotatingFileHandler
handler = RotatingFileHandler('app.log', maxBytes=1000000, backupCount=5)
```

---

## S

### Serialization

Proceso de convertir objetos Python a formato almacenable (JSON, pickle, etc.).

### `StreamHandler`

Handler de logging que escribe a streams (stdout, stderr).

### `sys.stderr` / `sys.stdout`

Streams estándar de error y salida. Logging usa stderr por defecto.

---

## T

### `try`

Palabra clave que inicia un bloque de código donde se capturan excepciones.

### Text Mode (default)

Modo de apertura donde el archivo se trata como texto (strings).

### Traceback

Información de la pila de llamadas cuando ocurre una excepción.

---

## U

### UTF-8

Codificación de caracteres Unicode más común. Soporta todos los caracteres.

---

## W

### `WARNING`

Nivel de logging (30) para situaciones inesperadas que no son errores.

### `with` Statement

Construcción para usar context managers. Garantiza limpieza automática.

```python
with open('file.txt') as f:
    data = f.read()
```

### Write Mode (`'w'`)

Modo de apertura que crea/sobrescribe el archivo.

---

## Y

### YAML (YAML Ain't Markup Language)

Formato de serialización legible, común para configuración.

```yaml
nombre: Ana
edad: 25
```

---

## 📚 Referencias

- [Python Glossary](https://docs.python.org/3/glossary.html)
- [Built-in Exceptions](https://docs.python.org/3/library/exceptions.html)
- [logging — Logging facility](https://docs.python.org/3/library/logging.html)

---

## 🔗 Navegación

| Anterior                              | Inicio                    | Siguiente                              |
| ------------------------------------- | ------------------------- | -------------------------------------- |
| [← Recursos](../4-recursos/README.md) | [Semana 05](../README.md) | [Semana 06 →](../../week-06/README.md) |
