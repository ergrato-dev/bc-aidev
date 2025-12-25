# 📄 Archivos de Texto en Python

## 🎯 Objetivos

- Abrir, leer y escribir archivos de texto
- Usar context managers (`with`) correctamente
- Manejar encoding (UTF-8)
- Utilizar `pathlib` para rutas

---

## 1. Fundamentos de I/O

### La función `open()`

```python
# Sintaxis básica
file = open(filename, mode, encoding='utf-8')
```

### Modos de apertura

| Modo   | Descripción         | Archivo existe | No existe |
| ------ | ------------------- | -------------- | --------- |
| `'r'`  | Lectura (default)   | Lee            | Error     |
| `'w'`  | Escritura           | Sobrescribe    | Crea      |
| `'a'`  | Append              | Añade al final | Crea      |
| `'x'`  | Creación exclusiva  | Error          | Crea      |
| `'r+'` | Lectura + escritura | Lee/escribe    | Error     |
| `'w+'` | Escritura + lectura | Sobrescribe    | Crea      |

### Modos binarios

Añade `'b'` para modo binario:

- `'rb'` - Lectura binaria
- `'wb'` - Escritura binaria

---

## 2. Context Managers (`with`)

### ⚠️ El Problema

```python
# ❌ MAL - El archivo puede quedar abierto
file = open('data.txt', 'r')
content = file.read()
# Si hay error aquí, file.close() nunca se ejecuta
file.close()
```

### ✅ La Solución

```python
# ✅ BIEN - El archivo SIEMPRE se cierra
with open('data.txt', 'r', encoding='utf-8') as file:
    content = file.read()
# Aquí el archivo ya está cerrado, incluso si hubo error
```

### Múltiples archivos

```python
# Abrir varios archivos simultáneamente
with open('input.txt', 'r') as infile, \
     open('output.txt', 'w') as outfile:
    content = infile.read()
    outfile.write(content.upper())
```

---

## 3. Métodos de Lectura

### `read()` - Todo el contenido

```python
with open('poem.txt', 'r', encoding='utf-8') as f:
    content = f.read()  # String con todo el archivo
    print(content)
```

### `readline()` - Una línea

```python
with open('poem.txt', 'r', encoding='utf-8') as f:
    first_line = f.readline()   # Primera línea
    second_line = f.readline()  # Segunda línea
```

### `readlines()` - Lista de líneas

```python
with open('poem.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()  # ['línea1\n', 'línea2\n', ...]

for line in lines:
    print(line.strip())  # strip() quita \n
```

### Iteración directa (recomendado para archivos grandes)

```python
# ✅ MEJOR - No carga todo en memoria
with open('large_file.txt', 'r', encoding='utf-8') as f:
    for line in f:
        process(line.strip())
```

---

## 4. Métodos de Escritura

### `write()` - Escribir string

```python
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('Primera línea\n')
    f.write('Segunda línea\n')
```

### `writelines()` - Escribir lista

```python
lines = ['Línea 1\n', 'Línea 2\n', 'Línea 3\n']

with open('output.txt', 'w', encoding='utf-8') as f:
    f.writelines(lines)
```

### `print()` con archivo

```python
with open('output.txt', 'w', encoding='utf-8') as f:
    print('Hola mundo', file=f)
    print('Otra línea', file=f)
```

---

## 5. Pathlib - Rutas Modernas

### Crear rutas

```python
from pathlib import Path

# Rutas multiplataforma
data_dir = Path('data')
file_path = data_dir / 'users.txt'  # data/users.txt

# Ruta absoluta
abs_path = Path('/home/user/project/data.txt')

# Ruta relativa al script actual
script_dir = Path(__file__).parent
config = script_dir / 'config.json'
```

### Operaciones comunes

```python
from pathlib import Path

path = Path('data/users.txt')

# Información de la ruta
path.name          # 'users.txt'
path.stem          # 'users'
path.suffix        # '.txt'
path.parent        # Path('data')
path.parts         # ('data', 'users.txt')

# Comprobaciones
path.exists()      # ¿Existe?
path.is_file()     # ¿Es archivo?
path.is_dir()      # ¿Es directorio?

# Ruta absoluta
path.absolute()    # /home/user/project/data/users.txt
path.resolve()     # Resuelve symlinks también
```

### Lectura/escritura directa

```python
from pathlib import Path

path = Path('data.txt')

# Lectura
content = path.read_text(encoding='utf-8')
data = path.read_bytes()  # Binario

# Escritura
path.write_text('Contenido', encoding='utf-8')
path.write_bytes(b'Binary data')
```

### Crear directorios

```python
from pathlib import Path

# Crear directorio (y padres si no existen)
Path('data/processed').mkdir(parents=True, exist_ok=True)
```

### Listar archivos

```python
from pathlib import Path

data_dir = Path('data')

# Todos los archivos
for file in data_dir.iterdir():
    print(file)

# Solo .txt
for txt_file in data_dir.glob('*.txt'):
    print(txt_file)

# Recursivo
for py_file in data_dir.rglob('*.py'):
    print(py_file)
```

---

## 6. Encoding

### UTF-8 siempre

```python
# ✅ SIEMPRE especificar encoding
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# ❌ Sin encoding puede fallar
with open('file.txt', 'r') as f:  # Usa encoding del sistema
    content = f.read()
```

### Detectar encoding

```python
# pip install chardet
import chardet

with open('unknown.txt', 'rb') as f:
    raw = f.read()
    result = chardet.detect(raw)
    encoding = result['encoding']

with open('unknown.txt', 'r', encoding=encoding) as f:
    content = f.read()
```

### Manejar errores de encoding

```python
# Ignorar caracteres problemáticos
with open('file.txt', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Reemplazar con ?
with open('file.txt', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()
```

---

## 7. Archivos Binarios

### Imágenes, PDFs, etc.

```python
# Copiar archivo binario
with open('image.png', 'rb') as src:
    with open('copy.png', 'wb') as dst:
        dst.write(src.read())

# Por chunks (archivos grandes)
CHUNK_SIZE = 8192  # 8KB

with open('large.bin', 'rb') as src:
    with open('copy.bin', 'wb') as dst:
        while chunk := src.read(CHUNK_SIZE):
            dst.write(chunk)
```

---

## 8. Operaciones con Archivos

### Copiar, mover, eliminar

```python
from pathlib import Path
import shutil

src = Path('original.txt')
dst = Path('backup/copy.txt')

# Copiar
shutil.copy(src, dst)           # Solo archivo
shutil.copy2(src, dst)          # Preserva metadata
shutil.copytree('dir1', 'dir2') # Directorio completo

# Mover/renombrar
src.rename('new_name.txt')
shutil.move('file.txt', 'new_dir/')

# Eliminar
Path('file.txt').unlink()              # Archivo
Path('empty_dir').rmdir()              # Dir vacío
shutil.rmtree('dir_with_contents')     # Dir con contenido
```

### Archivos temporales

```python
import tempfile

# Archivo temporal (se elimina automáticamente)
with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
    f.write('Datos temporales')
    print(f.name)  # Ruta al archivo

# Directorio temporal
with tempfile.TemporaryDirectory() as tmpdir:
    print(tmpdir)  # /tmp/xxx
    # Hacer operaciones
# Se elimina al salir del with
```

---

## 9. Patrones Comunes

### Procesar archivo línea por línea

```python
from pathlib import Path

def process_log(filepath: Path) -> dict:
    """Cuenta líneas por nivel de log."""
    counts = {'INFO': 0, 'WARNING': 0, 'ERROR': 0}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            for level in counts:
                if level in line:
                    counts[level] += 1
                    break

    return counts
```

### Leer configuración

```python
from pathlib import Path

def load_config(filepath: Path) -> dict:
    """Carga archivo key=value."""
    config = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()

    return config
```

### Append con timestamp

```python
from pathlib import Path
from datetime import datetime

def log_event(filepath: Path, message: str) -> None:
    """Añade evento con timestamp."""
    timestamp = datetime.now().isoformat()

    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")
```

---

## 📚 Resumen

| Operación   | Código                                           |
| ----------- | ------------------------------------------------ |
| Leer todo   | `path.read_text(encoding='utf-8')`               |
| Escribir    | `path.write_text(content, encoding='utf-8')`     |
| Leer líneas | `for line in open(path, encoding='utf-8')`       |
| Append      | `open(path, 'a', encoding='utf-8')`              |
| Crear dir   | `Path('dir').mkdir(parents=True, exist_ok=True)` |
| Listar      | `Path('dir').glob('*.txt')`                      |
| Existe      | `path.exists()`                                  |
| Eliminar    | `path.unlink()`                                  |

---

## ✅ Checklist

- [ ] Siempre usar `with` para abrir archivos
- [ ] Especificar `encoding='utf-8'`
- [ ] Usar `pathlib` en lugar de strings para rutas
- [ ] Iterar líneas para archivos grandes
- [ ] Cerrar archivos incluso si hay errores

---

_Siguiente: [Formatos de Datos](02-formatos-datos.md)_
