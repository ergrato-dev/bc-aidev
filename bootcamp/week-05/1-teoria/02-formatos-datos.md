# 📊 Formatos de Datos: CSV, JSON, YAML

## 🎯 Objetivos

- Leer y escribir archivos CSV
- Trabajar con JSON para datos estructurados
- Usar YAML para configuraciones
- Elegir el formato apropiado para cada caso

---

## 1. CSV (Comma-Separated Values)

### Lectura básica

```python
import csv

with open('users.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # Primera fila (encabezados)

    for row in reader:
        print(row)  # Lista: ['John', '25', 'john@email.com']
```

### DictReader (recomendado)

```python
import csv

with open('users.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)

    for row in reader:
        # Acceso por nombre de columna
        print(f"{row['name']} tiene {row['age']} años")
```

### Escritura básica

```python
import csv

data = [
    ['name', 'age', 'email'],
    ['Alice', 30, 'alice@email.com'],
    ['Bob', 25, 'bob@email.com'],
]

with open('output.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)
```

### DictWriter (recomendado)

```python
import csv

users = [
    {'name': 'Alice', 'age': 30, 'email': 'alice@email.com'},
    {'name': 'Bob', 'age': 25, 'email': 'bob@email.com'},
]

with open('output.csv', 'w', encoding='utf-8', newline='') as f:
    fieldnames = ['name', 'age', 'email']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()  # Escribe encabezados
    writer.writerows(users)
```

### Delimitadores personalizados

```python
import csv

# TSV (Tab-Separated Values)
with open('data.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        print(row)

# Punto y coma (común en español)
with open('data.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        print(row)
```

---

## 2. JSON (JavaScript Object Notation)

### Lectura

```python
import json

with open('config.json', 'r', encoding='utf-8') as f:
    data = json.load(f)  # Carga como dict/list

print(data['name'])
print(data['settings']['debug'])
```

### Escritura

```python
import json

data = {
    'name': 'Mi App',
    'version': '1.0.0',
    'settings': {
        'debug': True,
        'max_connections': 100
    },
    'users': ['alice', 'bob', 'charlie']
}

with open('config.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
```

### Parámetros útiles

```python
import json

# indent: formato legible
# ensure_ascii=False: permite caracteres UTF-8
# sort_keys: ordena claves alfabéticamente

with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(data, f,
              indent=2,
              ensure_ascii=False,
              sort_keys=True)
```

### Strings JSON

```python
import json

# Dict a string JSON
data = {'name': 'Alice', 'age': 30}
json_string = json.dumps(data, indent=2)
print(json_string)

# String JSON a dict
json_string = '{"name": "Bob", "age": 25}'
data = json.loads(json_string)
print(data['name'])
```

### Tipos de datos

| Python          | JSON        |
| --------------- | ----------- |
| `dict`          | object `{}` |
| `list`, `tuple` | array `[]`  |
| `str`           | string `""` |
| `int`, `float`  | number      |
| `True`, `False` | true, false |
| `None`          | null        |

### Serializar objetos custom

```python
import json
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class User:
    name: str
    email: str
    created_at: datetime

# Encoder personalizado
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

user = User('Alice', 'alice@email.com', datetime.now())

# Con encoder custom
json_str = json.dumps(user, cls=CustomEncoder, indent=2)
print(json_str)

# O usando dataclass
json_str = json.dumps(asdict(user), default=str, indent=2)
```

---

## 3. YAML (YAML Ain't Markup Language)

> **Nota**: Requiere `pip install pyyaml`

### Lectura

```python
import yaml

with open('config.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

print(data['database']['host'])
```

### Escritura

```python
import yaml

config = {
    'app': {
        'name': 'Mi Aplicación',
        'version': '1.0.0',
        'debug': True
    },
    'database': {
        'host': 'localhost',
        'port': 5432,
        'name': 'mydb'
    },
    'features': ['auth', 'api', 'logging']
}

with open('config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
```

### Resultado YAML

```yaml
app:
  debug: true
  name: Mi Aplicación
  version: 1.0.0
database:
  host: localhost
  name: mydb
  port: 5432
features:
  - auth
  - api
  - logging
```

### Múltiples documentos

```python
import yaml

# Archivo con múltiples documentos separados por ---
with open('multi.yaml', 'r') as f:
    documents = list(yaml.safe_load_all(f))

for doc in documents:
    print(doc)
```

---

## 4. Comparativa de Formatos

| Característica     | CSV             | JSON         | YAML          |
| ------------------ | --------------- | ------------ | ------------- |
| **Uso principal**  | Datos tabulares | APIs, config | Configuración |
| **Legibilidad**    | Media           | Alta         | Muy alta      |
| **Tipos de datos** | Solo strings    | Básicos      | Avanzados     |
| **Anidamiento**    | No              | Sí           | Sí            |
| **Comentarios**    | No              | No           | Sí            |
| **Stdlib**         | ✅ csv          | ✅ json      | ❌ pyyaml     |
| **Tamaño**         | Pequeño         | Mediano      | Mediano       |

### Cuándo usar cada uno

```python
# CSV: Datos tabulares, hojas de cálculo
# - Exportar a Excel
# - Datos de base de datos
# - Datasets para ML

# JSON: Datos estructurados
# - APIs REST
# - Configuración de aplicaciones
# - Intercambio de datos

# YAML: Configuración legible
# - Docker Compose
# - CI/CD (GitHub Actions)
# - Kubernetes
```

---

## 5. Patrones Comunes

### Cargar/guardar configuración

```python
import json
from pathlib import Path

class Config:
    """Gestor de configuración JSON."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.data = self._load()

    def _load(self) -> dict:
        if self.filepath.exists():
            return json.loads(self.filepath.read_text(encoding='utf-8'))
        return {}

    def save(self) -> None:
        self.filepath.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value) -> None:
        self.data[key] = value
        self.save()


# Uso
config = Config('settings.json')
config.set('theme', 'dark')
print(config.get('theme'))  # 'dark'
```

### Procesar CSV grande

```python
import csv
from pathlib import Path
from typing import Iterator

def process_csv_chunks(
    filepath: Path,
    chunk_size: int = 1000
) -> Iterator[list[dict]]:
    """Procesa CSV en chunks para archivos grandes."""

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        chunk = []

        for row in reader:
            chunk.append(row)

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        if chunk:  # Último chunk
            yield chunk


# Uso
for chunk in process_csv_chunks(Path('large_data.csv')):
    for row in chunk:
        process_row(row)
```

### Convertir entre formatos

```python
import csv
import json
from pathlib import Path

def csv_to_json(csv_path: Path, json_path: Path) -> None:
    """Convierte CSV a JSON."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def json_to_csv(json_path: Path, csv_path: Path) -> None:
    """Convierte JSON (lista de dicts) a CSV."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        return

    fieldnames = data[0].keys()

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
```

---

## 6. Manejo de Errores

```python
import json
import csv
from pathlib import Path

def safe_load_json(filepath: Path) -> dict | None:
    """Carga JSON con manejo de errores."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Archivo no encontrado: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON inválido: {e}")
        return None


def safe_load_csv(filepath: Path) -> list[dict]:
    """Carga CSV con manejo de errores."""
    rows = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=2):
                try:
                    rows.append(row)
                except Exception as e:
                    print(f"Error en línea {i}: {e}")
    except FileNotFoundError:
        print(f"Archivo no encontrado: {filepath}")

    return rows
```

---

## 📚 Resumen

| Operación     | CSV                         | JSON                 |
| ------------- | --------------------------- | -------------------- |
| Leer          | `csv.DictReader(f)`         | `json.load(f)`       |
| Escribir      | `csv.DictWriter(f, fields)` | `json.dump(data, f)` |
| String → Data | -                           | `json.loads(s)`      |
| Data → String | -                           | `json.dumps(data)`   |

---

## ✅ Checklist

- [ ] CSV: Usar `DictReader`/`DictWriter`
- [ ] CSV: Incluir `newline=''` al escribir
- [ ] JSON: Usar `indent` para legibilidad
- [ ] JSON: `ensure_ascii=False` para UTF-8
- [ ] YAML: Siempre `safe_load` (nunca `load`)
- [ ] Manejar errores de parsing

---

_Siguiente: [Excepciones](03-excepciones.md)_
