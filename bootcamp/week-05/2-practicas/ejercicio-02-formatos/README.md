# 📊 Ejercicio 02: Formatos de Datos (CSV y JSON)

## 🎯 Objetivos

- Leer y escribir archivos CSV
- Trabajar con JSON
- Convertir entre formatos
- Manejar errores de parsing

---

## 📋 Instrucciones

1. Abre `starter/main.py`
2. Descomenta cada paso y ejecútalo
3. Observa los archivos generados en `output/`

---

## Paso 1: Escribir CSV con DictWriter

Crear archivos CSV desde diccionarios.

```python
import csv
from pathlib import Path

users = [
    {'name': 'Alice', 'age': 30, 'email': 'alice@email.com'},
    {'name': 'Bob', 'age': 25, 'email': 'bob@email.com'},
    {'name': 'Carlos', 'age': 35, 'email': 'carlos@email.com'},
]

output_path = Path('output/users.csv')
output_path.parent.mkdir(exist_ok=True)

with open(output_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name', 'age', 'email'])
    writer.writeheader()
    writer.writerows(users)
```

**Descomenta** el Paso 1 y verifica el archivo CSV creado.

---

## Paso 2: Leer CSV con DictReader

Leer CSV como diccionarios.

```python
import csv
from pathlib import Path

csv_path = Path('output/users.csv')

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"{row['name']} tiene {row['age']} años")
```

**Descomenta** el Paso 2 en `starter/main.py`.

---

## Paso 3: Escribir JSON

Guardar datos estructurados en JSON.

```python
import json
from pathlib import Path

data = {
    'app': 'Mi Aplicación',
    'version': '1.0.0',
    'settings': {
        'debug': True,
        'max_users': 100
    },
    'features': ['auth', 'api', 'logging']
}

json_path = Path('output/config.json')

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
```

**Descomenta** el Paso 3 y revisa el JSON generado.

---

## Paso 4: Leer JSON

Cargar datos desde JSON.

```python
import json
from pathlib import Path

json_path = Path('output/config.json')

with open(json_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

print(f"App: {config['app']}")
print(f"Debug: {config['settings']['debug']}")
print(f"Features: {', '.join(config['features'])}")
```

**Descomenta** el Paso 4 en `starter/main.py`.

---

## Paso 5: Convertir CSV a JSON

Transformar datos entre formatos.

```python
import csv
import json
from pathlib import Path

# Leer CSV
csv_path = Path('output/users.csv')
with open(csv_path, 'r', encoding='utf-8') as f:
    users = list(csv.DictReader(f))

# Escribir JSON
json_path = Path('output/users.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(users, f, indent=2, ensure_ascii=False)
```

**Descomenta** el Paso 5 y compara ambos archivos.

---

## Paso 6: JSON Strings

Convertir entre strings y objetos.

```python
import json

# Dict a string JSON
data = {'name': 'Test', 'value': 42}
json_string = json.dumps(data, indent=2)
print(f"JSON string:\n{json_string}")

# String JSON a dict
json_input = '{"status": "ok", "count": 10}'
parsed = json.loads(json_input)
print(f"Parsed: {parsed}")
print(f"Count: {parsed['count']}")
```

**Descomenta** el Paso 6 en `starter/main.py`.

---

## Paso 7: Manejo de Errores

Manejar JSON inválido.

```python
import json

invalid_json = '{"name": "Test", invalid}'

try:
    data = json.loads(invalid_json)
except json.JSONDecodeError as e:
    print(f"Error de JSON: {e.msg}")
    print(f"Posición: línea {e.lineno}, columna {e.colno}")
```

**Descomenta** el Paso 7 y observa el manejo de errores.

---

## ✅ Verificación

Al completar, deberías tener:

- [ ] `output/users.csv` - Datos tabulares
- [ ] `output/config.json` - Configuración
- [ ] `output/users.json` - CSV convertido
- [ ] Entendimiento de DictReader/DictWriter

---

## 🔗 Siguiente

[Ejercicio 03: Excepciones](../ejercicio-03-excepciones/)
