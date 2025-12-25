# 📦 Módulos e Imports

## 🎯 Objetivos

- Comprender qué son los módulos en Python
- Dominar las diferentes formas de importar
- Conocer módulos de la biblioteca estándar
- Crear tus propios módulos
- Organizar código en paquetes

---

## 📋 Contenido

### 1. ¿Qué es un Módulo?

Un módulo es simplemente un **archivo Python (.py)** que contiene código reutilizable.

```python
# math_utils.py (nuestro módulo)
PI = 3.14159

def circle_area(radius: float) -> float:
    """Calculate circle area."""
    return PI * radius ** 2

def circle_perimeter(radius: float) -> float:
    """Calculate circle perimeter."""
    return 2 * PI * radius
```

---

### 2. Formas de Importar

#### Import completo

```python
import math

print(math.pi)          # 3.141592653589793
print(math.sqrt(16))    # 4.0
print(math.sin(0))      # 0.0
```

#### Import con alias

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ahora usamos el alias
array = np.array([1, 2, 3])
```

#### Import específico

```python
from math import pi, sqrt, sin

print(pi)         # 3.141592653589793
print(sqrt(16))   # 4.0 (sin prefijo math.)
```

#### Import con alias específico

```python
from math import pi as PI
from math import sqrt as raiz

print(PI)        # 3.141592653589793
print(raiz(16))  # 4.0
```

#### Import todo (⚠️ evitar)

```python
from math import *  # Importa TODO - contamina namespace

print(pi)
print(sqrt(16))

# ❌ Problemas:
# - No sabes de dónde viene cada función
# - Puede sobrescribir funciones existentes
# - Dificulta la lectura del código
```

---

### 3. Módulos de la Biblioteca Estándar

Python incluye muchos módulos útiles:

#### math - Operaciones matemáticas

```python
import math

print(math.pi)           # 3.141592653589793
print(math.e)            # 2.718281828459045
print(math.sqrt(16))     # 4.0
print(math.pow(2, 10))   # 1024.0
print(math.floor(3.7))   # 3
print(math.ceil(3.2))    # 4
print(math.log(100, 10)) # 2.0
```

#### random - Números aleatorios

```python
import random

# Semilla para reproducibilidad
random.seed(42)

print(random.random())        # Float entre 0 y 1
print(random.randint(1, 10))  # Entero entre 1 y 10
print(random.choice(['a', 'b', 'c']))  # Elemento aleatorio

# Mezclar lista
items = [1, 2, 3, 4, 5]
random.shuffle(items)
print(items)

# Muestra aleatoria
print(random.sample(range(100), 5))  # 5 números únicos
```

#### datetime - Fechas y horas

```python
from datetime import datetime, date, timedelta

# Fecha y hora actual
now = datetime.now()
print(now)  # 2025-12-23 15:30:45.123456

# Solo fecha
today = date.today()
print(today)  # 2025-12-23

# Formatear
print(now.strftime("%Y-%m-%d %H:%M"))  # 2025-12-23 15:30

# Operaciones con fechas
tomorrow = today + timedelta(days=1)
next_week = today + timedelta(weeks=1)
```

#### os y pathlib - Sistema de archivos

```python
import os
from pathlib import Path

# os (forma tradicional)
print(os.getcwd())         # Directorio actual
print(os.listdir('.'))     # Listar archivos

# pathlib (forma moderna - RECOMENDADA)
current = Path.cwd()
print(current)

# Crear rutas
data_path = Path('data') / 'train' / 'images'
print(data_path)  # data/train/images

# Verificar existencia
if data_path.exists():
    print("Path exists!")

# Listar archivos
for file in Path('.').glob('*.py'):
    print(file)
```

#### json - Trabajar con JSON

```python
import json

# Python dict a JSON string
data = {"name": "Ana", "age": 25, "skills": ["Python", "ML"]}
json_string = json.dumps(data, indent=2)
print(json_string)

# JSON string a Python dict
parsed = json.loads(json_string)
print(parsed["name"])  # Ana

# Guardar en archivo
with open('data.json', 'w') as f:
    json.dump(data, f, indent=2)

# Leer de archivo
with open('data.json', 'r') as f:
    loaded = json.load(f)
```

#### collections - Estructuras de datos avanzadas

```python
from collections import Counter, defaultdict, namedtuple

# Counter: contar elementos
words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
count = Counter(words)
print(count)  # Counter({'apple': 3, 'banana': 2, 'cherry': 1})
print(count.most_common(2))  # [('apple', 3), ('banana', 2)]

# defaultdict: dict con valor por defecto
word_count = defaultdict(int)
for word in words:
    word_count[word] += 1
print(dict(word_count))

# namedtuple: tupla con nombres
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)  # 10 20
```

---

### 4. Crear Tus Propios Módulos

#### Estructura básica

```
mi_proyecto/
├── main.py
└── utils/
    ├── __init__.py
    ├── math_utils.py
    └── string_utils.py
```

#### math_utils.py

```python
"""Utilidades matemáticas para el proyecto."""

def square(x: float) -> float:
    """Return x squared."""
    return x ** 2

def cube(x: float) -> float:
    """Return x cubed."""
    return x ** 3

def mean(numbers: list) -> float:
    """Calculate arithmetic mean."""
    return sum(numbers) / len(numbers)
```

#### string_utils.py

```python
"""Utilidades de strings para el proyecto."""

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    return text.strip().lower()

def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())
```

#### \_\_init\_\_.py

```python
"""Utils package."""
from .math_utils import square, cube, mean
from .string_utils import clean_text, word_count

__all__ = ['square', 'cube', 'mean', 'clean_text', 'word_count']
```

#### main.py

```python
# Importar del paquete
from utils import square, mean, clean_text

# O importar módulos específicos
from utils.math_utils import cube
from utils.string_utils import word_count

print(square(5))  # 25
print(mean([1, 2, 3, 4, 5]))  # 3.0
print(clean_text("  HELLO World  "))  # hello world
```

---

### 5. if \_\_name\_\_ == "\_\_main\_\_"

Permite ejecutar código solo cuando el archivo se ejecuta directamente:

```python
# calculator.py

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

# Este bloque solo se ejecuta si corremos calculator.py directamente
# No se ejecuta si importamos calculator desde otro archivo
if __name__ == "__main__":
    print("Testing calculator...")
    print(f"5 + 3 = {add(5, 3)}")
    print(f"5 - 3 = {subtract(5, 3)}")
```

```bash
# Ejecutar directamente
python calculator.py
# Output:
# Testing calculator...
# 5 + 3 = 8
# 5 - 3 = 2

# Importar desde otro archivo
# from calculator import add  # No imprime nada
```

---

### 6. Módulos Populares en ML/IA

```python
# Data Science / ML
import numpy as np           # Computación numérica
import pandas as pd          # Manipulación de datos
import matplotlib.pyplot as plt  # Visualización

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Deep Learning
import tensorflow as tf
import torch

# NLP
from transformers import pipeline
import spacy

# Convención de alias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
```

---

### 7. Buenas Prácticas de Import

```python
# ✅ BIEN - Orden estándar PEP 8
# 1. Standard library
import os
import sys
from pathlib import Path

# 2. Third party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 3. Local
from utils.math_utils import mean
from config import SETTINGS
```

```python
# ❌ MAL - Evitar
from math import *  # Import todo
import os, sys, json  # Múltiples en una línea
```

---

## 📚 Recursos Adicionales

- [Python Modules - Real Python](https://realpython.com/python-modules-packages/)
- [Python Standard Library](https://docs.python.org/3/library/)
- [PEP 8 - Imports](https://pep8.org/#imports)

---

## ✅ Checklist de Verificación

- [ ] Conozco las diferentes formas de importar módulos
- [ ] Sé usar módulos de la biblioteca estándar
- [ ] Puedo crear mis propios módulos
- [ ] Entiendo `if __name__ == "__main__"`
- [ ] Sigo las buenas prácticas de imports

---

_Volver a: [Semana 02](../README.md)_
