# 📁 Ejercicio 02: Estructurar Paquetes

## 🎯 Objetivo

Crear un paquete Python con múltiples módulos, `__init__.py` y subpaquetes.

---

## 📋 Instrucciones

### Paso 1: Crear Estructura de Paquete

Crea la siguiente estructura de carpetas y archivos:

```
starter/
├── main.py
└── data_tools/
    ├── __init__.py
    ├── readers.py
    └── writers.py
```

### Paso 2: Implementar Módulos

Implementa funciones de lectura y escritura de archivos.

### Paso 3: Configurar `__init__.py`

Exponer la API pública del paquete en `__init__.py`.

### Paso 4: Definir `__all__`

Controlar qué se exporta con `from package import *`.

### Paso 5: Agregar Subpaquete

Crear un subpaquete `transformers/` con utilidades adicionales.

---

## 📁 Estructura Final

```
starter/
├── main.py
└── data_tools/
    ├── __init__.py
    ├── readers.py
    ├── writers.py
    └── transformers/
        ├── __init__.py
        └── text.py
```

---

## ✅ Verificación

Deberías poder:

1. `from data_tools import read_json, write_json`
2. `from data_tools.readers import read_csv`
3. `from data_tools.transformers import clean_text`

---

## 🔗 Siguiente

Continúa con [Ejercicio 03: Imports](../ejercicio-03-imports/)

---

_Volver a: [Prácticas](../README.md)_
