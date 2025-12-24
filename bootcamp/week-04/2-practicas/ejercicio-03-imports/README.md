# 🔗 Ejercicio 03: Imports Absolutos y Relativos

## 🎯 Objetivo

Dominar las diferentes formas de importar módulos y paquetes en Python.

---

## 📋 Instrucciones

### Paso 1: Crear Estructura con Múltiples Niveles

Crea una estructura de paquete con varios niveles de anidamiento.

### Paso 2: Usar Imports Absolutos

Importar usando rutas completas desde la raíz del proyecto.

### Paso 3: Usar Imports Relativos

Importar usando notación de puntos (`.`, `..`).

### Paso 4: Organizar Imports según PEP 8

Ordenar imports: stdlib → third-party → local.

### Paso 5: Resolver Import Circular

Identificar y solucionar un problema de import circular.

---

## 📁 Estructura Final

```
starter/
├── main.py
└── myapp/
    ├── __init__.py
    ├── core.py
    ├── utils.py
    └── services/
        ├── __init__.py
        └── processor.py
```

---

## ✅ Verificación

Deberías poder:

1. Importar desde cualquier nivel del paquete
2. Usar imports relativos dentro del paquete
3. Ejecutar `python -m myapp.services.processor` sin errores

---

## 🔗 Siguiente

Continúa con [Ejercicio 04: Entornos Virtuales](../ejercicio-04-entornos/)

---

_Volver a: [Prácticas](../README.md)_
