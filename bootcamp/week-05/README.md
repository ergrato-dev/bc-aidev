# 📂 Semana 05: Manejo de Archivos y Excepciones

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Leer y escribir archivos de texto y binarios
- ✅ Usar context managers (`with`) para manejo seguro de recursos
- ✅ Trabajar con diferentes formatos: CSV, JSON, YAML
- ✅ Manejar excepciones con `try/except/else/finally`
- ✅ Crear excepciones personalizadas
- ✅ Aplicar logging para debugging y monitoreo
- ✅ Usar `pathlib` para manipulación de rutas

---

## 📚 Requisitos Previos

- ✅ Semana 01: Fundamentos de Python
- ✅ Semana 02: Estructuras de Datos
- ✅ Semana 03: Programación Orientada a Objetos
- ✅ Semana 04: Módulos, Paquetes y Entornos Virtuales

---

## 🗂️ Estructura de la Semana

```
week-05/
├── README.md                    # Este archivo
├── rubrica-evaluacion.md        # Criterios de evaluación
├── 0-assets/                    # Diagramas y recursos visuales
├── 1-teoria/                    # Material teórico
│   ├── 01-archivos-texto.md     # Lectura/escritura de archivos
│   ├── 02-formatos-datos.md     # CSV, JSON, YAML
│   ├── 03-excepciones.md        # Manejo de errores
│   └── 04-logging.md            # Sistema de logging
├── 2-practicas/                 # Ejercicios guiados
│   ├── ejercicio-01-archivos/   # Operaciones básicas con archivos
│   ├── ejercicio-02-formatos/   # Trabajar con CSV y JSON
│   ├── ejercicio-03-excepciones/# Manejo de excepciones
│   └── ejercicio-04-logging/    # Configurar logging
├── 3-proyecto/                  # Proyecto integrador
│   ├── README.md                # Log Analyzer - Analizador de logs
│   ├── starter/                 # Código inicial
│   └── .solution/               # Solución de referencia
├── 4-recursos/                  # Material complementario
│   ├── ebooks-free/
│   ├── videografia/
│   └── webgrafia/
└── 5-glosario/                  # Términos clave A-Z
```

---

## 📝 Contenidos

### 1. Teoría

| Archivo                                               | Tema                            | Duración |
| ----------------------------------------------------- | ------------------------------- | -------- |
| [01-archivos-texto.md](1-teoria/01-archivos-texto.md) | Lectura y escritura de archivos | 25 min   |
| [02-formatos-datos.md](1-teoria/02-formatos-datos.md) | CSV, JSON, YAML                 | 25 min   |
| [03-excepciones.md](1-teoria/03-excepciones.md)       | Manejo de excepciones           | 25 min   |
| [04-logging.md](1-teoria/04-logging.md)               | Sistema de logging              | 15 min   |

### 2. Prácticas

| Ejercicio                                             | Tema                       | Dificultad |
| ----------------------------------------------------- | -------------------------- | ---------- |
| [ejercicio-01](2-practicas/ejercicio-01-archivos/)    | Operaciones con archivos   | ⭐         |
| [ejercicio-02](2-practicas/ejercicio-02-formatos/)    | CSV y JSON                 | ⭐⭐       |
| [ejercicio-03](2-practicas/ejercicio-03-excepciones/) | Excepciones personalizadas | ⭐⭐       |
| [ejercicio-04](2-practicas/ejercicio-04-logging/)     | Configurar logging         | ⭐⭐       |

### 3. Proyecto

| Proyecto                             | Descripción                               |
| ------------------------------------ | ----------------------------------------- |
| [Log Analyzer](3-proyecto/README.md) | Herramienta para analizar archivos de log |

---

## ⏱️ Distribución del Tiempo

| Actividad    | Tiempo      |
| ------------ | ----------- |
| 📖 Teoría    | 1.5 horas   |
| 💻 Prácticas | 2.5 horas   |
| 🚀 Proyecto  | 2 horas     |
| **Total**    | **6 horas** |

---

## 📌 Entregables

1. **Ejercicios completados** (4 ejercicios)
2. **Proyecto Log Analyzer** funcional
3. **Cuestionario teórico** aprobado (≥70%)

---

## 🔗 Navegación

| ← Anterior                                            |           Inicio            |                                               Siguiente → |
| :---------------------------------------------------- | :-------------------------: | --------------------------------------------------------: |
| [Semana 04: Módulos y Paquetes](../week-04/README.md) | [Bootcamp](../../README.md) | [Semana 06: Matemáticas Esenciales](../week-06/README.md) |

---

## 💡 Tips de la Semana

> **Regla de oro**: Siempre usa `with` para abrir archivos. Garantiza que el archivo se cierre aunque ocurra un error.

```python
# ✅ BIEN - Context manager
with open('data.txt', 'r') as file:
    content = file.read()

# ❌ MAL - Puede dejar el archivo abierto
file = open('data.txt', 'r')
content = file.read()
file.close()  # ¿Y si hay error antes?
```

> **Excepciones específicas**: Captura excepciones específicas, no uses `except:` sin tipo.

```python
# ✅ BIEN - Específico
try:
    with open('file.txt') as f:
        data = f.read()
except FileNotFoundError:
    print("Archivo no encontrado")
except PermissionError:
    print("Sin permisos de lectura")

# ❌ MAL - Captura todo (incluso Ctrl+C)
try:
    with open('file.txt') as f:
        data = f.read()
except:
    print("Error")
```

---

_Última actualización: Diciembre 2025_
