# 📘 Semana 07: NumPy para Computación Numérica

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Comprender qué es NumPy y por qué es fundamental para Data Science
- ✅ Crear y manipular arrays N-dimensionales (ndarray)
- ✅ Aplicar indexing y slicing avanzado
- ✅ Realizar operaciones vectorizadas eficientes
- ✅ Entender y aplicar broadcasting
- ✅ Usar funciones universales (ufuncs) para cálculos
- ✅ Realizar operaciones de álgebra lineal básica
- ✅ Calcular estadísticas descriptivas con NumPy

---

## 📚 Requisitos Previos

- ✅ Python básico (variables, tipos de datos)
- ✅ Estructuras de datos (listas, tuplas)
- ✅ Funciones y módulos
- ✅ Programación Orientada a Objetos (básico)
- ✅ Matemáticas básicas (álgebra, matrices conceptualmente)

---

## 🗂️ Estructura de la Semana

```
week-07/
├── README.md                    # Este archivo
├── rubrica-evaluacion.md        # Criterios de evaluación
├── 0-assets/                    # Diagramas y visualizaciones
├── 1-teoria/                    # Material teórico
│   ├── 01-introduccion-numpy.md
│   ├── 02-creacion-arrays.md
│   ├── 03-indexing-slicing.md
│   └── 04-operaciones-broadcasting.md
├── 2-practicas/                 # Ejercicios guiados
│   ├── ejercicio-01-arrays/
│   ├── ejercicio-02-indexing/
│   ├── ejercicio-03-operaciones/
│   └── ejercicio-04-estadisticas/
├── 3-proyecto/                  # Proyecto integrador
│   ├── starter/
│   └── README.md
├── 4-recursos/                  # Material complementario
│   ├── ebooks-free/
│   ├── videografia/
│   └── webgrafia/
└── 5-glosario/                  # Términos clave
    └── README.md
```

---

## 📝 Contenidos

### 1. Teoría (1.5 horas)

| Archivo                                                                   | Tema                                        | Duración |
| ------------------------------------------------------------------------- | ------------------------------------------- | -------- |
| [01-introduccion-numpy.md](1-teoria/01-introduccion-numpy.md)             | ¿Qué es NumPy? ndarray vs listas            | 20 min   |
| [02-creacion-arrays.md](1-teoria/02-creacion-arrays.md)                   | Crear arrays: zeros, ones, arange, linspace | 25 min   |
| [03-indexing-slicing.md](1-teoria/03-indexing-slicing.md)                 | Indexing, slicing, fancy indexing           | 25 min   |
| [04-operaciones-broadcasting.md](1-teoria/04-operaciones-broadcasting.md) | Operaciones vectorizadas y broadcasting     | 20 min   |

### 2. Prácticas (2.5 horas)

| Ejercicio                                                           | Tema                                | Duración |
| ------------------------------------------------------------------- | ----------------------------------- | -------- |
| [ejercicio-01-arrays](2-practicas/ejercicio-01-arrays/)             | Creación y atributos de arrays      | 35 min   |
| [ejercicio-02-indexing](2-practicas/ejercicio-02-indexing/)         | Indexing y slicing multidimensional | 35 min   |
| [ejercicio-03-operaciones](2-practicas/ejercicio-03-operaciones/)   | Operaciones vectorizadas y ufuncs   | 40 min   |
| [ejercicio-04-estadisticas](2-practicas/ejercicio-04-estadisticas/) | Estadísticas y álgebra lineal       | 40 min   |

### 3. Proyecto (2 horas)

| Proyecto                              | Descripción                                 |
| ------------------------------------- | ------------------------------------------- |
| [Analizador de Imágenes](3-proyecto/) | Procesamiento de imágenes como arrays NumPy |

---

## ⏱️ Distribución del Tiempo

```
Total: 6 horas

┌─────────────────────────────────────────────────────────┐
│ Teoría        ███████░░░░░░░░░░░░░░░░░░░░░░  1.5h (25%) │
│ Prácticas     ████████████░░░░░░░░░░░░░░░░░  2.5h (42%) │
│ Proyecto      ████████░░░░░░░░░░░░░░░░░░░░░  2.0h (33%) │
└─────────────────────────────────────────────────────────┘
```

---

## 📌 Entregables

### Ejercicios Prácticos

- [ ] `ejercicio-01-arrays/starter/main.py` completado
- [ ] `ejercicio-02-indexing/starter/main.py` completado
- [ ] `ejercicio-03-operaciones/starter/main.py` completado
- [ ] `ejercicio-04-estadisticas/starter/main.py` completado

### Proyecto

- [ ] `image_processor.py` - Funciones de procesamiento
- [ ] `filters.py` - Filtros de imagen implementados
- [ ] `main.py` - CLI funcional
- [ ] Imagen procesada de ejemplo

---

## 🔧 Instalación

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Instalar NumPy
pip install numpy

# Verificar instalación
python -c "import numpy as np; print(f'NumPy {np.__version__}')"
```

---

## 💡 Conceptos Clave

### ¿Por qué NumPy?

```python
# ❌ Listas Python - Lento para operaciones numéricas
python_list = [1, 2, 3, 4, 5]
result = [x * 2 for x in python_list]  # Loop implícito

# ✅ NumPy - Operaciones vectorizadas (C bajo el capó)
import numpy as np
numpy_array = np.array([1, 2, 3, 4, 5])
result = numpy_array * 2  # Sin loop, ejecutado en C
```

### Velocidad: NumPy vs Listas

| Operación         | Lista Python | NumPy | Speedup  |
| ----------------- | ------------ | ----- | -------- |
| Suma 1M elementos | ~50ms        | ~1ms  | **50x**  |
| Multiplicación    | ~80ms        | ~1ms  | **80x**  |
| Producto punto    | ~200ms       | ~2ms  | **100x** |

### El ndarray

```python
import numpy as np

# Crear array 2D (matriz)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(matrix.shape)    # (2, 3) - 2 filas, 3 columnas
print(matrix.dtype)    # int64
print(matrix.ndim)     # 2 dimensiones
print(matrix.size)     # 6 elementos totales
```

---

## 🔗 Navegación

| Anterior                                 | Índice                         | Siguiente                                   |
| ---------------------------------------- | ------------------------------ | ------------------------------------------- |
| [← Semana 06: POO](../week-06/README.md) | [📚 Bootcamp](../../README.md) | [Semana 08: Pandas →](../week-08/README.md) |

---

## 📚 Recursos Recomendados

- 📖 [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- 📖 [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- 🎥 [NumPy Tutorial - freeCodeCamp](https://www.youtube.com/watch?v=QUT1VHiLmmI)
- 📝 [100 NumPy Exercises](https://github.com/rougier/numpy-100)

---

_Semana 07 de 36 | Módulo: Fundamentos_
