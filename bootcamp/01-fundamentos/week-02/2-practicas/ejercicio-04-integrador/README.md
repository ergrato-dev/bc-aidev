# 🔗 Ejercicio 04: Integrador

## 🎯 Objetivo

Combinar funciones y estructuras de datos para procesar información.

---

## 📋 Pasos

### Paso 1: Función con Lista

Crea funciones que procesen listas:

```python
def calculate_average(numbers: list) -> float:
    return sum(numbers) / len(numbers)
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Función con Diccionario

Procesa diccionarios con funciones:

```python
def get_student_info(student: dict) -> str:
    return f"{student['name']} - Grade: {student['grade']}"
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Funciones que Retornan Estructuras

Crea funciones que construyen estructuras:

```python
def create_student(name: str, grades: list) -> dict:
    return {"name": name, "grades": grades, "average": sum(grades)/len(grades)}
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Procesar Lista de Diccionarios

Combina comprehensions con funciones:

```python
students = [{"name": "Ana", "grade": 85}, ...]
passed = [s for s in students if s["grade"] >= 70]
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Análisis de Datos

Implementa análisis estadístico:

```python
def analyze_grades(students: list) -> dict:
    grades = [s["grade"] for s in students]
    return {"mean": mean(grades), "max": max(grades), ...}
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Pipeline de Procesamiento

Une todo en un flujo completo:

```python
def process_data(raw_data: list) -> dict:
    cleaned = clean_data(raw_data)
    analyzed = analyze_data(cleaned)
    return format_results(analyzed)
```

**Descomenta** la sección del Paso 6.

---

## ▶️ Ejecución

```bash
cd bootcamp/week-02/2-practicas/ejercicio-04-integrador
python starter/main.py
```

---

## ✅ Resultado Esperado

```
--- Paso 1: Función con Lista ---
Notas: [85, 90, 78, 92, 88]
Promedio: 86.60

--- Paso 2: Función con Diccionario ---
Ana - Grade: 85
Carlos - Grade: 92

--- Paso 3: Funciones que Retornan Estructuras ---
{'name': 'María', 'grades': [80, 85, 90], 'average': 85.0}

--- Paso 4: Procesar Lista de Diccionarios ---
Total estudiantes: 4
Aprobados: 3
Nombres aprobados: ['Ana', 'Carlos', 'Diana']

--- Paso 5: Análisis de Datos ---
Análisis: {'count': 4, 'mean': 76.25, 'min': 55, 'max': 92}

--- Paso 6: Pipeline de Procesamiento ---
Pipeline completo ejecutado
Resultado: {'total': 5, 'valid': 4, 'average': 81.25}
```

---

_Volver a: [Prácticas](../README.md)_
