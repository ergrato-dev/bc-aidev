# Ejercicio 04: Integrador
# Semana 02 - Bootcamp IA

# ============================================
# PASO 1: Función con Lista
# ============================================
print('--- Paso 1: Función con Lista ---')

# Función que procesa una lista
# Descomenta las siguientes líneas:

# def calculate_average(numbers: list) -> float:
#     """Calculate the average of a list of numbers."""
#     if not numbers:
#         return 0.0
#     return sum(numbers) / len(numbers)
#
# grades = [85, 90, 78, 92, 88]
# avg = calculate_average(grades)
# print(f"Notas: {grades}")
# print(f"Promedio: {avg:.2f}")

print()

# ============================================
# PASO 2: Función con Diccionario
# ============================================
print('--- Paso 2: Función con Diccionario ---')

# Función que procesa diccionarios
# Descomenta las siguientes líneas:

# def get_student_info(student: dict) -> str:
#     """Format student information."""
#     return f"{student['name']} - Grade: {student['grade']}"
#
# students = [
#     {"name": "Ana", "grade": 85},
#     {"name": "Carlos", "grade": 92}
# ]
#
# for student in students:
#     print(get_student_info(student))

print()

# ============================================
# PASO 3: Funciones que Retornan Estructuras
# ============================================
print('--- Paso 3: Funciones que Retornan Estructuras ---')

# Función que construye un diccionario
# Descomenta las siguientes líneas:

# def create_student(name: str, grades: list) -> dict:
#     """Create a student record with calculated average."""
#     return {
#         "name": name,
#         "grades": grades,
#         "average": sum(grades) / len(grades) if grades else 0
#     }
#
# maria = create_student("María", [80, 85, 90])
# print(maria)

print()

# ============================================
# PASO 4: Procesar Lista de Diccionarios
# ============================================
print('--- Paso 4: Procesar Lista de Diccionarios ---')

# Combina comprehensions con funciones
# Descomenta las siguientes líneas:

# students = [
#     {"name": "Ana", "grade": 85},
#     {"name": "Bob", "grade": 55},
#     {"name": "Carlos", "grade": 92},
#     {"name": "Diana", "grade": 73}
# ]
#
# # Filtrar aprobados (>= 70)
# passed = [s for s in students if s["grade"] >= 70]
# passed_names = [s["name"] for s in passed]
#
# print(f"Total estudiantes: {len(students)}")
# print(f"Aprobados: {len(passed)}")
# print(f"Nombres aprobados: {passed_names}")

print()

# ============================================
# PASO 5: Análisis de Datos
# ============================================
print('--- Paso 5: Análisis de Datos ---')

# Función de análisis estadístico
# Descomenta las siguientes líneas:

# def analyze_grades(students: list) -> dict:
#     """Analyze student grades and return statistics."""
#     grades = [s["grade"] for s in students]
#     return {
#         "count": len(grades),
#         "mean": sum(grades) / len(grades),
#         "min": min(grades),
#         "max": max(grades)
#     }
#
# students = [
#     {"name": "Ana", "grade": 85},
#     {"name": "Bob", "grade": 55},
#     {"name": "Carlos", "grade": 92},
#     {"name": "Diana", "grade": 73}
# ]
#
# analysis = analyze_grades(students)
# print(f"Análisis: {analysis}")

print()

# ============================================
# PASO 6: Pipeline de Procesamiento
# ============================================
print('--- Paso 6: Pipeline de Procesamiento ---')

# Pipeline completo de procesamiento
# Descomenta las siguientes líneas:

# def clean_data(raw_data: list) -> list:
#     """Remove invalid entries (None or negative)."""
#     return [x for x in raw_data if x is not None and x >= 0]
#
# def analyze_data(data: list) -> dict:
#     """Calculate statistics."""
#     return {
#         "count": len(data),
#         "sum": sum(data),
#         "average": sum(data) / len(data) if data else 0
#     }
#
# def format_results(analysis: dict, total: int) -> dict:
#     """Format final results."""
#     return {
#         "total": total,
#         "valid": analysis["count"],
#         "average": analysis["average"]
#     }
#
# # Datos crudos con algunos inválidos
# raw_scores = [85, None, 90, -5, 75, 75]
#
# # Pipeline
# cleaned = clean_data(raw_scores)
# analyzed = analyze_data(cleaned)
# result = format_results(analyzed, len(raw_scores))
#
# print("Pipeline completo ejecutado")
# print(f"Resultado: {result}")

print()

# ============================================
# FIN DEL EJERCICIO
# ============================================
print("=" * 50)
print("¡Ejercicio 04 completado!")
print("¡Prácticas de la Semana 02 terminadas!")
print("Siguiente: 3-proyecto/")
print("=" * 50)
