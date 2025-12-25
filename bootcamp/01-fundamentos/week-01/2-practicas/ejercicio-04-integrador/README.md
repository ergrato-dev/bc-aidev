# 🎯 Ejercicio 04: Ejercicio Integrador

## 🎯 Objetivos

- Integrar todos los conceptos de la semana
- Crear un mini programa funcional
- Aplicar variables, operadores y estructuras de control
- Simular un escenario de IA/ML

---

## 📋 Descripción

En este ejercicio crearás un **simulador de evaluación de modelos de Machine Learning**. El programa:

1. Define métricas de varios modelos
2. Clasifica modelos según su rendimiento
3. Encuentra el mejor modelo
4. Genera un reporte

---

## 📋 Instrucciones

Abre el archivo `starter/main.py` y sigue los pasos descomentando el código indicado.

---

### Paso 1: Definir Datos de Modelos

Crea variables con las métricas de diferentes modelos:

```python
# Cada modelo tiene: nombre, accuracy, precision, recall
modelos = [
    {"name": "Random Forest", "accuracy": 0.89, "precision": 0.87, "recall": 0.91},
    {"name": "SVM", "accuracy": 0.85, "precision": 0.88, "recall": 0.82},
    # ...más modelos
]
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Calcular F1-Score

El F1-Score es la media armónica de precision y recall:

```python
f1 = 2 * (precision * recall) / (precision + recall)
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Clasificar Modelos

Usar condicionales para clasificar cada modelo:

- Accuracy ≥ 0.90: "Excelente"
- Accuracy ≥ 0.80: "Bueno"
- Accuracy ≥ 0.70: "Aceptable"
- Accuracy < 0.70: "Necesita mejora"

**Descomenta** la sección del Paso 3.

---

### Paso 4: Encontrar Mejor Modelo

Usar bucle para encontrar el modelo con mayor accuracy:

**Descomenta** la sección del Paso 4.

---

### Paso 5: Filtrar Modelos

Usar comprensiones de lista para filtrar:

```python
buenos_modelos = [m for m in modelos if m["accuracy"] >= 0.80]
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Generar Reporte

Combinar todo para generar un reporte final.

**Descomenta** la sección del Paso 6.

---

## ✅ Verificación

El programa debe mostrar:

- Tabla con métricas de cada modelo
- Clasificación de cada modelo
- El mejor modelo
- Lista de modelos que cumplen el umbral
- Reporte resumen

---

## 🏆 Reto Extra

Si terminas antes, intenta:

1. Agregar más modelos
2. Calcular el promedio de accuracy
3. Ordenar modelos por F1-Score

---

_Anterior: [Ejercicio 03](../ejercicio-03-control-flujo/) | Volver a: [Prácticas](../README.md)_
