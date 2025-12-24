# 📋 Rúbrica de Evaluación - Semana 07

## 🎯 NumPy para Computación Numérica

---

## 📊 Distribución de Puntuación

| Tipo de Evidencia | Peso     | Puntos      |
| ----------------- | -------- | ----------- |
| 🧠 Conocimiento   | 30%      | 30 pts      |
| 💪 Desempeño      | 40%      | 40 pts      |
| 📦 Producto       | 30%      | 30 pts      |
| **Total**         | **100%** | **100 pts** |

---

## 🧠 Conocimiento (30 puntos)

### Conceptos Teóricos

| Criterio              | Excelente (10)                                    | Bueno (7)                         | Regular (4)                    | Insuficiente (0) |
| --------------------- | ------------------------------------------------- | --------------------------------- | ------------------------------ | ---------------- |
| **ndarray vs listas** | Explica diferencias de memoria, tipos y velocidad | Conoce diferencias principales    | Diferencia básica              | No diferencia    |
| **Broadcasting**      | Explica reglas y predice resultados               | Entiende concepto y casos comunes | Conoce el término              | No comprende     |
| **Vectorización**     | Aplica y explica por qué es eficiente             | Usa operaciones vectorizadas      | Mezcla loops con vectorización | Solo usa loops   |

### Preguntas de Verificación

1. ¿Por qué NumPy es más rápido que las listas de Python para operaciones numéricas?
2. ¿Qué significa que un array tenga shape `(3, 4, 2)`?
3. ¿Cuál es la diferencia entre `np.zeros()` y `np.empty()`?
4. Explica qué es broadcasting con un ejemplo
5. ¿Qué es una ufunc y para qué sirve?

---

## 💪 Desempeño (40 puntos)

### Ejercicio 01: Creación de Arrays (10 pts)

| Criterio       | Puntos | Descripción                                  |
| -------------- | ------ | -------------------------------------------- |
| Arrays básicos | 3      | Crear con `array()`, `zeros()`, `ones()`     |
| Rangos         | 3      | Usar `arange()` y `linspace()` correctamente |
| Atributos      | 2      | Inspeccionar shape, dtype, ndim, size        |
| Reshape        | 2      | Cambiar forma de arrays correctamente        |

### Ejercicio 02: Indexing y Slicing (10 pts)

| Criterio        | Puntos | Descripción                     |
| --------------- | ------ | ------------------------------- |
| Indexing básico | 2      | Acceder elementos por índice    |
| Slicing 1D      | 2      | Usar start:stop:step            |
| Slicing 2D/3D   | 3      | Slicing multidimensional        |
| Fancy indexing  | 3      | Indexing con arrays y booleanos |

### Ejercicio 03: Operaciones Vectorizadas (10 pts)

| Criterio                        | Puntos | Descripción                        |
| ------------------------------- | ------ | ---------------------------------- |
| Operaciones elemento a elemento | 3      | +, -, \*, /, \*\* entre arrays     |
| Broadcasting                    | 3      | Operar arrays de diferentes shapes |
| Ufuncs                          | 2      | Usar np.sin, np.exp, np.sqrt, etc. |
| Agregaciones                    | 2      | sum, mean, std, min, max con axis  |

### Ejercicio 04: Estadísticas y Álgebra Lineal (10 pts)

| Criterio                  | Puntos | Descripción                           |
| ------------------------- | ------ | ------------------------------------- |
| Estadísticas descriptivas | 3      | Media, mediana, varianza, percentiles |
| Producto matricial        | 3      | np.dot, @ operator                    |
| Transposición             | 2      | .T y np.transpose                     |
| Operaciones de conjunto   | 2      | unique, where, argmax, argmin         |

---

## 📦 Producto (30 puntos)

### Proyecto: Analizador de Imágenes

| Criterio          | Excelente (30)                                  | Bueno (22)             | Regular (15)             | Insuficiente (0)  |
| ----------------- | ----------------------------------------------- | ---------------------- | ------------------------ | ----------------- |
| **Funcionalidad** | Todas las funciones implementadas y funcionando | 80% funciones          | 60% funciones            | <50% funciones    |
| **Código**        | Vectorizado, eficiente, bien documentado        | Mayormente vectorizado | Mezcla loops/vectorizado | Solo loops        |
| **Filtros**       | 4+ filtros implementados correctamente          | 3 filtros              | 2 filtros                | 1 o ningún filtro |

### Funciones Requeridas

```python
# Mínimo requerido para aprobar
def load_image(path: str) -> np.ndarray:
    """Cargar imagen como array NumPy."""
    pass

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convertir imagen RGB a escala de grises."""
    pass

def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Ajustar brillo de la imagen."""
    pass

def apply_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    """Aplicar umbral binario a imagen."""
    pass
```

### Filtros Adicionales (Bonus)

| Filtro                      | Puntos Extra |
| --------------------------- | ------------ |
| Inversión de colores        | +2           |
| Ajuste de contraste         | +2           |
| Blur (convolución básica)   | +3           |
| Detección de bordes (Sobel) | +3           |

---

## 📈 Escala de Calificación

| Rango  | Calificación | Descripción                   |
| ------ | ------------ | ----------------------------- |
| 90-100 | A            | Excelente dominio de NumPy    |
| 80-89  | B            | Buen manejo, minor issues     |
| 70-79  | C            | Competente, cumple requisitos |
| 60-69  | D            | Necesita mejorar              |
| <60    | F            | No aprobado                   |

---

## ✅ Criterios de Aprobación

- [ ] Mínimo **70%** en cada tipo de evidencia
- [ ] Todos los ejercicios completados
- [ ] Proyecto funcional con mínimo 3 funciones
- [ ] Código usa operaciones vectorizadas (no loops innecesarios)
- [ ] Código documentado con docstrings

---

## 🚫 Penalizaciones

| Infracción                                      | Penalización    |
| ----------------------------------------------- | --------------- |
| Uso excesivo de loops en lugar de vectorización | -10 pts         |
| Código sin documentación                        | -5 pts          |
| Entrega tardía (por día)                        | -5 pts          |
| Código copiado                                  | -100% + reporte |

---

## 📝 Entrega

### Formato

- Archivos `.py` ejecutables
- Seguir estructura de carpetas del proyecto
- Incluir `requirements.txt` si usa dependencias adicionales

### Fecha Límite

- **Ejercicios**: Fin de la semana
- **Proyecto**: Domingo 23:59

### Método de Entrega

- Push a repositorio del bootcamp
- Verificar que código ejecuta sin errores

---

## 💡 Consejos para Máxima Puntuación

1. **Vectoriza todo**: Evita loops `for` cuando NumPy tiene una función
2. **Usa broadcasting**: Entiende las reglas y aplícalas
3. **Documenta**: Explica qué hace cada función
4. **Prueba edge cases**: Arrays vacíos, shapes incompatibles
5. **Lee la documentación**: NumPy tiene funciones para casi todo

---

_Rúbrica v1.0 | Semana 07 | NumPy_
