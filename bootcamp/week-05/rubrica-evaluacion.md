# 📋 Rúbrica de Evaluación - Semana 05

## Manejo de Archivos y Excepciones

---

## 📊 Distribución de Puntos

| Tipo de Evidencia | Porcentaje | Puntos      |
| ----------------- | ---------- | ----------- |
| 🧠 Conocimiento   | 30%        | 30 pts      |
| 💪 Desempeño      | 40%        | 40 pts      |
| 📦 Producto       | 30%        | 30 pts      |
| **Total**         | **100%**   | **100 pts** |

---

## 🧠 Conocimiento (30 pts)

### Cuestionario Teórico

| Criterio              | Excelente (10)                                          | Bueno (7)                                     | Regular (5)              | Insuficiente (0)          |
| --------------------- | ------------------------------------------------------- | --------------------------------------------- | ------------------------ | ------------------------- |
| **Archivos y I/O**    | Comprende modos de apertura, encoding, context managers | Entiende lo básico, algunos detalles confusos | Conocimiento superficial | No demuestra comprensión  |
| **Formatos de datos** | Domina CSV, JSON, YAML y sus casos de uso               | Conoce los formatos pero confunde usos        | Solo conoce un formato   | No comprende los formatos |
| **Excepciones**       | Entiende jerarquía, flujo try/except/else/finally       | Conoce try/except básico                      | Confunde conceptos       | No comprende excepciones  |

---

## 💪 Desempeño (40 pts)

### Ejercicios Prácticos

| Ejercicio           | Criterios                                        | Puntos |
| ------------------- | ------------------------------------------------ | ------ |
| **01: Archivos**    | Usa `with`, maneja encoding, pathlib correcto    | 10 pts |
| **02: Formatos**    | Lee/escribe CSV y JSON correctamente             | 10 pts |
| **03: Excepciones** | Excepciones específicas, personalizadas, reraise | 10 pts |
| **04: Logging**     | Configura niveles, formatters, handlers          | 10 pts |

### Criterios por Ejercicio

#### Ejercicio 01: Archivos

| Nivel        | Descripción                                                  | Puntos |
| ------------ | ------------------------------------------------------------ | ------ |
| Excelente    | Usa `with`, pathlib, maneja encoding UTF-8, método apropiado | 10     |
| Bueno        | Funciona correctamente, algunos detalles de estilo           | 7-9    |
| Regular      | Funciona pero sin context manager o con errores menores      | 4-6    |
| Insuficiente | No funciona o ignora prácticas básicas                       | 0-3    |

#### Ejercicio 02: Formatos

| Nivel        | Descripción                                                   | Puntos |
| ------------ | ------------------------------------------------------------- | ------ |
| Excelente    | CSV con DictReader/Writer, JSON con indent, manejo de errores | 10     |
| Bueno        | Funciona correctamente, formato limpio                        | 7-9    |
| Regular      | Lee/escribe pero con código ineficiente                       | 4-6    |
| Insuficiente | No maneja los formatos correctamente                          | 0-3    |

#### Ejercicio 03: Excepciones

| Nivel        | Descripción                                              | Puntos |
| ------------ | -------------------------------------------------------- | ------ |
| Excelente    | Excepciones específicas, custom exceptions, else/finally | 10     |
| Bueno        | try/except correcto, captura específica                  | 7-9    |
| Regular      | Funciona pero con except: genérico                       | 4-6    |
| Insuficiente | No maneja excepciones o las suprime                      | 0-3    |

#### Ejercicio 04: Logging

| Nivel        | Descripción                                                    | Puntos |
| ------------ | -------------------------------------------------------------- | ------ |
| Excelente    | Múltiples handlers, formateo personalizado, niveles apropiados | 10     |
| Bueno        | Logging funcional, niveles correctos                           | 7-9    |
| Regular      | Solo basicConfig, sin personalización                          | 4-6    |
| Insuficiente | Usa print() en lugar de logging                                | 0-3    |

---

## 📦 Producto (30 pts)

### Proyecto: Log Analyzer

| Criterio            | Excelente (6)                                    | Bueno (4)                | Regular (2)          | Insuficiente (0) |
| ------------------- | ------------------------------------------------ | ------------------------ | -------------------- | ---------------- |
| **Parsing de logs** | Parsea múltiples formatos, regex eficiente       | Parsea formato principal | Parsing básico       | No parsea        |
| **Filtrado**        | Filtra por nivel, fecha, patrón                  | Filtra por nivel         | Filtro limitado      | Sin filtrado     |
| **Estadísticas**    | Conteo por nivel, errores frecuentes, timeline   | Estadísticas básicas     | Solo conteo          | Sin stats        |
| **Exportación**     | Exporta a JSON/CSV, formato limpio               | Exporta a un formato     | Export parcial       | No exporta       |
| **Manejo errores**  | Excepciones apropiadas, mensajes útiles, logging | Try/except básico        | Errores no manejados | Crashes          |

### Checklist de Funcionalidades

```
□ Lectura de archivos de log
  □ Soporta múltiples archivos
  □ Detecta encoding automáticamente
  □ Maneja archivos grandes (streaming)

□ Parsing de entradas
  □ Extrae timestamp, nivel, mensaje
  □ Soporta formato estándar
  □ Maneja líneas malformadas

□ Filtrado y búsqueda
  □ Filtro por nivel (INFO, WARNING, ERROR)
  □ Filtro por rango de fechas
  □ Búsqueda por patrón/keyword

□ Análisis y estadísticas
  □ Conteo por nivel de log
  □ Errores más frecuentes
  □ Timeline de eventos

□ Exportación
  □ Resumen a JSON
  □ Logs filtrados a CSV
  □ Reporte en texto

□ Calidad de código
  □ Usa context managers
  □ Excepciones personalizadas
  □ Logging del propio analyzer
  □ Documentación clara
```

---

## 📈 Escala de Calificación

| Puntuación | Calificación | Descripción                               |
| ---------- | ------------ | ----------------------------------------- |
| 90-100     | A            | Excelente - Dominio completo              |
| 80-89      | B            | Bueno - Competente con detalles menores   |
| 70-79      | C            | Satisfactorio - Cumple requisitos mínimos |
| 60-69      | D            | En desarrollo - Necesita refuerzo         |
| 0-59       | F            | Insuficiente - No cumple requisitos       |

---

## ✅ Criterios de Aprobación

- **Mínimo 70%** en cada tipo de evidencia
- **Proyecto funcional** que analice archivos de log
- **Código limpio** siguiendo convenciones Python

---

## 🚫 Penalizaciones

| Infracción                       | Penalización |
| -------------------------------- | ------------ |
| No usar context managers         | -5 pts       |
| except: sin tipo específico      | -3 pts       |
| Usar print() para logging        | -3 pts       |
| Rutas hardcodeadas (sin pathlib) | -2 pts       |
| Sin manejo de encoding           | -2 pts       |
| Código sin documentar            | -2 pts       |
| Entrega tardía (por día)         | -5 pts       |

---

## 📝 Notas Adicionales

### Archivos y Context Managers

```python
# ✅ REQUERIDO - Siempre usar with
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# ❌ PENALIZADO - Sin context manager
f = open('file.txt')
content = f.read()
f.close()
```

### Excepciones

```python
# ✅ REQUERIDO - Excepciones específicas
try:
    process_file(path)
except FileNotFoundError:
    logger.error(f"File not found: {path}")
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON: {e}")

# ❌ PENALIZADO - Captura genérica
try:
    process_file(path)
except:
    pass
```

### Logging

```python
# ✅ REQUERIDO - Usar logging
import logging
logger = logging.getLogger(__name__)
logger.info("Processing file: %s", filename)

# ❌ PENALIZADO - Usar print
print(f"Processing file: {filename}")
```

---

_Volver a: [Semana 05](README.md)_
