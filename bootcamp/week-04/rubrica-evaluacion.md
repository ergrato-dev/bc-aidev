# 📋 Rúbrica de Evaluación - Semana 04

## Módulos, Paquetes y Entornos Virtuales

---

## 🧠 Conocimiento (30%)

### Cuestionario Teórico

| Criterio     | Excelente (100%)                                    | Bueno (75%)                       | Suficiente (50%)          | Insuficiente (0%)            |
| ------------ | --------------------------------------------------- | --------------------------------- | ------------------------- | ---------------------------- |
| **Módulos**  | Explica módulo vs script, `__name__`, `__file__`    | Entiende diferencia módulo/script | Concepto básico de módulo | No comprende módulos         |
| **Paquetes** | Domina `__init__.py`, namespace packages, `__all__` | Entiende estructura de paquetes   | Crea paquetes básicos     | No comprende paquetes        |
| **Imports**  | Domina absolutos, relativos, `sys.path`             | Usa imports correctamente         | Imports básicos funcionan | Errores constantes de import |
| **Entornos** | Explica aislamiento, `pip freeze`, reproducibilidad | Crea y usa entornos virtuales     | Concepto básico de venv   | No usa entornos virtuales    |

---

## 💪 Desempeño (40%)

### Ejercicio 01: Módulos Propios

| Criterio             | Puntos | Descripción                               |
| -------------------- | ------ | ----------------------------------------- |
| Módulo creado        | 5      | Archivo `.py` con funciones reutilizables |
| Importación correcta | 5      | `import` y `from...import` funcionan      |
| `if __name__`        | 5      | Módulo ejecutable y también importable    |
| Docstrings           | 5      | Documentación en módulo y funciones       |
| **Total**            | **20** |                                           |

### Ejercicio 02: Paquetes

| Criterio            | Puntos | Descripción                        |
| ------------------- | ------ | ---------------------------------- |
| Estructura correcta | 5      | Carpetas con `__init__.py`         |
| Submódulos          | 5      | Al menos 2 submódulos funcionales  |
| `__init__.py` útil  | 5      | Expone API pública del paquete     |
| `__all__` definido  | 5      | Control de `from package import *` |
| **Total**           | **20** |                                    |

### Ejercicio 03: Imports

| Criterio               | Puntos | Descripción                                    |
| ---------------------- | ------ | ---------------------------------------------- |
| Imports absolutos      | 5      | Funcionan desde cualquier ubicación            |
| Imports relativos      | 5      | Funcionan dentro del paquete                   |
| Sin errores circulares | 5      | Diseño evita dependencias circulares           |
| Organización           | 5      | Imports ordenados (stdlib, third-party, local) |
| **Total**              | **20** |                                                |

### Ejercicio 04: Entornos Virtuales

| Criterio           | Puntos | Descripción                          |
| ------------------ | ------ | ------------------------------------ |
| Crear venv         | 5      | `python -m venv .venv` funciona      |
| Activar/desactivar | 5      | Sabe activar en su sistema operativo |
| Instalar paquetes  | 5      | `pip install` dentro del entorno     |
| requirements.txt   | 5      | Genera y usa archivo de dependencias |
| **Total**          | **20** |                                      |

**Total Desempeño: 80 puntos → 40%**

---

## 📦 Producto (30%)

### Proyecto: CLI Utils Package

| Criterio          | Excelente (100%)                             | Bueno (75%)                     | Suficiente (50%)                | Insuficiente (0%)       |
| ----------------- | -------------------------------------------- | ------------------------------- | ------------------------------- | ----------------------- |
| **Estructura**    | Paquete completo con setup.py/pyproject.toml | Estructura de paquete correcta  | Paquete básico funcional        | No es un paquete válido |
| **Funcionalidad** | 4+ utilidades CLI funcionando                | 3 utilidades funcionando        | 2 utilidades básicas            | Menos de 2 utilidades   |
| **Instalable**    | `pip install -e .` funciona                  | Instalación con ajustes menores | Requiere instrucciones manuales | No instalable           |
| **CLI**           | Entry points configurados                    | CLI funciona con scripts        | CLI manual                      | Sin interfaz CLI        |
| **Documentación** | README completo, docstrings, ejemplos        | README con instrucciones        | README básico                   | Sin documentación       |
| **Código**        | Type hints, clean code, tests                | Type hints, código limpio       | Código funcional                | Código desorganizado    |

### Rúbrica Detallada del Proyecto

| Componente                           | Puntos | Criterios                       |
| ------------------------------------ | ------ | ------------------------------- |
| **Estructura del paquete**           | 15     |                                 |
| - `__init__.py` correcto             | 5      | Expone API pública              |
| - Submódulos organizados             | 5      | Separación por responsabilidad  |
| - `pyproject.toml` o `setup.py`      | 5      | Metadatos completos             |
| **Utilidades CLI**                   | 25     |                                 |
| - File utils (contar líneas, buscar) | 7      | Funcionalidad completa          |
| - Text utils (formatear, limpiar)    | 6      | Funcionalidad completa          |
| - System utils (info sistema)        | 6      | Funcionalidad completa          |
| - Custom util (creatividad)          | 6      | Utilidad adicional original     |
| **Instalación**                      | 15     |                                 |
| - `pip install -e .` funciona        | 10     | Instalación sin errores         |
| - Entry points CLI                   | 5      | Comandos accesibles globalmente |
| **Calidad**                          | 15     |                                 |
| - Type hints                         | 5      | Anotaciones de tipos            |
| - Docstrings                         | 5      | Documentación de funciones      |
| - Manejo de errores                  | 5      | Excepciones apropiadas          |
| **Total**                            | **70** |                                 |

---

## 📊 Tabla de Conversión

| Puntos | Calificación | Nivel        |
| ------ | ------------ | ------------ |
| 90-100 | A            | Excelente    |
| 80-89  | B            | Muy Bueno    |
| 70-79  | C            | Bueno        |
| 60-69  | D            | Suficiente   |
| < 60   | F            | Insuficiente |

---

## ✅ Checklist de Entrega

### Ejercicios

- [ ] Ejercicio 01: Módulo `math_utils.py` funcional
- [ ] Ejercicio 02: Paquete `data_tools/` estructurado
- [ ] Ejercicio 03: Imports sin errores
- [ ] Ejercicio 04: Entorno virtual con requirements.txt

### Proyecto

- [ ] Paquete `cli_utils/` con estructura correcta
- [ ] Al menos 4 utilidades implementadas
- [ ] `pyproject.toml` con metadatos
- [ ] Instalable con `pip install -e .`
- [ ] README.md con instrucciones de uso
- [ ] Entry points configurados (comandos CLI)

---

## 🎯 Criterios de Aprobación

- ✅ Mínimo **70%** en cada categoría
- ✅ Proyecto funcional e instalable
- ✅ Todos los ejercicios completados
- ✅ Entorno virtual utilizado durante el desarrollo

---

## 📝 Notas

- El código debe seguir PEP 8
- Usar type hints en todas las funciones
- Los imports deben estar organizados (isort)
- El paquete debe ser instalable localmente

---

_Rúbrica Semana 04 · Última actualización: Diciembre 2025_
