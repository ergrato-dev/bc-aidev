# 🐍 Introducción a Python y su Rol en IA

## 🎯 Objetivos

- Comprender qué es Python y por qué domina el ecosistema de IA
- Conocer la historia y filosofía de Python
- Configurar el entorno de desarrollo
- Ejecutar tu primer programa en Python

---

## 📋 Contenido

### 1. ¿Qué es Python?

Python es un **lenguaje de programación de alto nivel**, interpretado y de propósito general. Fue creado por **Guido van Rossum** y lanzado en 1991.

#### Características principales:

| Característica      | Descripción                        |
| ------------------- | ---------------------------------- |
| **Sintaxis clara**  | Código legible, similar al inglés  |
| **Interpretado**    | No requiere compilación            |
| **Tipado dinámico** | No necesitas declarar tipos        |
| **Multiparadigma**  | Soporta OOP, funcional, procedural |
| **Extensible**      | Miles de librerías disponibles     |

---

### 2. ¿Por qué Python para IA/ML?

Python se ha convertido en el **lenguaje estándar para Inteligencia Artificial**. Aquí está el por qué:

#### 🏆 Razones del dominio de Python en IA

```
┌─────────────────────────────────────────────────────────────┐
│                    ECOSISTEMA PYTHON PARA IA                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   📊 Datos          🤖 Machine Learning    🧠 Deep Learning │
│   ─────────         ──────────────────     ───────────────  │
│   • NumPy           • Scikit-learn         • TensorFlow     │
│   • Pandas          • XGBoost              • PyTorch        │
│   • Matplotlib      • LightGBM             • Keras          │
│                                                             │
│   📝 NLP            👁️ Computer Vision     🚀 Deployment    │
│   ─────────         ──────────────────     ───────────────  │
│   • NLTK            • OpenCV               • FastAPI        │
│   • spaCy           • Pillow               • Flask          │
│   • Hugging Face    • torchvision          • Docker         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Comparación con otros lenguajes

| Aspecto                 | Python           | R      | Java    | C++             |
| ----------------------- | ---------------- | ------ | ------- | --------------- |
| Curva de aprendizaje    | ⭐ Fácil         | Media  | Difícil | Muy difícil     |
| Librerías de IA         | ⭐⭐⭐ Excelente | Buena  | Media   | Media           |
| Velocidad de desarrollo | ⭐⭐⭐ Rápida    | Rápida | Lenta   | Muy lenta       |
| Comunidad IA            | ⭐⭐⭐ Enorme    | Grande | Media   | Pequeña         |
| Rendimiento             | Media            | Baja   | Alta    | ⭐⭐⭐ Muy alta |

> 💡 **Nota**: Python no es el más rápido, pero las librerías de IA están optimizadas en C/C++ internamente.

---

### 3. El Zen de Python

Python tiene una filosofía de diseño. Ejecuta esto en Python:

```python
import this
```

Los principios más importantes:

```
Beautiful is better than ugly.          # Bello es mejor que feo
Explicit is better than implicit.       # Explícito es mejor que implícito
Simple is better than complex.          # Simple es mejor que complejo
Readability counts.                     # La legibilidad cuenta
```

---

### 4. Configuración del Entorno

#### Opción A: Python directo (Recomendado para empezar)

```bash
# Verificar instalación
python --version    # Debe ser 3.11+

# Crear entorno virtual
python -m venv .venv

# Activar entorno (Linux/Mac)
source .venv/bin/activate

# Activar entorno (Windows)
.venv\Scripts\activate

# Verificar que está activo
which python
```

#### Opción B: Conda (Recomendado para Deep Learning)

```bash
# Crear entorno
conda create -n ai-bootcamp python=3.11

# Activar
conda activate ai-bootcamp
```

#### Opción C: Docker (Entorno reproducible)

```bash
# Usar docker compose del bootcamp
docker compose up --build
```

---

### 5. Tu Primer Programa

Crea un archivo llamado `hello.py`:

```python
# hello.py
# Mi primer programa en Python para IA

# Imprimir un mensaje
print("¡Hola, Inteligencia Artificial!")

# Variables básicas
name = "Estudiante"
week = 1

# f-string (Python moderno)
print(f"Bienvenido {name} a la semana {week} del bootcamp")

# Operación simple
result = 2 + 2
print(f"2 + 2 = {result}")
```

Ejecutar:

```bash
python hello.py
```

Salida esperada:

```
¡Hola, Inteligencia Artificial!
Bienvenido Estudiante a la semana 1 del bootcamp
2 + 2 = 4
```

---

### 6. Python Interactivo (REPL)

REPL = **R**ead **E**val **P**rint **L**oop

```bash
# Iniciar Python interactivo
python
```

```python
>>> 2 + 2
4
>>> "Hola" + " " + "IA"
'Hola IA'
>>> exit()
```

También puedes usar **IPython** para una experiencia mejorada:

```bash
pip install ipython
ipython
```

---

### 7. Jupyter Notebooks

Para exploración de datos y prototipos, usamos **Jupyter Notebooks**:

```bash
# Instalar
pip install jupyter

# Iniciar
jupyter notebook
```

Los notebooks permiten:

- Código + texto + visualizaciones en un solo documento
- Ejecución celda por celda
- Ideal para experimentación

> ⚠️ **Importante**: Los notebooks son para exploración. El código de producción va en archivos `.py`.

---

## 📊 Resumen Visual

```
                    PYTHON EN EL MUNDO DE LA IA
                    ===========================

    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
    │   DATOS     │ ──────► │   MODELO    │ ──────► │ PREDICCIÓN  │
    │  (Pandas)   │         │(Scikit/PyT) │         │   (API)     │
    └─────────────┘         └─────────────┘         └─────────────┘
           │                       │                       │
           ▼                       ▼                       ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                        PYTHON                                │
    │          El pegamento que une todo el pipeline              │
    └─────────────────────────────────────────────────────────────┘
```

---

## ✅ Checklist de Verificación

- [ ] Entiendo qué es Python y sus características
- [ ] Comprendo por qué Python domina en IA/ML
- [ ] Tengo Python 3.11+ instalado
- [ ] Puedo crear y activar un entorno virtual
- [ ] Ejecuté mi primer programa "Hola Mundo"
- [ ] Probé el REPL de Python

---

## 📚 Recursos Adicionales

- [Python.org - Tutorial oficial](https://docs.python.org/3/tutorial/)
- [Real Python - Guía para principiantes](https://realpython.com/python-first-steps/)
- [Python para Data Science - Kaggle](https://www.kaggle.com/learn/python)

---

_Siguiente: [02 - Variables y Tipos de Datos](02-variables-tipos-datos.md)_
