# 🤖 Instrucciones para GitHub Copilot

## 📋 Contexto del Bootcamp

Este es un **Bootcamp de Inteligencia Artificial: Zero to Hero** estructurado para llevar a estudiantes de cero a héroe en desarrollo de IA y Machine Learning.

### 📊 Datos del Bootcamp

- **Duración**: 36 semanas (~9 meses)
- **Dedicación semanal**: 6 horas
- **Total de horas**: ~216 horas
- **Nivel de salida**: Desarrollador IA/ML Junior
- **Enfoque**: Python moderno, Machine Learning, Deep Learning, LLMs
- **Stack**: Python, NumPy, Pandas, Scikit-learn, TensorFlow/PyTorch, Hugging Face

---

## 🎯 Objetivos de Aprendizaje

Al finalizar el bootcamp, los estudiantes serán capaces de:

- ✅ Dominar Python para ciencia de datos e IA
- ✅ Comprender fundamentos matemáticos (álgebra lineal, estadística, cálculo)
- ✅ Manipular y visualizar datos con NumPy, Pandas, Matplotlib
- ✅ Implementar algoritmos de Machine Learning con Scikit-learn
- ✅ Construir redes neuronales con TensorFlow/PyTorch
- ✅ Desarrollar modelos de Deep Learning (CNNs, RNNs, Transformers)
- ✅ Trabajar con NLP y LLMs usando Hugging Face
- ✅ Desplegar modelos en producción (MLOps básico)

---

## 📚 Estructura del Bootcamp

### Distribución por Módulos

#### **Fundamentos (Semanas 1-8)** - 48 horas

- Python moderno para Data Science
- Matemáticas esenciales (álgebra lineal, estadística, probabilidad)
- NumPy para computación numérica
- Pandas para manipulación de datos
- Matplotlib y Seaborn para visualización
- SQL básico para datos

#### **Machine Learning (Semanas 9-18)** - 60 horas

- Fundamentos de ML y tipos de aprendizaje
- Regresión lineal y logística
- Árboles de decisión y Random Forest
- SVM, KNN, Naive Bayes
- Clustering (K-Means, DBSCAN)
- Feature Engineering y selección de características
- Validación cruzada y métricas de evaluación
- Scikit-learn avanzado

#### **Deep Learning (Semanas 19-28)** - 60 horas

- Redes neuronales desde cero
- TensorFlow y Keras fundamentals
- PyTorch fundamentals
- Redes Neuronales Convolucionales (CNNs)
- Redes Neuronales Recurrentes (RNNs, LSTM, GRU)
- Arquitectura Transformer
- Transfer Learning
- Regularización y optimización

#### **Especialización (Semanas 29-34)** - 36 horas

- Procesamiento de Lenguaje Natural (NLP)
- Hugging Face Transformers
- Large Language Models (LLMs)
- Fine-tuning y RAG
- Computer Vision avanzado
- MLOps: deployment y APIs

#### **Proyecto Final (Semanas 35-36)** - 12 horas

- Proyecto end-to-end integrando todo lo aprendido
- Documentación y presentación
- Deploy en producción

---

## 🗂️ Estructura de Carpetas

Cada semana sigue esta estructura estándar:

```
bootcamp/week-XX/
├── README.md                 # Descripción y objetivos de la semana
├── rubrica-evaluacion.md     # Criterios de evaluación detallados
├── 0-assets/                 # Imágenes, diagramas, datasets
├── 1-teoria/                 # Material teórico (archivos .md y notebooks)
├── 2-practicas/              # Ejercicios guiados paso a paso
├── 3-proyecto/               # Proyecto semanal integrador
├── 4-recursos/               # Recursos adicionales
│   ├── ebooks-free/          # Libros electrónicos gratuitos
│   ├── videografia/          # Videos y tutoriales recomendados
│   └── webgrafia/            # Enlaces y documentación
└── 5-glosario/               # Términos clave de la semana (A-Z)
    └── README.md
```

### 📁 Carpetas Raíz

- **`_assets/`**: Recursos visuales globales (logos, headers, etc.)
- **`_docs/`**: Documentación general que aplica a todo el bootcamp
- **`_scripts/`**: Scripts de automatización y utilidades
- **`bootcamp/`**: Contenido semanal del bootcamp

---

## 🎓 Componentes de Cada Semana

### 1. **Teoría** (1-teoria/)

- Archivos markdown y Jupyter Notebooks con explicaciones conceptuales
- Ejemplos de código con comentarios claros
- Diagramas y visualizaciones (preferir SVG)
- Fórmulas matemáticas con LaTeX
- Referencias a documentación oficial

### 2. **Prácticas** (2-practicas/)

- Ejercicios guiados paso a paso
- Incremento progresivo de dificultad
- Soluciones comentadas
- Casos de uso del mundo real

#### 📋 Formato de Ejercicios

Los ejercicios son **tutoriales guiados**, NO tareas con TODOs. El estudiante aprende ejecutando y modificando código:

**README.md del ejercicio:**

```markdown
### Paso 1: Nombre del Concepto

Explicación del concepto con ejemplo:

\`\`\`python

# Ejemplo explicativo

import numpy as np
result = np.array([1, 2, 3]).mean()
\`\`\`

**Abre `starter/main.py`** y descomenta la sección correspondiente.
```

**starter/main.py:**

```python
# ============================================
# PASO 1: Nombre del Concepto
# ============================================
print('--- Paso 1: Nombre del Concepto ---')

# Explicación breve del concepto
# Descomenta las siguientes líneas:
# import numpy as np
# data = np.array([1, 2, 3, 4, 5])
# print('Media:', data.mean())

print()
```

**solution/main.py:**

```python
# ============================================
# PASO 1: Nombre del Concepto
# ============================================
print('--- Paso 1: Nombre del Concepto ---')

import numpy as np
data = np.array([1, 2, 3, 4, 5])
print('Media:', data.mean())
```

#### ❌ NO usar este formato en ejercicios:

```python
# ❌ INCORRECTO - Este formato es para PROYECTOS, no ejercicios
result = None  # TODO: Implementar
```

#### ✅ Usar este formato en ejercicios:

```python
# ✅ CORRECTO - Código comentado para descomentar
# Descomenta las siguientes líneas:
# result = data.mean()
# print('Resultado:', result)
```

### 3. **Proyecto** (3-proyecto/)

- Proyecto integrador que consolida lo aprendido
- README.md con instrucciones claras
- Código inicial o plantillas cuando sea apropiado
- Criterios de evaluación específicos

#### 📋 Formato de Proyecto (con TODOs)

A diferencia de los ejercicios, el proyecto SÍ usa TODOs para que el estudiante implemente desde cero:

**starter/main.py:**

```python
# ============================================
# FUNCIÓN: train_model
# Entrenar un modelo de clasificación
# ============================================

def train_model(X_train, y_train):
    """
    Entrena un modelo de clasificación.

    Args:
        X_train: Features de entrenamiento
        y_train: Labels de entrenamiento

    Returns:
        model: Modelo entrenado
    """
    # TODO: Implementar lógica de entrenamiento
    pass
```

### 4. **Recursos** (4-recursos/)

- **ebooks-free/**: Libros gratuitos relevantes
- **videografia/**: Videos tutoriales complementarios
- **webgrafia/**: Enlaces a documentación y artículos

### 5. **Glosario** (5-glosario/)

- Términos técnicos ordenados alfabéticamente
- Definiciones claras y concisas
- Fórmulas matemáticas cuando aplique
- Ejemplos de código cuando sea útil

---

## 📝 Convenciones de Código

### Estilo Python Moderno

```python
# ✅ BIEN - Type hints
def calculate_accuracy(y_true: list, y_pred: list) -> float:
    """Calcula la precisión del modelo."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)

# ✅ BIEN - f-strings
model_name = "RandomForest"
print(f"Entrenando modelo: {model_name}")

# ✅ BIEN - List comprehensions
squared = [x ** 2 for x in range(10)]

# ✅ BIEN - Context managers
with open('data.csv', 'r') as file:
    content = file.read()

# ✅ BIEN - Pathlib para rutas
from pathlib import Path
data_path = Path('data') / 'train.csv'

# ❌ MAL - Concatenación de strings
print("Modelo: " + model_name)  # Usar f-strings

# ❌ MAL - Rutas con strings
data_path = 'data/train.csv'  # Usar pathlib
```

### Nomenclatura

- **Variables y funciones**: snake_case
- **Constantes globales**: UPPER_SNAKE_CASE
- **Clases**: PascalCase
- **Archivos**: snake_case.py
- **Notebooks**: XX_nombre_descriptivo.ipynb

### Imports

```python
# ✅ BIEN - Orden estándar de imports
# 1. Standard library
import os
from pathlib import Path

# 2. Third party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 3. Local
from utils import load_data
```

---

## 🧪 Testing

El bootcamp enseña testing de modelos ML con **pytest**.

### Estructura de Tests

```python
# test_model.py
import pytest
import numpy as np

def test_model_accuracy():
    """Test que el modelo alcanza precisión mínima."""
    model = train_model(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    assert accuracy >= 0.8, f"Accuracy {accuracy} < 0.8"

def test_prediction_shape():
    """Test que las predicciones tienen la forma correcta."""
    predictions = model.predict(X_test)
    assert predictions.shape == y_test.shape
```

---

## 📖 Documentación

### README.md de Semana

Debe incluir:

1. **Título y descripción**
2. **🎯 Objetivos de aprendizaje**
3. **📚 Requisitos previos**
4. **🗂️ Estructura de la semana**
5. **📝 Contenidos** (con enlaces a teoría/prácticas)
6. **⏱️ Distribución del tiempo** (6 horas)
7. **📌 Entregables**
8. **🔗 Navegación** (anterior/siguiente semana)

### Archivos de Teoría

```markdown
# Título del Tema

## 🎯 Objetivos

- Objetivo 1
- Objetivo 2

## 📋 Contenido

### 1. Introducción

### 2. Conceptos Clave

### 3. Fundamentos Matemáticos

### 4. Implementación en Python

### 5. Ejemplos Prácticos

## 📚 Recursos Adicionales

## ✅ Checklist de Verificación
```

---

## 🎨 Recursos Visuales y Estándares de Diseño

### Formato de Assets

- ✅ **Preferir SVG** para todos los diagramas, iconos y gráficos
- ❌ **NO usar ASCII art** para diagramas o visualizaciones
- ✅ Usar PNG/JPG solo para screenshots o fotografías
- ✅ Optimizar imágenes antes de incluirlas

### Organización de Assets Semanales (0-assets/)

- ✅ **Numerar archivos** en orden lógico de consulta: `01-diagrama-flujo.svg`, `02-arquitectura-modelo.svg`
- ✅ **Vincular en teoría**: Todo asset debe estar referenciado en `1-teoria/` o donde agregue valor
- ✅ **Nombres descriptivos**: `03-confusion-matrix.png` en lugar de `imagen3.png`
- ✅ **No assets huérfanos**: Si no se usa en ningún archivo, no debe existir
- ✅ **Agrupación por tema** (opcional): `01a-`, `01b-` para assets relacionados

```markdown
<!-- Ejemplo de vinculación en teoría -->

![Arquitectura de Red Neuronal](../0-assets/01-arquitectura-red-neuronal.svg)

<!-- Con descripción accesible -->

![Matriz de confusión mostrando TP, TN, FP, FN](../0-assets/02-confusion-matrix.png)
```

### Tema Visual

- 🌙 **Tema dark** para todos los assets visuales
- ❌ **Sin degradés** (gradients) en diseños
- ✅ Colores sólidos y contrastes claros
- ✅ Paleta consistente basada en Python (#3776AB) y AI (#FF6F00)

### Tipografía

- ✅ **Fuentes sans-serif** exclusivamente
- ✅ Recomendadas: Inter, Roboto, Open Sans, System UI
- ❌ **NO usar fuentes serif** (Times, Georgia, etc.)
- ✅ Mantener jerarquía visual clara

### Calidad de SVGs - Verificación de Textos

**Problema común**: Textos que desbordan contenedores o se superponen con bordes.

**Estrategia de prevención:**

1. **Padding interno obligatorio**

   - Mínimo 8-12px de espacio entre texto y bordes del contenedor
   - Nunca colocar texto pegado al borde

2. **Dimensionamiento de contenedores**

   - Calcular ancho mínimo: `(caracteres × 8px) + 24px` para fuente 14px
   - Altura mínima: `líneas × line-height + 16px`
   - Preferir contenedores más grandes que ajustados

3. **Textos largos**

   - Dividir en múltiples líneas si supera 25-30 caracteres
   - Usar abreviaciones técnicas estándar cuando sea apropiado
   - Considerar tooltip o leyenda externa para descripciones largas

4. **Verificación antes de commit**

   - Abrir SVG en navegador al 100% y 150% de zoom
   - Verificar que ningún texto toque o cruce bordes
   - Comprobar legibilidad en tema dark

5. **Atributos SVG recomendados**

   ```xml
   <!-- Texto con espacio seguro -->
   <rect x="10" y="10" width="200" height="50" rx="8"/>
   <text x="20" y="40" font-size="14">Texto con padding</text>

   <!-- Texto centrado (más seguro) -->
   <text x="110" y="40" text-anchor="middle">Texto centrado</text>
   ```

6. **Checklist de calidad SVG**
   - [ ] Padding mínimo 8px en todos los lados
   - [ ] Textos no tocan bordes
   - [ ] Legible a 100% y 150% zoom
   - [ ] Funciona en tema dark
   - [ ] Fuente sans-serif utilizada

### Fórmulas Matemáticas

- ✅ Usar LaTeX para fórmulas en markdown
- ✅ Usar MathJax o KaTeX para renderizado
- ✅ Incluir explicación textual de cada fórmula

---

## 🌐 Idioma y Nomenclatura

### Código y Comentarios Técnicos

- ✅ **Nomenclatura en inglés** (variables, funciones, clases)
- ✅ **Comentarios de código en inglés**
- ✅ Usar términos técnicos estándar de la industria

```python
# ✅ CORRECTO - inglés
def train_neural_network(X_train, y_train, epochs=100):
    """Train a neural network model."""
    model = Sequential([
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model.fit(X_train, y_train, epochs=epochs)

# ❌ INCORRECTO - español en código
def entrenar_red_neuronal(X_entrenamiento, y_entrenamiento, epocas=100):
    """Entrenar un modelo de red neuronal."""
    pass
```

### Documentación

- ✅ **Documentación en español** (READMEs, teoría, guías)
- ✅ Explicaciones y tutoriales en español
- ✅ Comentarios educativos en español cuando expliquen conceptos

```python
# ✅ CORRECTO - código en inglés, explicación en español
def calculate_gradient(X, y, weights):
    """
    Calcula el gradiente para descenso de gradiente.

    En Machine Learning, el gradiente nos indica la dirección
    de máximo crecimiento de la función de pérdida.
    Para minimizar, nos movemos en dirección opuesta.
    """
    predictions = X @ weights
    error = predictions - y
    gradient = (2 / len(y)) * X.T @ error
    return gradient
```

---

## 🔐 Mejores Prácticas

### Código Limpio

- Nombres descriptivos y significativos
- Funciones pequeñas con una sola responsabilidad
- Docstrings en todas las funciones públicas
- Type hints cuando sea posible
- Evitar anidamiento profundo

### Reproducibilidad

- Fijar seeds aleatorios para reproducibilidad
- Documentar versiones de librerías
- Usar requirements.txt o environment.yml
- Guardar modelos y checkpoints

```python
# ✅ BIEN - Reproducibilidad
import numpy as np
import random
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
```

### Gestión de Datos

- No subir datasets grandes al repositorio
- Usar .gitignore para datos y modelos
- Documentar fuentes de datos
- Incluir scripts de descarga cuando aplique

---

## 📊 Evaluación

Cada semana incluye **tres tipos de evidencias**:

1. **Conocimiento 🧠** (30%): Evaluaciones teóricas, cuestionarios
2. **Desempeño 💪** (40%): Ejercicios prácticos, notebooks completados
3. **Producto 📦** (30%): Proyecto entregable funcional

### Criterios de Aprobación

- Mínimo **70%** en cada tipo de evidencia
- Entrega puntual de proyectos
- Código funcional y bien documentado
- Modelos con métricas mínimas especificadas

---

## 🚀 Metodología de Aprendizaje

### Estrategias Didácticas

- **Aprendizaje Basado en Proyectos (ABP)**: Proyectos semanales integradores
- **Práctica Deliberada**: Ejercicios incrementales
- **Kaggle Challenges**: Problemas reales de competiciones
- **Code Review**: Revisión de código entre estudiantes
- **Paper Reading**: Lectura guiada de papers fundamentales

### Distribución del Tiempo (6h/semana)

- **Teoría**: 1.5 horas
- **Prácticas**: 2.5 horas
- **Proyecto**: 2 horas

---

## 🤖 Instrucciones para Copilot

Cuando trabajes en este proyecto:

### Límites de Respuesta

1. **Divide respuestas largas**

   - ❌ **NUNCA generar respuestas que superen los límites de tokens**
   - ✅ **SIEMPRE dividir contenido extenso en múltiples entregas**
   - ✅ Crear contenido por secciones, esperar confirmación del usuario
   - ✅ Priorizar calidad sobre cantidad en cada entrega

2. **Estrategia de División**
   - Para semanas completas: dividir por carpetas (teoria → practicas → proyecto)
   - Para archivos grandes: dividir por secciones lógicas
   - Siempre indicar claramente qué parte se entrega y qué falta

### Generación de Código

1. **Usa siempre Python moderno (3.13+)**

   - Type hints
   - f-strings
   - Pathlib
   - Match statements cuando aplique
   - Walrus operator cuando mejore legibilidad

2. **Gestión de Paquetes**

   - ✅ **SIEMPRE usar entornos virtuales** (venv, conda, o poetry)
   - ✅ **pip + venv** es perfectamente válido para aprendizaje
   - ✅ **conda** recomendado para Deep Learning (mejor manejo de CUDA/cuDNN)
   - ✅ **poetry** ideal para proyectos con dependencias complejas
   - ✅ Documentar dependencias en requirements.txt o environment.yml
   - Comandos recomendados:

     ```bash
     # Opción 1: venv + pip (simple, universal)
     python -m venv .venv
     source .venv/bin/activate  # Linux/Mac
     pip install -r requirements.txt

     # Opción 2: conda (recomendado para Deep Learning)
     conda create -n ai-bootcamp python=3.11
     conda activate ai-bootcamp
     pip install -r requirements.txt

     # Opción 3: poetry (gestión avanzada de dependencias)
     poetry install
     ```

3. **Docker y Docker Compose** (Entornos Controlados)

   - ✅ **Usar Docker** para garantizar entornos limpios, estables y reproducibles
   - ✅ **docker compose** para orquestar servicios (Jupyter, bases de datos, APIs)
   - ✅ **Pre-requisito**: Conocimiento básico de Docker
   - ✅ Incluir `Dockerfile` y `docker-compose.yml` en proyectos que lo requieran
   - Beneficios:

     - Elimina "funciona en mi máquina"
     - Versiones exactas de Python y dependencias
     - Fácil setup para nuevos estudiantes
     - Entorno idéntico en cualquier OS

   - Comandos recomendados:

     ```bash
     # Construir y levantar entorno
     docker compose up --build

     # Ejecutar en modo interactivo
     docker compose run --rm app python script.py

     # Acceder a Jupyter Lab
     docker compose up jupyter
     # Abrir http://localhost:8888

     # Limpiar entorno
     docker compose down -v
     ```

   - Estructura recomendada:

     ```
     proyecto/
     ├── Dockerfile
     ├── docker-compose.yml
     ├── requirements.txt
     ├── src/
     └── notebooks/
     ```

4. **Jupyter Notebooks**

   - ✅ Usar para exploración y visualización
   - ✅ Limpiar outputs antes de commit
   - ✅ Incluir markdown explicativo
   - ❌ NO para código de producción

5. **Comenta el código de manera educativa**

   - Explica conceptos para principiantes
   - Incluye referencias a documentación cuando sea útil
   - Usa comentarios que enseñen, no solo describan

6. **Proporciona ejemplos completos y funcionales**
   - Código que se pueda copiar y ejecutar
   - Incluye casos de uso realistas
   - Muestra tanto lo que se debe hacer como lo que se debe evitar

### Creación de Contenido

1. **Estructura clara y progresiva**

   - De lo simple a lo complejo
   - Conceptos construidos sobre conocimientos previos
   - Repetición espaciada de conceptos clave

2. **Ejemplos del mundo real**

   - Casos de uso prácticos y relevantes
   - Proyectos que los estudiantes puedan mostrar en portfolios
   - Problemas que encontrarán en el desarrollo real

3. **Enfoque moderno**
   - Usar las últimas versiones estables de librerías
   - Enfocarse en mejores prácticas actuales
   - Mencionar tendencias y estado del arte

### Respuestas y Ayuda

1. **Explicaciones claras**

   - Lenguaje simple y directo
   - Evitar jerga innecesaria
   - Proporcionar analogías cuando sea útil

2. **Código comentado**

   - Explicar cada paso importante
   - Destacar conceptos clave
   - Señalar posibles errores comunes

3. **Recursos adicionales**
   - Referencias a documentación oficial
   - Enlaces a papers relevantes
   - Tutoriales y cursos complementarios

---

## 📚 Referencias Oficiales

- **Python Docs**: https://docs.python.org/3/
- **NumPy**: https://numpy.org/doc/
- **Pandas**: https://pandas.pydata.org/docs/
- **Scikit-learn**: https://scikit-learn.org/stable/
- **TensorFlow**: https://www.tensorflow.org/
- **PyTorch**: https://pytorch.org/docs/
- **Hugging Face**: https://huggingface.co/docs
- **Kaggle**: https://www.kaggle.com/

---

## 🔗 Enlaces Importantes

- **Repositorio**: https://github.com/epti-dev/bc-aidev
- **Documentación general**: [\_docs/README.md](_docs/README.md)
- **Primera semana**: [bootcamp/week-01/README.md](bootcamp/week-01/README.md)

---

## ✅ Checklist para Nuevas Semanas

Cuando crees contenido para una nueva semana:

- [ ] Crear estructura de carpetas completa
- [ ] README.md con objetivos y estructura
- [ ] Material teórico en 1-teoria/
- [ ] Ejercicios prácticos en 2-practicas/
- [ ] Proyecto integrador en 3-proyecto/
- [ ] Recursos adicionales en 4-recursos/
- [ ] Glosario de términos en 5-glosario/
- [ ] Rúbrica de evaluación
- [ ] Verificar coherencia con semanas anteriores
- [ ] Revisar progresión de dificultad
- [ ] Probar código de ejemplos
- [ ] Verificar que notebooks ejecutan correctamente

---

## 💡 Notas Finales

- **Prioridad**: Claridad sobre brevedad
- **Enfoque**: Aprendizaje práctico sobre teoría abstracta
- **Objetivo**: Preparar desarrolladores IA/ML listos para trabajar
- **Filosofía**: De Zero a Hero, paso a paso

---

_Última actualización: Diciembre 2025_
_Versión: 1.0_
