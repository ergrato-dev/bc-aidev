# 🧹 Ejercicio 01: Preprocesamiento de Texto

## 🎯 Objetivo

Implementar un pipeline completo de preprocesamiento de texto para NLP.

---

## 📋 Descripción

En este ejercicio aprenderás a limpiar y normalizar texto paso a paso, aplicando técnicas fundamentales de preprocesamiento que son esenciales para cualquier proyecto de NLP.

---

## 🔧 Pasos del Ejercicio

### Paso 1: Conversión a Minúsculas

La normalización de case es el primer paso para reducir la variabilidad del texto:

```python
text = "HOLA Mundo"
text_lower = text.lower()
# "hola mundo"
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

### Paso 2: Eliminar Puntuación

Usamos expresiones regulares para eliminar caracteres no deseados:

```python
import re
text = "¡Hola, mundo!"
text_clean = re.sub(r'[^\w\s]', '', text)
# "Hola mundo"
```

### Paso 3: Eliminar Números

En muchos casos, los números no aportan información semántica:

```python
text = "Tengo 3 gatos y 2 perros"
text_no_nums = re.sub(r'\d+', '', text)
# "Tengo  gatos y  perros"
```

### Paso 4: Eliminar Espacios Extra

Después de las limpiezas, pueden quedar espacios múltiples:

```python
text = "Hola   mundo  cruel"
text_clean = re.sub(r'\s+', ' ', text).strip()
# "Hola mundo cruel"
```

### Paso 5: Eliminar Acentos

Normalizar caracteres acentuados (opcional según el caso):

```python
import unicodedata

def remove_accents(text):
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

text = "niño está aquí"
text_no_accents = remove_accents(text)
# "nino esta aqui"
```

### Paso 6: Pipeline Completo

Combina todos los pasos en una función reutilizable:

```python
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

---

## 📁 Estructura

```
ejercicio-01-preprocesamiento/
├── README.md
└── starter/
    └── main.py
```

---

## ▶️ Ejecución

```bash
cd bootcamp/04-especializacion/week-29/2-practicas/ejercicio-01-preprocesamiento
python starter/main.py
```

---

## ✅ Criterios de Éxito

- [ ] El pipeline convierte texto a minúsculas
- [ ] Elimina correctamente la puntuación
- [ ] Elimina números del texto
- [ ] Normaliza espacios múltiples
- [ ] La función `preprocess()` combina todos los pasos

---

## 🔗 Recursos

- [Documentación de `re` (regex)](https://docs.python.org/3/library/re.html)
- [unicodedata](https://docs.python.org/3/library/unicodedata.html)
