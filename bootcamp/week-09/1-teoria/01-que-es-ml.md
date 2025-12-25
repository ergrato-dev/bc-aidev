# 🤖 ¿Qué es Machine Learning?

## 🎯 Objetivos

- Comprender la definición de Machine Learning
- Conocer la historia y evolución del ML
- Identificar aplicaciones reales de ML
- Entender la diferencia entre programación tradicional y ML

---

## 1. Definición de Machine Learning

### Definición Clásica (Arthur Samuel, 1959)

> "Machine Learning es el campo de estudio que da a las computadoras la habilidad de aprender sin ser explícitamente programadas."

### Definición Moderna (Tom Mitchell, 1997)

> "Un programa de computadora se dice que aprende de la experiencia E con respecto a alguna tarea T y alguna medida de rendimiento P, si su rendimiento en T, medido por P, mejora con la experiencia E."

### En Palabras Simples

Machine Learning es una rama de la Inteligencia Artificial que permite a los sistemas **aprender patrones** a partir de datos y **mejorar su rendimiento** con la experiencia, sin necesidad de programar reglas explícitas.

---

## 2. Programación Tradicional vs Machine Learning

![Programación Tradicional vs Machine Learning](../0-assets/01-programacion-vs-ml.svg)

### Programación Tradicional

```python
# Enfoque tradicional: reglas explícitas
def clasificar_email_tradicional(email):
    palabras_spam = ['gratis', 'ganador', 'premio', 'urgente']

    for palabra in palabras_spam:
        if palabra in email.lower():
            return 'spam'
    return 'no spam'
```

**Problema**: ¿Qué pasa cuando los spammers cambian las palabras?

### Machine Learning

```
Datos + Resultados → Algoritmo ML → Reglas (Modelo)
```

```python
# Enfoque ML: el modelo aprende las reglas
from sklearn.naive_bayes import MultinomialNB

# Entrenamiento: el modelo aprende de ejemplos
modelo = MultinomialNB()
modelo.fit(emails_entrenamiento, etiquetas)

# Predicción: aplica lo aprendido
prediccion = modelo.predict(nuevo_email)
```

**Ventaja**: El modelo se adapta a nuevos patrones automáticamente.

---

## 3. ¿Por qué Machine Learning Ahora?

### Tres Factores Clave

![Factores clave del auge de ML](../0-assets/02-factores-ml.svg)

1. **Datos**: Explosión de datos (redes sociales, sensores, transacciones)
2. **Poder de Cómputo**: GPUs, cloud computing, hardware especializado
3. **Algoritmos**: Avances en deep learning, frameworks accesibles

---

## 4. Aplicaciones de Machine Learning

### Visión por Computadora

- Reconocimiento facial
- Vehículos autónomos
- Diagnóstico médico por imágenes
- Control de calidad en manufactura

### Procesamiento de Lenguaje Natural (NLP)

- Asistentes virtuales (Siri, Alexa)
- Traducción automática
- Análisis de sentimientos
- Chatbots y LLMs (ChatGPT)

### Sistemas de Recomendación

- Netflix (películas)
- Spotify (música)
- Amazon (productos)
- YouTube (videos)

### Finanzas

- Detección de fraude
- Trading algorítmico
- Scoring crediticio
- Predicción de riesgos

### Salud

- Diagnóstico de enfermedades
- Descubrimiento de fármacos
- Predicción de epidemias
- Medicina personalizada

### Otros

- Filtros de spam
- Reconocimiento de voz
- Juegos (AlphaGo)
- Predicción del clima

---

## 5. Componentes de un Sistema ML

```python
# Componentes esenciales
datos = cargar_datos()              # 1. DATOS
X = extraer_features(datos)          # 2. FEATURES (características)
y = obtener_labels(datos)            # 3. LABELS (etiquetas)
modelo = SeleccionarAlgoritmo()      # 4. ALGORITMO
modelo.fit(X, y)                     # 5. ENTRENAMIENTO
predicciones = modelo.predict(X_nuevo)  # 6. PREDICCIÓN
evaluar(predicciones, y_real)        # 7. EVALUACIÓN
```

### Vocabulario Esencial

| Término          | Definición                              | Ejemplo                      |
| ---------------- | --------------------------------------- | ---------------------------- |
| **Dataset**      | Conjunto de datos para entrenar/evaluar | Tabla con 1000 emails        |
| **Features (X)** | Características/atributos de entrada    | Palabras del email, longitud |
| **Labels (y)**   | Etiquetas/resultados a predecir         | spam / no spam               |
| **Training**     | Proceso de aprendizaje del modelo       | Ajustar parámetros           |
| **Prediction**   | Aplicar modelo a datos nuevos           | Clasificar email nuevo       |
| **Model**        | Representación matemática aprendida     | Fórmula/reglas encontradas   |

---

## 6. El Proceso de Machine Learning

![Pipeline de Machine Learning](../0-assets/03-pipeline-ml.svg)

---

## 7. Ejemplo: Predicción de Precios de Casas

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Datos
datos = pd.DataFrame({
    'metros_cuadrados': [50, 80, 120, 150, 200],
    'habitaciones': [1, 2, 3, 3, 4],
    'precio': [100000, 180000, 250000, 300000, 400000]
})

# 2. Features y Labels
X = datos[['metros_cuadrados', 'habitaciones']]  # Features
y = datos['precio']                               # Label

# 3. División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Modelo
modelo = LinearRegression()

# 5. Entrenamiento
modelo.fit(X_train, y_train)

# 6. Predicción
casa_nueva = [[100, 2]]  # 100m², 2 habitaciones
precio_predicho = modelo.predict(casa_nueva)
print(f'Precio predicho: ${precio_predicho[0]:,.0f}')

# 7. Evaluación
score = modelo.score(X_test, y_test)
print(f'R² Score: {score:.2f}')
```

---

## 8. ¿Cuándo Usar Machine Learning?

### ✅ Usar ML cuando:

- Hay patrones complejos difíciles de programar manualmente
- Existe suficiente cantidad de datos históricos
- El problema requiere adaptación a nuevos datos
- Las reglas son difíciles de expresar explícitamente

### ❌ NO usar ML cuando:

- El problema tiene reglas claras y simples
- No hay suficientes datos
- Se necesita explicabilidad total de las decisiones
- Una solución determinística funciona bien

---

## ✅ Resumen

| Concepto       | Descripción                             |
| -------------- | --------------------------------------- |
| ML             | Sistemas que aprenden de datos          |
| vs Tradicional | Aprende reglas en lugar de programarlas |
| Features       | Variables de entrada (X)                |
| Labels         | Variable a predecir (y)                 |
| Training       | Proceso de aprendizaje                  |
| Prediction     | Aplicar lo aprendido                    |

---

## 🔗 Navegación

| Anterior                 | Siguiente                                         |
| ------------------------ | ------------------------------------------------- |
| [← Índice](../README.md) | [Tipos de Aprendizaje →](02-tipos-aprendizaje.md) |
