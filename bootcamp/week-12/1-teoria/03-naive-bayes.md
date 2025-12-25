# Naive Bayes

## 🎯 Objetivos

- Comprender el Teorema de Bayes
- Entender la asunción "naive" de independencia
- Conocer los diferentes tipos de Naive Bayes
- Aplicar Naive Bayes para clasificación de texto

## 📋 Contenido

### 1. Teorema de Bayes

Base matemática del algoritmo:

$$P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}$$

![Naive Bayes](../0-assets/04-naive-bayes.svg)

| Término   | Nombre     | Significado                          |
| --------- | ---------- | ------------------------------------ |
| $P(y\|X)$ | Posterior  | Probabilidad de clase dado los datos |
| $P(X\|y)$ | Likelihood | Probabilidad de datos dada la clase  |
| $P(y)$    | Prior      | Probabilidad a priori de la clase    |
| $P(X)$    | Evidence   | Probabilidad total de los datos      |

### 2. La Asunción "Naive"

Naive Bayes asume que las features son **condicionalmente independientes** dada la clase:

$$P(X|y) = P(x_1|y) \cdot P(x_2|y) \cdot ... \cdot P(x_n|y) = \prod_{i=1}^{n} P(x_i|y)$$

**¿Por qué "naive"?**

- Esta asunción raramente es cierta en la realidad
- Pero simplifica enormemente el cálculo
- Y funciona sorprendentemente bien en la práctica

### 3. Predicción

Para clasificar, elegimos la clase con mayor probabilidad posterior:

$$\hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i|y)$$

Como $P(X)$ es constante para todas las clases, se puede ignorar.

### 4. Tipos de Naive Bayes

#### GaussianNB (Features Continuas)

Asume que las features siguen una distribución normal.

$$P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$$

```python
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
```

**Uso**: Features numéricas continuas (Iris, mediciones).

#### MultinomialNB (Conteos)

Para features que representan conteos o frecuencias.

```python
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=1.0)  # alpha: suavizado Laplace
mnb.fit(X_train_counts, y_train)
y_pred = mnb.predict(X_test_counts)
```

**Uso**: Clasificación de texto con TF-IDF o bag-of-words.

#### BernoulliNB (Binario)

Para features binarias (presencia/ausencia).

```python
from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train_binary, y_train)
y_pred = bnb.predict(X_test_binary)
```

**Uso**: Texto binarizado, flags de presencia.

#### ComplementNB (Clases Desbalanceadas)

Diseñado específicamente para clases desbalanceadas.

```python
from sklearn.naive_bayes import ComplementNB

cnb = ComplementNB(alpha=1.0)
cnb.fit(X_train, y_train)
y_pred = cnb.predict(X_test)
```

### 5. Suavizado Laplace (alpha)

Evita probabilidades cero cuando una feature no aparece en el entrenamiento.

$$P(x_i|y) = \frac{count(x_i, y) + \alpha}{count(y) + \alpha \cdot n}$$

```python
# Sin suavizado (riesgo de prob=0)
mnb_no_smooth = MultinomialNB(alpha=0)

# Suavizado Laplace (default)
mnb_laplace = MultinomialNB(alpha=1.0)

# Suavizado menor
mnb_light = MultinomialNB(alpha=0.1)
```

### 6. Naive Bayes para Clasificación de Texto

El caso de uso más exitoso de Naive Bayes.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Pipeline completo para texto
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', MultinomialNB(alpha=0.1))
])

text_clf.fit(X_train_text, y_train)
y_pred = text_clf.predict(X_test_text)
```

#### Ejemplo: Clasificación de Spam

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Cargar datos de ejemplo
categories = ['rec.sport.hockey', 'sci.med', 'soc.religion.christian', 'talk.politics.guns']
newsgroups = fetch_20newsgroups(subset='all', categories=categories)

X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target, test_size=0.2, random_state=42
)

# Vectorizar
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Entrenar
mnb = MultinomialNB(alpha=0.1)
mnb.fit(X_train_tfidf, y_train)

# Evaluar
y_pred = mnb.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))
```

### 7. Probabilidades

Naive Bayes da probabilidades naturalmente:

```python
# Probabilidades de cada clase
proba = mnb.predict_proba(X_test_tfidf)

# Log-probabilidades (más estables numéricamente)
log_proba = mnb.predict_log_proba(X_test_tfidf)
```

### 8. Ventajas y Desventajas

#### ✅ Ventajas

- **Muy rápido** de entrenar y predecir
- Funciona bien con **pocas muestras**
- Excelente para **clasificación de texto**
- Da **probabilidades** calibradas
- Escalable a datasets grandes
- Simple y fácil de implementar

#### ❌ Desventajas

- Asunción de independencia raramente cierta
- Mal rendimiento si features están correlacionadas
- No captura interacciones entre features
- Sensible a features irrelevantes (GaussianNB)

### 9. Comparación de Variantes

| Variante          | Features  | Mejor Para            |
| ----------------- | --------- | --------------------- |
| **GaussianNB**    | Continuas | Datos numéricos, Iris |
| **MultinomialNB** | Conteos   | Texto, TF-IDF, BoW    |
| **BernoulliNB**   | Binarias  | Texto binarizado      |
| **ComplementNB**  | Conteos   | Texto desbalanceado   |

### 10. Ejemplo Completo

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Cargar datos
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenar GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(gnb, X, y, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Evaluar
y_pred = gnb.predict(X_test)
print(f"\nTest Accuracy: {gnb.score(X_test, y_test):.4f}")

# Probabilidades
proba = gnb.predict_proba(X_test[:3])
print("\nProbabilidades (primeras 3 muestras):")
for i, p in enumerate(proba):
    print(f"  Muestra {i+1}: {dict(zip(iris.target_names, p.round(3)))}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## ✅ Checklist de Verificación

- [ ] Comprendo el Teorema de Bayes
- [ ] Entiendo la asunción de independencia condicional
- [ ] Conozco los diferentes tipos de Naive Bayes
- [ ] Sé cuándo usar cada variante
- [ ] Puedo aplicar Naive Bayes para texto con sklearn

---

## 📚 Recursos

- [Naive Bayes - sklearn](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [Text Classification Tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
