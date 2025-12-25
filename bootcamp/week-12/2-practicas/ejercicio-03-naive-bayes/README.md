# Ejercicio 03: Naive Bayes para Clasificación de Texto

## 🎯 Objetivo

Implementar Naive Bayes para clasificación de texto usando 20 Newsgroups.

## 📋 Instrucciones

### Paso 1: Cargar Dataset de Texto

Usamos 20 Newsgroups (subconjunto de 4 categorías).

```python
from sklearn.datasets import fetch_20newsgroups

categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics', 'talk.politics.guns']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

print(f"Documentos: {len(newsgroups.data)}")
print(f"Categorías: {newsgroups.target_names}")
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

### Paso 2: Explorar Datos

Veamos ejemplos de documentos.

```python
for i in range(2):
    print(f"--- Documento {i+1} ---")
    print(f"Categoría: {newsgroups.target_names[newsgroups.target[i]]}")
    print(f"Texto: {newsgroups.data[i][:200]}...")
```

**Descomenta** la sección del Paso 2.

### Paso 3: Vectorizar Texto

Convertimos texto a vectores TF-IDF.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"Vocabulario: {len(tfidf.vocabulary_)} palabras")
print(f"Train shape: {X_train_tfidf.shape}")
```

**Descomenta** la sección del Paso 3.

### Paso 4: MultinomialNB

Entrenamos Naive Bayes para texto.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_tfidf, y_train)
y_pred = mnb.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

**Descomenta** la sección del Paso 4.

### Paso 5: Tuning de Alpha

Probamos diferentes valores de suavizado.

```python
from sklearn.model_selection import cross_val_score

alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]

for alpha in alphas:
    mnb = MultinomialNB(alpha=alpha)
    scores = cross_val_score(mnb, X_train_tfidf, y_train, cv=5)
    print(f"alpha={alpha:5.3f}: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Descomenta** la sección del Paso 5.

### Paso 6: Pipeline Completo y Evaluación

```python
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', MultinomialNB(alpha=0.1))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"Test Accuracy: {pipeline.score(X_test, y_test):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))
```

**Descomenta** la sección del Paso 6.

## ✅ Resultado Esperado

- Accuracy ≥ 0.85 en clasificación de texto
- Identificación del alpha óptimo
- Classification report por categoría

## 🔗 Recursos

- [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
