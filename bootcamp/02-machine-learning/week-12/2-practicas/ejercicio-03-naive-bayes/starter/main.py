"""
Ejercicio 03: Naive Bayes para Clasificación de Texto
=====================================================
Implementa MultinomialNB para clasificar documentos.
"""

# ============================================
# PASO 1: Cargar Dataset de Texto
# ============================================
print('--- Paso 1: Cargar Dataset de Texto ---')

# Descomenta las siguientes líneas:
# from sklearn.datasets import fetch_20newsgroups
#
# categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics', 'talk.politics.guns']
#
# print("Descargando dataset (puede tomar unos segundos)...")
# newsgroups = fetch_20newsgroups(
#     subset='all',
#     categories=categories,
#     remove=('headers', 'footers', 'quotes')
# )
#
# print(f"Total documentos: {len(newsgroups.data)}")
# print(f"Categorías: {newsgroups.target_names}")

print()

# ============================================
# PASO 2: Explorar Datos
# ============================================
print('--- Paso 2: Explorar Datos ---')

# Descomenta las siguientes líneas:
# print("Ejemplos de documentos:\n")
# for i in range(2):
#     print(f"--- Documento {i+1} ---")
#     print(f"Categoría: {newsgroups.target_names[newsgroups.target[i]]}")
#     text_preview = newsgroups.data[i][:300].replace('\n', ' ')
#     print(f"Texto: {text_preview}...")
#     print()

print()

# ============================================
# PASO 3: Vectorizar Texto
# ============================================
print('--- Paso 3: Vectorizar Texto ---')

# Descomenta las siguientes líneas:
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(
#     newsgroups.data, newsgroups.target,
#     test_size=0.2,
#     random_state=42
# )
#
# tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
# X_train_tfidf = tfidf.fit_transform(X_train)
# X_test_tfidf = tfidf.transform(X_test)
#
# print(f"Vocabulario: {len(tfidf.vocabulary_)} palabras únicas")
# print(f"Train shape: {X_train_tfidf.shape}")
# print(f"Test shape: {X_test_tfidf.shape}")
# print(f"Sparsity: {100 * (1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])):.2f}%")

print()

# ============================================
# PASO 4: MultinomialNB Básico
# ============================================
print('--- Paso 4: MultinomialNB Básico ---')

# Descomenta las siguientes líneas:
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score
#
# mnb = MultinomialNB(alpha=1.0)  # Suavizado Laplace
# mnb.fit(X_train_tfidf, y_train)
# y_pred = mnb.predict(X_test_tfidf)
#
# print(f"MultinomialNB (alpha=1.0)")
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print()

# ============================================
# PASO 5: Tuning de Alpha
# ============================================
print('--- Paso 5: Tuning de Alpha ---')

# Descomenta las siguientes líneas:
# from sklearn.model_selection import cross_val_score
#
# alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
# best_alpha = None
# best_score = 0
#
# print("Probando diferentes valores de alpha:")
# for alpha in alphas:
#     mnb = MultinomialNB(alpha=alpha)
#     scores = cross_val_score(mnb, X_train_tfidf, y_train, cv=5)
#     mean_score = scores.mean()
#     print(f"  alpha={alpha:5.3f}: {mean_score:.4f} ± {scores.std():.4f}")
#     if mean_score > best_score:
#         best_score = mean_score
#         best_alpha = alpha
#
# print(f"\nMejor alpha: {best_alpha}")

print()

# ============================================
# PASO 6: Pipeline Completo y Evaluación
# ============================================
print('--- Paso 6: Pipeline Completo y Evaluación ---')

# Descomenta las siguientes líneas:
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report
#
# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
#     ('clf', MultinomialNB(alpha=best_alpha))
# ])
#
# # Entrenar con texto original
# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
#
# print(f"Test Accuracy: {pipeline.score(X_test, y_test):.4f}")
# print(f"\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))
#
# # Ejemplo de predicción
# print("\n--- Predicción de ejemplo ---")
# sample_texts = [
#     "NASA launched a new satellite to explore Mars",
#     "The Yankees won the World Series championship"
# ]
# predictions = pipeline.predict(sample_texts)
# for text, pred in zip(sample_texts, predictions):
#     print(f"'{text[:50]}...' -> {newsgroups.target_names[pred]}")

print()
print('=' * 50)
print('Ejercicio completado!')
