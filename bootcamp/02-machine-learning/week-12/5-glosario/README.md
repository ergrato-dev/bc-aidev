# Glosario Semana 12: SVM, KNN y Naive Bayes

## A

### Accuracy

Proporción de predicciones correctas sobre el total. `(TP + TN) / (TP + TN + FP + FN)`.

### Alpha (α)

En Naive Bayes, parámetro de suavizado Laplace que evita probabilidades cero.

## B

### Bayes, Teorema de

Fórmula para calcular probabilidades condicionales: `P(A|B) = P(B|A) × P(A) / P(B)`.

### Bias-Variance Tradeoff

Balance entre error por simplificación (bias) y error por sensibilidad a datos (variance).

### BernoulliNB

Variante de Naive Bayes para features binarias (presencia/ausencia).

## C

### C (parámetro SVM)

Parámetro de regularización. C alto = menos errores permitidos, C bajo = margen más amplio.

### Conditional Independence

Asunción de Naive Bayes: features son independientes dado la clase.

### Cross-Validation

Técnica de evaluación que divide datos en k partes para entrenar y validar múltiples veces.

## D

### Decision Boundary

Frontera que separa clases en el espacio de features.

### Distance Metric

Función que mide similitud entre puntos. Ejemplos: Euclidiana, Manhattan.

## E

### Euclidean Distance

Distancia en línea recta: `√Σ(xi - yi)²`.

### Evidence

En Bayes, probabilidad total de los datos observados P(X).

## F

### F1-Score

Media armónica de precision y recall: `2 × (P × R) / (P + R)`.

### Feature Space

Espacio multidimensional donde cada dimensión es una feature.

## G

### Gamma (γ)

En SVM con kernel RBF, controla el alcance de influencia de cada punto.

### GaussianNB

Naive Bayes para features continuas que asume distribución normal.

### GridSearchCV

Búsqueda exhaustiva de hiperparámetros con validación cruzada.

## H

### Hard Margin

SVM que no permite errores de clasificación. Solo funciona con datos linealmente separables.

### Hiperparámetro

Parámetro que se define antes del entrenamiento (k en KNN, C en SVM).

### Hyperplane

Superficie de decisión en SVM que separa las clases.

## I

### Instance-based Learning

Aprendizaje que guarda instancias de entrenamiento (como KNN). También llamado lazy learning.

## K

### K (en KNN)

Número de vecinos a considerar para la predicción.

### Kernel

Función que transforma datos a un espacio de mayor dimensión en SVM.

### Kernel Trick

Técnica que permite calcular productos en espacio transformado sin transformar explícitamente.

### KNN (K-Nearest Neighbors)

Algoritmo que clasifica basándose en los k vecinos más cercanos.

## L

### Laplace Smoothing

Técnica para evitar probabilidades cero añadiendo α a los conteos.

### Lazy Learning

Algoritmos que no construyen modelo explícito, solo guardan datos (KNN).

### Likelihood

Probabilidad de observar los datos dada una clase: P(X|y).

### Linear Kernel

Kernel SVM para datos linealmente separables: `K(x,y) = x·y`.

## M

### Manhattan Distance

Suma de diferencias absolutas: `Σ|xi - yi|`.

### Margin

En SVM, distancia entre el hiperplano y los puntos más cercanos.

### Maximum Margin Classifier

SVM busca el hiperplano que maximiza el margen.

### Minkowski Distance

Generalización de distancias: `(Σ|xi - yi|^p)^(1/p)`.

### MultinomialNB

Naive Bayes para conteos/frecuencias, común en clasificación de texto.

## N

### Naive Bayes

Clasificador probabilístico basado en Teorema de Bayes con asunción de independencia.

### n_neighbors

Parámetro de KNN que especifica el número de vecinos (k).

### Normalization

Escalar features a un rango común. Esencial para KNN y SVM.

## O

### Overfitting

Modelo que memoriza datos de entrenamiento pero no generaliza.

## P

### Pipeline

En sklearn, secuencia de transformaciones seguida de un estimador.

### Polynomial Kernel

Kernel SVM: `K(x,y) = (γx·y + r)^d`.

### Posterior

Probabilidad de clase después de observar datos: P(y|X).

### Precision

Proporción de predicciones positivas correctas: `TP / (TP + FP)`.

### Prior

Probabilidad a priori de una clase: P(y).

## R

### RBF Kernel (Radial Basis Function)

Kernel gaussiano: `K(x,y) = exp(-γ||x-y||²)`. El más usado en SVM.

### Recall

Proporción de positivos reales detectados: `TP / (TP + FN)`.

### Regularization

Técnicas para prevenir overfitting (parámetro C en SVM).

## S

### Soft Margin

SVM que permite algunos errores de clasificación.

### StandardScaler

Normaliza features a media 0 y varianza 1. Esencial antes de KNN/SVM.

### Support Vectors

Puntos de entrenamiento más cercanos al hiperplano que definen el margen.

### SVC

Support Vector Classification en sklearn.

### SVR

Support Vector Regression en sklearn.

## T

### TF-IDF

Term Frequency-Inverse Document Frequency. Vectorización de texto.

## U

### Underfitting

Modelo demasiado simple que no captura patrones.

## V

### Vectorizer

En sklearn, transforma texto a vectores numéricos (CountVectorizer, TfidfVectorizer).

### Voting (en KNN)

Mecanismo de decisión por mayoría entre los k vecinos.

## W

### Weights (en KNN)

'uniform' = todos los vecinos pesan igual, 'distance' = más cercanos pesan más.

---

## 📐 Fórmulas Clave

### Distancias

- **Euclidiana**: $d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$
- **Manhattan**: $d(x,y) = \sum_{i=1}^{n}|x_i - y_i|$
- **Minkowski**: $d(x,y) = \left(\sum_{i=1}^{n}|x_i - y_i|^p\right)^{1/p}$

### Teorema de Bayes

$$P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}$$

### Margen SVM

$$\text{margen} = \frac{2}{||w||}$$

### Kernels

- **Linear**: $K(x,y) = x \cdot y$
- **RBF**: $K(x,y) = e^{-\gamma||x-y||^2}$
- **Polynomial**: $K(x,y) = (\gamma x \cdot y + r)^d$
