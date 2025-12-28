"""
Ejercicio 01: Preprocesamiento de Texto
======================================

Aprende a limpiar y normalizar texto para NLP.

Instrucciones:
1. Lee cada sección y descomenta el código
2. Ejecuta el script después de cada paso
3. Observa cómo cambia el texto en cada etapa
"""

import re
import unicodedata

# ============================================
# PASO 1: Conversión a Minúsculas
# ============================================
print("--- Paso 1: Conversión a Minúsculas ---")

# La normalización de case reduce la variabilidad del vocabulario
# "Hola" y "hola" se tratan como la misma palabra

# Descomenta las siguientes líneas:
# text = "HOLA Mundo, ¿Cómo ESTÁS?"
# text_lower = text.lower()
# print(f'Original: {text}')
# print(f'Minúsculas: {text_lower}')

print()


# ============================================
# PASO 2: Eliminar Puntuación
# ============================================
print("--- Paso 2: Eliminar Puntuación ---")

# Usamos regex para eliminar caracteres que no son palabras ni espacios
# \w = caracteres de palabra (letras, números, guión bajo)
# \s = espacios en blanco

# Descomenta las siguientes líneas:
# text = "¡Hola, mundo! ¿Cómo estás? #NLP @python"
# text_no_punct = re.sub(r'[^\w\s]', '', text)
# print(f'Original: {text}')
# print(f'Sin puntuación: {text_no_punct}')

print()


# ============================================
# PASO 3: Eliminar Números
# ============================================
print("--- Paso 3: Eliminar Números ---")

# En muchos casos de NLP, los números no aportan información semántica
# \d+ = uno o más dígitos

# Descomenta las siguientes líneas:
# text = "Tengo 3 gatos, 2 perros y 100 peces"
# text_no_nums = re.sub(r'\d+', '', text)
# print(f'Original: {text}')
# print(f'Sin números: {text_no_nums}')

print()


# ============================================
# PASO 4: Eliminar Espacios Extra
# ============================================
print("--- Paso 4: Eliminar Espacios Extra ---")

# Después de eliminar caracteres, pueden quedar espacios múltiples
# \s+ = uno o más espacios en blanco

# Descomenta las siguientes líneas:
# text = "Hola   mundo    cruel   "
# text_clean = re.sub(r'\s+', ' ', text).strip()
# print(f'Original: "{text}"')
# print(f'Limpio: "{text_clean}"')

print()


# ============================================
# PASO 5: Eliminar Acentos (Opcional)
# ============================================
print("--- Paso 5: Eliminar Acentos ---")

# Normalizar acentos puede ser útil para algunos casos
# NFKD descompone caracteres en base + modificadores


def remove_accents(text: str) -> str:
    """Elimina acentos y diacríticos del texto."""
    # Descomenta las siguientes líneas:
    # nfkd = unicodedata.normalize('NFKD', text)
    # return ''.join(c for c in nfkd if not unicodedata.combining(c))
    pass  # Elimina esta línea cuando descomentes


# Descomenta las siguientes líneas:
# text = "El niño está aquí con su mamá"
# text_no_accents = remove_accents(text)
# print(f'Original: {text}')
# print(f'Sin acentos: {text_no_accents}')

print()


# ============================================
# PASO 6: Pipeline Completo
# ============================================
print("--- Paso 6: Pipeline Completo ---")


def preprocess(
    text: str, remove_nums: bool = True, normalize_accents: bool = False
) -> str:
    """
    Pipeline completo de preprocesamiento.

    Args:
        text: Texto a procesar
        remove_nums: Si eliminar números
        normalize_accents: Si eliminar acentos

    Returns:
        Texto preprocesado
    """
    # Descomenta las siguientes líneas:
    # # 1. Minúsculas
    # text = text.lower()
    #
    # # 2. Eliminar puntuación
    # text = re.sub(r'[^\w\s]', '', text)
    #
    # # 3. Eliminar números (opcional)
    # if remove_nums:
    #     text = re.sub(r'\d+', '', text)
    #
    # # 4. Eliminar acentos (opcional)
    # if normalize_accents:
    #     text = remove_accents(text)
    #
    # # 5. Normalizar espacios
    # text = re.sub(r'\s+', ' ', text).strip()
    #
    # return text
    pass  # Elimina esta línea cuando descomentes


# Descomenta las siguientes líneas para probar:
# textos_prueba = [
#     "¡Hola Mundo! ¿Cómo estás?",
#     "Python 3.11 es GENIAL para NLP!!!",
#     "El niño tiene 5 años y está muy feliz 😊",
#     "   Espacios   múltiples   aquí   ",
# ]
#
# print('Prueba del pipeline:')
# print('-' * 50)
# for texto in textos_prueba:
#     resultado = preprocess(texto)
#     print(f'Input:  "{texto}"')
#     print(f'Output: "{resultado}"')
#     print()

print()


# ============================================
# PASO 7: Procesar Múltiples Documentos
# ============================================
print("--- Paso 7: Procesar Corpus ---")


def preprocess_corpus(documents: list) -> list:
    """Preprocesa una lista de documentos."""
    # Descomenta la siguiente línea:
    # return [preprocess(doc) for doc in documents]
    pass  # Elimina esta línea cuando descomentes


# Descomenta las siguientes líneas:
# corpus = [
#     "El Machine Learning es FASCINANTE!",
#     "NLP procesa texto en 2024.",
#     "¿Quieres aprender Python?",
# ]
#
# corpus_limpio = preprocess_corpus(corpus)
# print('Corpus original:')
# for doc in corpus:
#     print(f'  - {doc}')
#
# print('\nCorpus preprocesado:')
# for doc in corpus_limpio:
#     print(f'  - {doc}')

print()
print("=" * 50)
print("¡Ejercicio completado!")
print("Ahora sabes preprocesar texto para NLP.")
