"""
Ejercicio 02: Tokenización
==========================

Aprende diferentes estrategias de tokenización y construcción de vocabulario.

Instrucciones:
1. Lee cada sección y descomenta el código
2. Ejecuta el script después de cada paso
3. Compara los resultados de diferentes tokenizadores
"""

import re
from collections import Counter
from typing import Dict, List

# ============================================
# PASO 1: Tokenización Simple (split)
# ============================================
print("--- Paso 1: Tokenización Simple ---")

# La forma más básica: dividir por espacios
# Simple pero no maneja puntuación

# Descomenta las siguientes líneas:
# text = "Hola mundo cruel"
# tokens = text.split()
# print(f'Texto: "{text}"')
# print(f'Tokens: {tokens}')
# print(f'Número de tokens: {len(tokens)}')

print()


# ============================================
# PASO 2: Tokenización con Regex
# ============================================
print("--- Paso 2: Tokenización con Regex ---")

# Regex permite más control sobre qué se considera token
# \b\w+\b = palabras completas (sin puntuación)


def tokenize_regex(text: str) -> List[str]:
    """Tokeniza usando expresiones regulares."""
    # Descomenta la siguiente línea:
    # return re.findall(r'\b\w+\b', text.lower())
    pass  # Elimina esta línea cuando descomentes


# Descomenta las siguientes líneas:
# text = "¡Hola, mundo! ¿Cómo estás?"
# tokens = tokenize_regex(text)
# print(f'Texto: "{text}"')
# print(f'Tokens: {tokens}')

print()


# ============================================
# PASO 3: Tokenización con NLTK
# ============================================
print("--- Paso 3: Tokenización con NLTK ---")

# NLTK ofrece tokenizadores más sofisticados
# Nota: Requiere descargar datos: nltk.download('punkt')

# Descomenta las siguientes líneas:
# import nltk
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt', quiet=True)
#     nltk.download('punkt_tab', quiet=True)
#
# from nltk.tokenize import word_tokenize
#
# text = "¡Hola, mundo! ¿Cómo estás?"
# tokens = word_tokenize(text)
# print(f'Texto: "{text}"')
# print(f'Tokens NLTK: {tokens}')
# print(f'Nota: NLTK separa la puntuación como tokens')

print()


# ============================================
# PASO 4: Tokenización por Oraciones
# ============================================
print("--- Paso 4: Tokenización por Oraciones ---")

# Útil para dividir documentos en oraciones

# Descomenta las siguientes líneas:
# from nltk.tokenize import sent_tokenize
#
# text = """Python es genial. ¿No crees?
# Machine Learning es el futuro. ¡Aprende NLP!"""
#
# sentences = sent_tokenize(text)
# print(f'Texto:\n"{text}"')
# print(f'\nOraciones encontradas: {len(sentences)}')
# for i, sent in enumerate(sentences, 1):
#     print(f'  {i}. {sent}')

print()


# ============================================
# PASO 5: Construir Vocabulario
# ============================================
print("--- Paso 5: Construir Vocabulario ---")


def build_vocab(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
    """
    Construye vocabulario desde una lista de textos.

    Args:
        texts: Lista de documentos
        min_freq: Frecuencia mínima para incluir token

    Returns:
        Diccionario token -> índice
    """
    # Descomenta las siguientes líneas:
    # # Recolectar todos los tokens
    # all_tokens = []
    # for text in texts:
    #     tokens = re.findall(r'\b\w+\b', text.lower())
    #     all_tokens.extend(tokens)
    #
    # # Contar frecuencias
    # counts = Counter(all_tokens)
    #
    # # Tokens especiales
    # vocab = {
    #     '<PAD>': 0,   # Padding
    #     '<UNK>': 1,   # Unknown (fuera de vocabulario)
    #     '<BOS>': 2,   # Beginning of sequence
    #     '<EOS>': 3,   # End of sequence
    # }
    #
    # # Añadir tokens frecuentes
    # for token, count in counts.most_common():
    #     if count >= min_freq:
    #         vocab[token] = len(vocab)
    #
    # return vocab
    pass  # Elimina esta línea cuando descomentes


# Descomenta las siguientes líneas:
# corpus = [
#     "el gato come pescado",
#     "el perro come carne",
#     "el gato bebe leche",
#     "el perro bebe agua",
# ]
#
# vocab = build_vocab(corpus)
# print(f'Corpus: {corpus}')
# print(f'\nVocabulario ({len(vocab)} tokens):')
# for token, idx in vocab.items():
#     print(f'  {token}: {idx}')

print()


# ============================================
# PASO 6: Encode y Decode
# ============================================
print("--- Paso 6: Encode y Decode ---")


def encode(text: str, vocab: Dict[str, int]) -> List[int]:
    """Convierte texto a lista de índices."""
    # Descomenta las siguientes líneas:
    # tokens = re.findall(r'\b\w+\b', text.lower())
    # unk_id = vocab.get('<UNK>', 1)
    # return [vocab.get(token, unk_id) for token in tokens]
    pass  # Elimina esta línea cuando descomentes


def decode(ids: List[int], vocab: Dict[str, int]) -> List[str]:
    """Convierte lista de índices a tokens."""
    # Descomenta las siguientes líneas:
    # id_to_token = {v: k for k, v in vocab.items()}
    # return [id_to_token.get(i, '<UNK>') for i in ids]
    pass  # Elimina esta línea cuando descomentes


# Descomenta las siguientes líneas:
# # Usar el vocabulario del paso anterior
# if vocab:
#     text = "el gato come pescado"
#     encoded = encode(text, vocab)
#     decoded = decode(encoded, vocab)
#
#     print(f'Texto original: "{text}"')
#     print(f'Encoded (IDs): {encoded}')
#     print(f'Decoded: {decoded}')
#
#     # Probar con palabra desconocida (OOV)
#     text_oov = "el tigre come pescado"
#     encoded_oov = encode(text_oov, vocab)
#     decoded_oov = decode(encoded_oov, vocab)
#
#     print(f'\nTexto con OOV: "{text_oov}"')
#     print(f'Encoded: {encoded_oov}')
#     print(f'Decoded: {decoded_oov}')
#     print(f'Nota: "tigre" se mapea a <UNK> (índice 1)')

print()


# ============================================
# PASO 7: Clase Tokenizer Completa
# ============================================
print("--- Paso 7: Clase Tokenizer ---")


class SimpleTokenizer:
    """Tokenizador simple con vocabulario."""

    def __init__(self):
        self.vocab = {}
        self.id_to_token = {}

    def fit(self, texts: List[str], min_freq: int = 1):
        """Construye vocabulario desde textos."""
        # Descomenta las siguientes líneas:
        # self.vocab = build_vocab(texts, min_freq)
        # self.id_to_token = {v: k for k, v in self.vocab.items()}
        pass  # Elimina esta línea cuando descomentes

    def encode(self, text: str) -> List[int]:
        """Convierte texto a IDs."""
        # Descomenta las siguientes líneas:
        # tokens = re.findall(r'\b\w+\b', text.lower())
        # unk_id = self.vocab.get('<UNK>', 1)
        # return [self.vocab.get(t, unk_id) for t in tokens]
        pass  # Elimina esta línea cuando descomentes

    def decode(self, ids: List[int]) -> List[str]:
        """Convierte IDs a tokens."""
        # Descomenta la siguiente línea:
        # return [self.id_to_token.get(i, '<UNK>') for i in ids]
        pass  # Elimina esta línea cuando descomentes

    @property
    def vocab_size(self) -> int:
        """Tamaño del vocabulario."""
        return len(self.vocab)


# Descomenta las siguientes líneas:
# tokenizer = SimpleTokenizer()
# tokenizer.fit(corpus, min_freq=1)
#
# print(f'Tokenizer entrenado con {tokenizer.vocab_size} tokens')
#
# test_text = "el perro come carne"
# ids = tokenizer.encode(test_text)
# tokens = tokenizer.decode(ids)
#
# print(f'Texto: "{test_text}"')
# print(f'IDs: {ids}')
# print(f'Tokens: {tokens}')

print()
print("=" * 50)
print("¡Ejercicio completado!")
print("Ahora sabes tokenizar texto y construir vocabularios.")
