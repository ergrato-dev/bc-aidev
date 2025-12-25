# Proyecto: Analizador de Texto
# Semana 02 - Bootcamp IA
#
# Implementa las funciones marcadas con TODO

import string


# ============================================
# TEXTO DE PRUEBA
# ============================================
SAMPLE_TEXT = """
Machine learning is a subset of artificial intelligence. Machine learning 
algorithms learn from data and improve over time. Data is the foundation 
of machine learning. Without data, machine learning models cannot learn.
Deep learning is a type of machine learning that uses neural networks.
Neural networks are inspired by the human brain and can process complex data.
"""


# ============================================
# FUNCIÓN: clean_text
# ============================================
def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    - Convert to lowercase
    - Remove punctuation
    - Remove extra whitespace
    
    Args:
        text: Raw text string
    
    Returns:
        Cleaned text string
    """
    # TODO: Implementar limpieza de texto
    # 1. Convertir a minúsculas con .lower()
    # 2. Eliminar puntuación con str.translate()
    # 3. Eliminar espacios extra con .split() y ' '.join()
    pass


# ============================================
# FUNCIÓN: count_words
# ============================================
def count_words(text: str) -> int:
    """
    Count total words in text.
    
    Args:
        text: Cleaned text string
    
    Returns:
        Number of words
    """
    # TODO: Implementar conteo de palabras
    # Usar .split() para obtener lista de palabras
    pass


# ============================================
# FUNCIÓN: count_unique_words
# ============================================
def count_unique_words(text: str) -> int:
    """
    Count unique words using a set.
    
    Args:
        text: Cleaned text string
    
    Returns:
        Number of unique words
    """
    # TODO: Implementar conteo de palabras únicas
    # Usar set() para eliminar duplicados
    pass


# ============================================
# FUNCIÓN: word_frequency
# ============================================
def word_frequency(text: str) -> dict:
    """
    Calculate word frequency.
    
    Args:
        text: Cleaned text string
    
    Returns:
        Dictionary with word counts {word: count}
    """
    # TODO: Implementar frecuencia de palabras
    # Opción 1: Dict comprehension o loop
    # Opción 2: from collections import Counter
    pass


# ============================================
# FUNCIÓN: top_n_words
# ============================================
def top_n_words(freq: dict, n: int = 5) -> list:
    """
    Get top N most frequent words.
    
    Args:
        freq: Word frequency dictionary
        n: Number of top words to return
    
    Returns:
        List of tuples [(word, count), ...]
    """
    # TODO: Implementar top N palabras
    # Usar sorted() con key=lambda y reverse=True
    # Retornar los primeros n elementos
    pass


# ============================================
# FUNCIÓN: text_statistics
# ============================================
def text_statistics(text: str) -> dict:
    """
    Generate complete text statistics.
    
    Args:
        text: Raw text string
    
    Returns:
        Dictionary with all statistics
    """
    # TODO: Implementar estadísticas completas
    # 1. Limpiar texto con clean_text()
    # 2. Calcular: total_words, unique_words, char_count
    # 3. Calcular: avg_word_length, word_freq, top_words
    # 4. Retornar diccionario con todas las estadísticas
    pass


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================
def main():
    """Main function to run the text analyzer."""
    print("=" * 50)
    print("ANALIZADOR DE TEXTO")
    print("=" * 50)
    
    # Mostrar texto original
    print("\nTexto original (primeros 100 caracteres):")
    print(f'"{SAMPLE_TEXT.strip()[:100]}..."')
    
    # TODO: Descomentar cuando implementes las funciones
    
    # # Obtener estadísticas
    # stats = text_statistics(SAMPLE_TEXT)
    # 
    # if stats:
    #     # Mostrar estadísticas
    #     print("\n--- Estadísticas ---")
    #     print(f"Total palabras: {stats['total_words']}")
    #     print(f"Palabras únicas: {stats['unique_words']}")
    #     print(f"Caracteres: {stats['char_count']}")
    #     print(f"Promedio letras/palabra: {stats['avg_word_length']:.2f}")
    #     
    #     # Mostrar top palabras
    #     print("\n--- Top 5 Palabras ---")
    #     for i, (word, count) in enumerate(stats['top_words'], 1):
    #         print(f"{i}. {word}: {count}")
    
    print("\n" + "=" * 50)
    print("FIN DEL ANÁLISIS")
    print("=" * 50)


if __name__ == "__main__":
    main()
