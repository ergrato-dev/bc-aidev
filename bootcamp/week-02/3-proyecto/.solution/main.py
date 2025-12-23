# Proyecto: Analizador de Texto - SOLUCIÓN
# Semana 02 - Bootcamp IA

import string
from collections import Counter


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
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Eliminar espacios extra
    text = ' '.join(text.split())
    
    return text


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
    words = text.split()
    return len(words)


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
    words = text.split()
    unique = set(words)
    return len(unique)


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
    words = text.split()
    return dict(Counter(words))


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
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:n]


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
    # Limpiar texto
    cleaned = clean_text(text)
    
    # Calcular estadísticas básicas
    total_words = count_words(cleaned)
    unique_words = count_unique_words(cleaned)
    char_count = len(cleaned.replace(' ', ''))
    
    # Calcular promedio de longitud de palabras
    words = cleaned.split()
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    
    # Frecuencia de palabras
    freq = word_frequency(cleaned)
    top_words = top_n_words(freq, 5)
    
    return {
        'total_words': total_words,
        'unique_words': unique_words,
        'char_count': char_count,
        'avg_word_length': avg_word_length,
        'word_frequency': freq,
        'top_words': top_words
    }


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
    
    # Obtener estadísticas
    stats = text_statistics(SAMPLE_TEXT)
    
    # Mostrar estadísticas
    print("\n--- Estadísticas ---")
    print(f"Total palabras: {stats['total_words']}")
    print(f"Palabras únicas: {stats['unique_words']}")
    print(f"Caracteres: {stats['char_count']}")
    print(f"Promedio letras/palabra: {stats['avg_word_length']:.2f}")
    
    # Mostrar top palabras
    print("\n--- Top 5 Palabras ---")
    for i, (word, count) in enumerate(stats['top_words'], 1):
        print(f"{i}. {word}: {count}")
    
    print("\n" + "=" * 50)
    print("FIN DEL ANÁLISIS")
    print("=" * 50)


if __name__ == "__main__":
    main()
