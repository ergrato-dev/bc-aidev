"""
Proyecto: Analizador de Sentimientos Multilingüe
================================================

Construye un sistema de análisis de sentimientos con soporte multilingüe.

Instrucciones:
1. Completa los TODOs en cada función
2. Ejecuta y prueba con diferentes textos
3. Experimenta con más idiomas
"""

from typing import Optional


class SentimentAnalyzer:
    """
    Analizador de sentimientos multilingüe usando Hugging Face Transformers.

    Características:
    - Detección automática de idioma
    - Soporte multilingüe (en, es, fr, de, it, nl)
    - Análisis individual y batch
    - Generación de reportes
    """

    # Mapeo de estrellas a sentimiento
    STAR_TO_SENTIMENT = {
        1: "NEGATIVE",
        2: "NEGATIVE",
        3: "NEUTRAL",
        4: "POSITIVE",
        5: "POSITIVE",
    }

    def __init__(
        self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    ):
        """
        Inicializa el analizador con el modelo especificado.

        Args:
            model_name: Nombre del modelo en Hugging Face Hub
        """
        # TODO: Importar pipeline de transformers
        # from transformers import pipeline

        # TODO: Cargar el modelo de sentiment-analysis
        # self.classifier = pipeline('sentiment-analysis', model=model_name)

        # TODO: Intentar importar langdetect (con fallback)
        # try:
        #     from langdetect import detect
        #     self.detect_language = detect
        # except ImportError:
        #     self.detect_language = lambda x: 'unknown'

        print(f"Analizador inicializado con modelo: {model_name}")

    def _parse_stars(self, label: str) -> int:
        """
        Extrae el número de estrellas del label.

        Args:
            label: Label del modelo (e.g., "5 stars", "1 star")

        Returns:
            Número de estrellas (1-5)
        """
        # TODO: Extraer número del label
        # El formato es "X stars" o "X star"
        # return int(label.split()[0])
        pass

    def _get_sentiment(self, stars: int) -> str:
        """
        Convierte estrellas a categoría de sentimiento.

        Args:
            stars: Número de estrellas (1-5)

        Returns:
            Sentimiento: POSITIVE, NEUTRAL, NEGATIVE
        """
        # TODO: Usar STAR_TO_SENTIMENT para mapear
        # return self.STAR_TO_SENTIMENT.get(stars, 'NEUTRAL')
        pass

    def detect_lang(self, text: str) -> str:
        """
        Detecta el idioma del texto.

        Args:
            text: Texto a analizar

        Returns:
            Código de idioma (e.g., 'en', 'es', 'fr')
        """
        # TODO: Usar langdetect con manejo de errores
        # try:
        #     return self.detect_language(text)
        # except Exception:
        #     return 'unknown'
        pass

    def analyze(self, text: str) -> dict:
        """
        Analiza el sentimiento de un texto.

        Args:
            text: Texto a analizar

        Returns:
            dict con: text, language, sentiment, stars, confidence
        """
        # TODO: Implementar análisis completo
        # 1. Detectar idioma
        # language = self.detect_lang(text)

        # 2. Obtener predicción del modelo
        # result = self.classifier(text)[0]

        # 3. Parsear estrellas y sentimiento
        # stars = self._parse_stars(result['label'])
        # sentiment = self._get_sentiment(stars)

        # 4. Retornar resultado estructurado
        # return {
        #     'text': text,
        #     'language': language,
        #     'sentiment': sentiment,
        #     'stars': stars,
        #     'confidence': result['score']
        # }
        pass

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """
        Analiza múltiples textos eficientemente.

        Args:
            texts: Lista de textos a analizar

        Returns:
            Lista de resultados de análisis
        """
        # TODO: Implementar análisis en batch
        # return [self.analyze(text) for text in texts]
        pass

    def generate_report(self, results: list[dict]) -> str:
        """
        Genera un reporte estadístico de los resultados.

        Args:
            results: Lista de resultados de analyze()

        Returns:
            Reporte en formato texto
        """
        # TODO: Calcular estadísticas
        # total = len(results)
        # positive = sum(1 for r in results if r['sentiment'] == 'POSITIVE')
        # neutral = sum(1 for r in results if r['sentiment'] == 'NEUTRAL')
        # negative = sum(1 for r in results if r['sentiment'] == 'NEGATIVE')
        # avg_confidence = sum(r['confidence'] for r in results) / total

        # TODO: Contar idiomas
        # languages = {}
        # for r in results:
        #     lang = r['language']
        #     languages[lang] = languages.get(lang, 0) + 1

        # TODO: Generar reporte formateado
        # report = f"""
        # === Reporte de Análisis de Sentimientos ===
        #
        # Total de textos analizados: {total}
        #
        # Distribución de sentimientos:
        #   - Positivos: {positive} ({positive/total*100:.1f}%)
        #   - Neutrales: {neutral} ({neutral/total*100:.1f}%)
        #   - Negativos: {negative} ({negative/total*100:.1f}%)
        #
        # Confianza promedio: {avg_confidence:.2%}
        #
        # Idiomas detectados:
        # """
        # for lang, count in sorted(languages.items()):
        #     report += f"  - {lang}: {count} textos\n"
        #
        # return report
        pass


def display_result(result: dict) -> None:
    """Muestra un resultado de forma visual."""
    # TODO: Implementar display visual
    # stars_emoji = '⭐' * result['stars']
    # sentiment_emoji = {
    #     'POSITIVE': '😊',
    #     'NEUTRAL': '😐',
    #     'NEGATIVE': '😞'
    # }.get(result['sentiment'], '❓')
    #
    # print(f"\nTexto: \"{result['text'][:60]}...\"" if len(result['text']) > 60
    #       else f"\nTexto: \"{result['text']}\"")
    # print(f"  Idioma: {result['language']}")
    # print(f"  Sentimiento: {result['sentiment']} {sentiment_emoji} ({stars_emoji})")
    # print(f"  Confianza: {result['confidence']:.2%}")
    pass


def main():
    """Función principal del proyecto."""
    print("=" * 60)
    print("  ANALIZADOR DE SENTIMIENTOS MULTILINGÜE")
    print("=" * 60)

    # TODO: Crear instancia del analizador
    # print('\nCargando modelo (esto puede tomar unos segundos)...')
    # analyzer = SentimentAnalyzer()

    # Textos de prueba en diferentes idiomas
    test_texts = [
        # Inglés
        "I absolutely love this product! It exceeded all my expectations.",
        "This is the worst experience I've ever had. Terrible service.",
        "The product is okay, nothing special but it works.",
        # Español
        "¡Me encanta! Es exactamente lo que necesitaba.",
        "Muy decepcionado con la calidad, no lo recomiendo.",
        "Funciona bien, cumple su función básica.",
        # Francés
        "C'est magnifique! Je suis très satisfait de mon achat.",
        "Quelle déception, le produit ne correspond pas à la description.",
        # Alemán
        "Fantastisch! Sehr gute Qualität und schnelle Lieferung.",
        "Leider bin ich enttäuscht, das Produkt ist defekt angekommen.",
        # Italiano
        "Prodotto eccellente, lo consiglio a tutti!",
        "Non sono soddisfatto, qualità molto bassa.",
    ]

    # TODO: Analizar todos los textos
    # print('\n--- Analizando textos ---')
    # results = analyzer.analyze_batch(test_texts)

    # TODO: Mostrar resultados individuales
    # for result in results:
    #     display_result(result)

    # TODO: Generar y mostrar reporte
    # print('\n' + '=' * 60)
    # report = analyzer.generate_report(results)
    # print(report)

    # TODO: Modo interactivo
    # print('\n--- Modo Interactivo ---')
    # print('Escribe un texto para analizar (o "salir" para terminar):')
    #
    # while True:
    #     text = input('\n> ').strip()
    #     if text.lower() in ['salir', 'exit', 'quit', 'q']:
    #         print('¡Hasta luego!')
    #         break
    #     if not text:
    #         continue
    #     result = analyzer.analyze(text)
    #     display_result(result)

    print("\n[Completa los TODOs para activar el analizador]")


if __name__ == "__main__":
    main()
