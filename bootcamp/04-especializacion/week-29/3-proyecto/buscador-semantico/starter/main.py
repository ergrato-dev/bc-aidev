"""
Proyecto: Buscador Semántico
============================

Implementa un motor de búsqueda semántica usando word embeddings.

Instrucciones:
1. Completa cada método marcado con TODO
2. Ejecuta el script para probar tu implementación
3. Compara con solution/main.py si necesitas ayuda
"""

import re
from typing import List, Optional, Tuple

import numpy as np


class SemanticSearchEngine:
    """Motor de búsqueda semántica usando word embeddings."""

    def __init__(self, model_name: str = "glove-wiki-gigaword-50"):
        """
        Inicializa el buscador con un modelo de embeddings.

        Args:
            model_name: Nombre del modelo de Gensim a cargar
        """
        self.model = None
        self.documents: List[str] = []
        self.doc_embeddings: List[np.ndarray] = []

        # TODO: Cargar el modelo de embeddings usando gensim.downloader
        # import gensim.downloader as api
        # self.model = api.load(model_name)
        pass

    def preprocess(self, text: str) -> str:
        """
        Preprocesa texto: minúsculas, elimina puntuación y números.

        Args:
            text: Texto a procesar

        Returns:
            Texto limpio y normalizado
        """
        # TODO: Implementar preprocesamiento
        # 1. Convertir a minúsculas
        # 2. Eliminar puntuación
        # 3. Eliminar números
        # 4. Normalizar espacios
        pass

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Calcula embedding de texto como promedio de palabras.

        Args:
            text: Texto para calcular embedding

        Returns:
            Vector embedding del texto
        """
        # TODO: Implementar cálculo de embedding
        # 1. Preprocesar texto
        # 2. Tokenizar (split)
        # 3. Obtener vectores de palabras que estén en el modelo
        # 4. Calcular promedio
        # 5. Si no hay palabras válidas, retornar vector de ceros
        pass

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calcula similaridad coseno entre dos vectores.

        Args:
            v1: Primer vector
            v2: Segundo vector

        Returns:
            Similaridad coseno (entre -1 y 1)
        """
        # TODO: Implementar similaridad coseno
        # cos(θ) = (A · B) / (||A|| × ||B||)
        pass

    def index_documents(self, documents: List[str]) -> None:
        """
        Indexa una lista de documentos calculando sus embeddings.

        Args:
            documents: Lista de documentos a indexar
        """
        # TODO: Implementar indexación
        # 1. Guardar documentos
        # 2. Calcular embedding para cada documento
        # 3. Guardar embeddings
        pass

    def add_document(self, document: str) -> int:
        """
        Añade un documento al índice.

        Args:
            document: Documento a añadir

        Returns:
            Índice del documento añadido
        """
        # TODO: Implementar añadir documento
        # 1. Añadir a lista de documentos
        # 2. Calcular y guardar embedding
        # 3. Retornar índice
        pass

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Busca documentos similares a la consulta.

        Args:
            query: Texto de búsqueda
            top_k: Número de resultados a retornar

        Returns:
            Lista de (documento, score) ordenada por relevancia
        """
        # TODO: Implementar búsqueda
        # 1. Calcular embedding de la query
        # 2. Calcular similaridad con cada documento
        # 3. Ordenar por similaridad descendente
        # 4. Retornar top_k resultados
        pass


def main():
    """Función principal para probar el buscador."""

    # Dataset de ejemplo
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "Natural language processing analyzes human language",
        "Computer vision enables machines to interpret images",
        "Reinforcement learning trains agents through rewards",
        "Python is widely used for data science projects",
        "TensorFlow and PyTorch are popular deep learning frameworks",
        "Word embeddings represent words as dense vectors",
        "Supervised learning requires labeled training data",
        "Unsupervised learning finds patterns without labels",
    ]

    print("=" * 60)
    print("🔍 Buscador Semántico")
    print("=" * 60)

    # TODO: Descomentar cuando hayas implementado la clase
    # print("\nCargando modelo de embeddings...")
    # engine = SemanticSearchEngine('glove-wiki-gigaword-50')
    #
    # print(f"Indexando {len(documents)} documentos...")
    # engine.index_documents(documents)
    # print("✓ Indexación completada")
    #
    # # Pruebas de búsqueda
    # queries = [
    #     "neural networks and AI",
    #     "programming languages",
    #     "image recognition",
    # ]
    #
    # for query in queries:
    #     print(f"\n{'─' * 60}")
    #     print(f"Query: \"{query}\"")
    #     print(f"{'─' * 60}")
    #
    #     results = engine.search(query, top_k=3)
    #
    #     for i, (doc, score) in enumerate(results, 1):
    #         print(f"  {i}. [{score:.4f}] {doc}")
    #
    # # Modo interactivo (opcional)
    # print(f"\n{'=' * 60}")
    # print("Modo interactivo (escribe 'salir' para terminar)")
    # print("=" * 60)
    #
    # while True:
    #     query = input("\n🔎 Buscar: ").strip()
    #     if query.lower() in ['salir', 'exit', 'quit', 'q']:
    #         print("¡Hasta luego!")
    #         break
    #     if not query:
    #         continue
    #
    #     results = engine.search(query, top_k=3)
    #     print("\nResultados:")
    #     for i, (doc, score) in enumerate(results, 1):
    #         print(f"  {i}. [{score:.4f}] {doc}")

    print("\n⚠️  Implementa los métodos de la clase SemanticSearchEngine")
    print("   y descomenta el código en main() para probar.")


if __name__ == "__main__":
    main()
