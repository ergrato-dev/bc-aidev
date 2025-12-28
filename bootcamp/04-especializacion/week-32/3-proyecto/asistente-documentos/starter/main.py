"""
Proyecto: Asistente de Documentos con RAG
=========================================

Implementa un sistema RAG completo para responder preguntas sobre documentos.

Tu tarea:
1. Completar las clases marcadas con TODO
2. Ejecutar y probar con los documentos de ejemplo
3. Añadir tus propios documentos
"""

from dataclasses import dataclass, field
from typing import Optional

# ============================================
# CONFIGURACIÓN
# ============================================

SAMPLE_DOCUMENTS = {
    "python_intro": """
    Python es un lenguaje de programación de alto nivel, interpretado y de propósito general.
    Fue creado por Guido van Rossum y lanzado por primera vez en 1991. Python enfatiza la
    legibilidad del código con su uso notable de sangría significativa. Su filosofía de diseño
    enfatiza la legibilidad del código con el uso de sangría significativa. Python es
    dinámicamente tipado y recolector de basura. Soporta múltiples paradigmas de programación,
    incluyendo programación estructurada, orientada a objetos y funcional.
    """,
    "ml_basics": """
    Machine Learning es un subconjunto de la inteligencia artificial que permite a los sistemas
    aprender y mejorar automáticamente a partir de la experiencia sin ser programados explícitamente.
    Se enfoca en el desarrollo de programas de computadora que pueden acceder a datos y usarlos
    para aprender por sí mismos. Hay tres tipos principales de Machine Learning: aprendizaje
    supervisado, donde el modelo aprende de datos etiquetados; aprendizaje no supervisado,
    donde el modelo encuentra patrones en datos no etiquetados; y aprendizaje por refuerzo,
    donde el modelo aprende a través de recompensas y castigos.
    """,
    "deep_learning": """
    Deep Learning es una rama del Machine Learning basada en redes neuronales artificiales
    con múltiples capas. Estas redes intentan simular el comportamiento del cerebro humano
    para aprender de grandes cantidades de datos. El deep learning impulsa muchos servicios
    y aplicaciones de inteligencia artificial que mejoran la automatización, realizando
    tareas analíticas y físicas sin intervención humana. Las arquitecturas más comunes
    incluyen CNNs para procesamiento de imágenes, RNNs para datos secuenciales, y
    Transformers para procesamiento de lenguaje natural.
    """,
    "rag_system": """
    RAG, o Retrieval-Augmented Generation, es un enfoque de IA que combina la recuperación
    de información con la generación de texto. En lugar de depender únicamente del conocimiento
    codificado en los parámetros del modelo, RAG permite a los modelos de lenguaje acceder
    a bases de conocimiento externas en tiempo de ejecución. El proceso funciona en dos pasos:
    primero, se recuperan documentos relevantes usando búsqueda semántica; luego, estos
    documentos se proporcionan como contexto al modelo de lenguaje para generar respuestas
    más precisas y fundamentadas. RAG es especialmente útil para reducir las alucinaciones
    y mantener el conocimiento actualizado sin necesidad de reentrenar el modelo.
    """,
}


# ============================================
# DATACLASSES
# ============================================


@dataclass
class Chunk:
    """Representa un chunk de documento."""

    text: str
    source: str
    chunk_id: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """Resultado de una búsqueda."""

    chunk: Chunk
    score: float


@dataclass
class RAGResponse:
    """Respuesta del sistema RAG."""

    question: str
    answer: str
    sources: list[str]
    context_chunks: list[Chunk]


# ============================================
# CLASE: DocumentProcessor
# ============================================


class DocumentProcessor:
    """Procesa y divide documentos en chunks."""

    def __init__(self):
        self.documents: dict[str, str] = {}

    def load_document(self, text: str, source: str) -> None:
        """
        Carga un documento.

        Args:
            text: Contenido del documento
            source: Identificador único del documento
        """
        # TODO: Implementar
        # 1. Limpiar el texto (normalizar espacios)
        # 2. Guardar en self.documents con source como key
        pass

    def load_multiple(self, documents: dict[str, str]) -> None:
        """Carga múltiples documentos."""
        # TODO: Implementar
        pass

    def chunk_text(
        self, text: str, chunk_size: int = 300, overlap: int = 50
    ) -> list[str]:
        """
        Divide un texto en chunks con overlap.

        Args:
            text: Texto a dividir
            chunk_size: Tamaño de cada chunk en caracteres
            overlap: Caracteres de solapamiento

        Returns:
            Lista de strings (chunks)
        """
        # TODO: Implementar
        # 1. Iterar sobre el texto con paso (chunk_size - overlap)
        # 2. Extraer substring de tamaño chunk_size
        # 3. Intentar cortar en el último espacio para no cortar palabras
        # 4. Retornar lista de chunks
        pass

    def chunk_all_documents(
        self, chunk_size: int = 300, overlap: int = 50
    ) -> list[Chunk]:
        """
        Divide todos los documentos cargados en chunks.

        Returns:
            Lista de objetos Chunk
        """
        # TODO: Implementar
        # 1. Iterar sobre self.documents
        # 2. Aplicar chunk_text a cada documento
        # 3. Crear objetos Chunk con source y chunk_id únicos
        # 4. Retornar lista de Chunks
        pass


# ============================================
# CLASE: VectorStore
# ============================================


class VectorStore:
    """Almacén de vectores usando ChromaDB."""

    def __init__(self, collection_name: str = "documents"):
        """
        Inicializa el almacén de vectores.

        Args:
            collection_name: Nombre de la colección en ChromaDB
        """
        # TODO: Implementar
        # 1. Importar chromadb
        # 2. Crear cliente
        # 3. Crear o obtener colección
        pass

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        Añade chunks a la colección.

        Args:
            chunks: Lista de objetos Chunk
        """
        # TODO: Implementar
        # 1. Extraer documents, ids, metadatas de los chunks
        # 2. Llamar a collection.add()
        pass

    def search(
        self, query: str, n_results: int = 3, source_filter: Optional[str] = None
    ) -> list[SearchResult]:
        """
        Busca chunks relevantes para una query.

        Args:
            query: Texto de búsqueda
            n_results: Número de resultados
            source_filter: Filtrar por fuente (opcional)

        Returns:
            Lista de SearchResult ordenada por relevancia
        """
        # TODO: Implementar
        # 1. Preparar parámetros de query
        # 2. Aplicar filtro de source si se proporciona
        # 3. Ejecutar query
        # 4. Convertir resultados a SearchResult
        pass

    def clear(self) -> None:
        """Limpia la colección."""
        # TODO: Implementar
        pass


# ============================================
# CLASE: ResponseGenerator
# ============================================


class ResponseGenerator:
    """Genera respuestas basadas en contexto."""

    def __init__(self, use_llm: bool = False):
        """
        Inicializa el generador.

        Args:
            use_llm: Si True, intenta usar OpenAI (requiere API key)
        """
        self.use_llm = use_llm
        self.client = None

        if use_llm:
            # TODO: Inicializar cliente OpenAI si hay API key
            pass

    def build_prompt(self, question: str, context_chunks: list[Chunk]) -> str:
        """
        Construye el prompt con contexto.

        Args:
            question: Pregunta del usuario
            context_chunks: Chunks de contexto

        Returns:
            Prompt completo para el LLM
        """
        # TODO: Implementar
        # 1. Formatear cada chunk con su fuente
        # 2. Crear prompt con instrucciones claras
        # 3. Incluir la pregunta
        pass

    def generate(self, question: str, context_chunks: list[Chunk]) -> str:
        """
        Genera respuesta basada en el contexto.

        Args:
            question: Pregunta del usuario
            context_chunks: Chunks de contexto

        Returns:
            Respuesta generada
        """
        # TODO: Implementar
        # Si use_llm y client disponible: usar OpenAI
        # Si no: generar respuesta simple extrayendo info del contexto
        pass


# ============================================
# CLASE: RAGAssistant
# ============================================


class RAGAssistant:
    """Asistente RAG completo."""

    def __init__(self, collection_name: str = "assistant", use_llm: bool = False):
        """
        Inicializa el asistente RAG.

        Args:
            collection_name: Nombre de la colección
            use_llm: Si usar LLM para generación
        """
        # TODO: Inicializar componentes
        # self.processor = DocumentProcessor()
        # self.store = VectorStore(collection_name)
        # self.generator = ResponseGenerator(use_llm)
        pass

    def load_documents(
        self, documents: dict[str, str], chunk_size: int = 300, overlap: int = 50
    ) -> None:
        """
        Carga y procesa documentos.

        Args:
            documents: Dict de {nombre: contenido}
            chunk_size: Tamaño de chunks
            overlap: Solapamiento
        """
        # TODO: Implementar
        # 1. Cargar documentos en processor
        # 2. Dividir en chunks
        # 3. Indexar en store
        pass

    def answer(
        self, question: str, n_context: int = 3, source_filter: Optional[str] = None
    ) -> RAGResponse:
        """
        Responde una pregunta usando RAG.

        Args:
            question: Pregunta del usuario
            n_context: Número de chunks de contexto
            source_filter: Filtrar por fuente

        Returns:
            RAGResponse con respuesta y metadatos
        """
        # TODO: Implementar
        # 1. Buscar chunks relevantes
        # 2. Generar respuesta
        # 3. Extraer fuentes
        # 4. Crear y retornar RAGResponse
        pass

    def chat(self) -> None:
        """Inicia modo de chat interactivo."""
        # TODO: Implementar
        # Loop que lee preguntas y muestra respuestas
        # Comando 'salir' para terminar
        pass


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================


def main():
    """Función principal del proyecto."""
    print("=" * 60)
    print("🤖 Asistente de Documentos con RAG")
    print("=" * 60)

    # TODO: Implementar flujo principal
    # 1. Crear instancia de RAGAssistant
    # 2. Cargar documentos de ejemplo (SAMPLE_DOCUMENTS)
    # 3. Ejecutar algunas preguntas de prueba
    # 4. Mostrar respuestas con fuentes

    print("\n⚠️ Implementa las clases y funciones marcadas con TODO")
    print("Luego ejecuta el script para probar tu asistente.")

    # Ejemplo de uso esperado:
    # assistant = RAGAssistant()
    # assistant.load_documents(SAMPLE_DOCUMENTS)
    #
    # questions = [
    #     "¿Quién creó Python?",
    #     "¿Qué es Machine Learning?",
    #     "¿Cómo funciona RAG?"
    # ]
    #
    # for q in questions:
    #     response = assistant.answer(q)
    #     print(f"\n❓ {response.question}")
    #     print(f"💡 {response.answer}")
    #     print(f"📚 Fuentes: {', '.join(response.sources)}")


if __name__ == "__main__":
    main()
