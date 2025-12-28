"""
Proyecto: Asistente Especializado
=================================

Crea un chatbot con personalidad usando prompt engineering.

Instrucciones:
1. Implementa cada función marcada con TODO
2. Elige o crea una personalidad para tu asistente
3. Prueba con conversaciones de ejemplo

Tip: Empieza con las funciones básicas y añade features gradualmente.
"""

from dataclasses import dataclass
from typing import Optional

# ============================================
# CONFIGURACIÓN DEL ASISTENTE
# ============================================


@dataclass
class AssistantConfig:
    """Configuración de personalidad del asistente."""

    name: str
    role: str
    expertise: str
    tone: str
    traits: list[str]
    greeting: str
    out_of_scope_response: str


# TODO: Define la configuración de tu asistente
# Puedes usar uno de los ejemplos o crear uno propio
ASSISTANT_CONFIG = AssistantConfig(
    name="",  # TODO: Nombre del asistente
    role="",  # TODO: Rol/profesión
    expertise="",  # TODO: Área de conocimiento
    tone="",  # TODO: Tono (friendly, professional, etc.)
    traits=[],  # TODO: Lista de rasgos de personalidad
    greeting="",  # TODO: Mensaje de bienvenida
    out_of_scope_response="",  # TODO: Respuesta para preguntas fuera de alcance
)


# ============================================
# PROMPT TEMPLATES
# ============================================

SYSTEM_PROMPT_TEMPLATE = """
# TODO: Diseña el system prompt
# Incluye:
# - Nombre y rol
# - Personalidad y traits
# - Área de expertise
# - Reglas de comportamiento
# - Formato de respuesta
"""

CONVERSATION_TEMPLATE = """
# TODO: Diseña el template de conversación
# Incluye:
# - System prompt
# - Historial de conversación
# - Input del usuario
# - Formato para la respuesta
"""


# ============================================
# MEMORIA CONVERSACIONAL
# ============================================


class ConversationMemory:
    """Maneja el historial de conversación."""

    def __init__(self, max_turns: int = 10):
        """
        Inicializa la memoria.

        Args:
            max_turns: Número máximo de turnos a recordar
        """
        # TODO: Implementar inicialización
        # - Lista para mensajes
        # - Límite de turnos
        pass

    def add_message(self, role: str, content: str) -> None:
        """
        Añade un mensaje al historial.

        Args:
            role: 'user' o 'assistant'
            content: Contenido del mensaje
        """
        # TODO: Implementar
        # - Añadir mensaje con rol
        # - Respetar límite de turnos
        pass

    def get_history(self) -> str:
        """
        Retorna el historial formateado.

        Returns:
            Historial como string formateado
        """
        # TODO: Implementar
        # - Formatear mensajes como "User: ...\nAssistant: ..."
        pass

    def clear(self) -> None:
        """Limpia el historial."""
        # TODO: Implementar
        pass

    def get_summary(self) -> dict:
        """
        Retorna estadísticas de la conversación.

        Returns:
            Dict con métricas de la conversación
        """
        # TODO: Implementar
        # - Número de mensajes
        # - Mensajes por rol
        pass


# ============================================
# ASISTENTE PRINCIPAL
# ============================================


class SpecializedAssistant:
    """Asistente virtual especializado con personalidad."""

    def __init__(self, config: AssistantConfig, model_name: str = "gpt2"):
        """
        Inicializa el asistente.

        Args:
            config: Configuración de personalidad
            model_name: Modelo a usar para generación
        """
        # TODO: Implementar inicialización
        # - Guardar configuración
        # - Crear memoria
        # - Cargar modelo (opcional - puede ser simulado)
        pass

    def build_system_prompt(self) -> str:
        """
        Construye el system prompt basado en la configuración.

        Returns:
            System prompt formateado
        """
        # TODO: Implementar
        # - Usar config para llenar template
        pass

    def build_full_prompt(self, user_input: str) -> str:
        """
        Construye el prompt completo con contexto.

        Args:
            user_input: Mensaje del usuario

        Returns:
            Prompt completo para el modelo
        """
        # TODO: Implementar
        # - System prompt
        # - Historial
        # - Input actual
        pass

    def preprocess_input(self, user_input: str) -> str:
        """
        Preprocesa el input del usuario.

        Args:
            user_input: Input crudo del usuario

        Returns:
            Input limpio y validado
        """
        # TODO: Implementar
        # - Limpiar whitespace
        # - Validar longitud
        # - Sanitizar si es necesario
        pass

    def is_in_scope(self, user_input: str) -> bool:
        """
        Verifica si la pregunta está dentro del alcance.

        Args:
            user_input: Pregunta del usuario

        Returns:
            True si está en alcance, False si no
        """
        # TODO: Implementar
        # - Detectar palabras clave del dominio
        # - O usar heurísticas simples
        pass

    def generate_response(self, user_input: str) -> str:
        """
        Genera una respuesta al input del usuario.

        Args:
            user_input: Mensaje del usuario

        Returns:
            Respuesta del asistente
        """
        # TODO: Implementar
        # 1. Preprocesar input
        # 2. Verificar si está en alcance
        # 3. Construir prompt
        # 4. Generar respuesta (modelo o simulada)
        # 5. Actualizar memoria
        # 6. Retornar respuesta
        pass

    def greet(self) -> str:
        """
        Retorna el saludo inicial.

        Returns:
            Mensaje de bienvenida
        """
        # TODO: Implementar
        pass

    def chat(self, user_input: str) -> str:
        """
        Método principal de chat.

        Args:
            user_input: Mensaje del usuario

        Returns:
            Respuesta del asistente
        """
        # TODO: Implementar
        # - Wrapper principal que use generate_response
        pass


# ============================================
# INTERFAZ DE USUARIO
# ============================================


def run_chat_interface(assistant: SpecializedAssistant) -> None:
    """
    Ejecuta la interfaz de chat en terminal.

    Args:
        assistant: Instancia del asistente
    """
    # TODO: Implementar
    # - Mostrar saludo
    # - Loop de chat
    # - Comandos especiales (/quit, /clear, /stats)
    # - Manejo de errores
    pass


# ============================================
# MAIN
# ============================================


def main():
    """Función principal."""
    print("=" * 50)
    print("Proyecto: Asistente Especializado")
    print("=" * 50)

    # TODO: Implementar
    # 1. Verificar que ASSISTANT_CONFIG esté completo
    # 2. Crear instancia de SpecializedAssistant
    # 3. Ejecutar interfaz de chat

    print("\n⚠️  Proyecto no implementado")
    print("Completa los TODOs para crear tu asistente.")
    print("\nPasos sugeridos:")
    print("1. Define ASSISTANT_CONFIG con tu personalidad")
    print("2. Implementa ConversationMemory")
    print("3. Implementa SpecializedAssistant")
    print("4. Implementa run_chat_interface")
    print("5. Prueba con conversaciones de ejemplo")


if __name__ == "__main__":
    main()
