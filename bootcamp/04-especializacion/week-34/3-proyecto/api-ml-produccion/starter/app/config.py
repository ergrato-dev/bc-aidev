"""
Configuración de la aplicación con Pydantic Settings.

Pydantic Settings permite:
- Cargar configuración desde variables de entorno
- Validar tipos automáticamente
- Definir valores por defecto
- Cargar desde archivos .env
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configuración de la aplicación.

    Los valores se cargan de:
    1. Variables de entorno
    2. Archivo .env
    3. Valores por defecto
    """

    # === API ===
    # TODO: Agregar campos de configuración

    # app_name: str = Field(default="ML API", description="Nombre de la aplicación")
    # app_version: str = Field(default="1.0.0", description="Versión de la API")
    # debug: bool = Field(default=False, description="Modo debug")

    # === Server ===
    # host: str = Field(default="0.0.0.0", description="Host del servidor")
    # port: int = Field(default=8000, description="Puerto del servidor")

    # === Model ===
    # model_path: str = Field(default="ml_models/model.pkl", description="Ruta al modelo")
    # model_version: str = Field(default="1.0.0", description="Versión del modelo")

    # === Logging ===
    # log_level: str = Field(default="INFO", description="Nivel de logging")

    # === Monitoring ===
    # prometheus_enabled: bool = Field(default=True, description="Habilitar métricas")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Obtener settings (cached).

    Usa lru_cache para evitar cargar settings múltiples veces.
    """
    return Settings()
