"""
Aplicación FastAPI principal.

Este es el punto de entrada de la aplicación.
Aquí se configura FastAPI, middleware y routers.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# TODO: Importar configuración y módulos
# from app.config import get_settings
# from app.services.ml_model import ml_model
# from app.routers import health, predict
# from app.monitoring.metrics import MODEL_LOADED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# Lifecycle Manager
# ============================================
# TODO: Implementar startup y shutdown

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Manejar startup y shutdown."""
#     # Startup
#     logger.info("🚀 Iniciando API ML...")
#     settings = get_settings()
#
#     # Cargar modelo
#     try:
#         ml_model.load()
#         MODEL_LOADED.labels(model_version=ml_model.version).set(1)
#         logger.info(f"✅ Modelo cargado: v{ml_model.version}")
#     except Exception as e:
#         logger.error(f"❌ Error cargando modelo: {e}")
#         MODEL_LOADED.labels(model_version="unknown").set(0)
#
#     yield
#
#     # Shutdown
#     logger.info("👋 Apagando API ML...")


# ============================================
# Crear aplicación FastAPI
# ============================================
# TODO: Configurar FastAPI con settings

# settings = get_settings()

app = FastAPI(
    title="ML API Production",  # TODO: Usar settings.app_name
    version="1.0.0",  # TODO: Usar settings.app_version
    description="API de Machine Learning lista para producción",
    # lifespan=lifespan  # TODO: Descomentar cuando lifespan esté listo
)


# ============================================
# Middleware
# ============================================
# TODO: Agregar middleware CORS

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# ============================================
# Routers
# ============================================
# TODO: Incluir routers

# app.include_router(health.router)
# app.include_router(predict.router)


# ============================================
# Endpoint raíz
# ============================================


@app.get("/")
def root():
    """Endpoint raíz con información de la API."""
    return {
        "name": "ML API Production",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


# ============================================
# Endpoint de métricas (temporal)
# ============================================
# TODO: Mover a router de monitoring

# from app.monitoring.metrics import get_metrics
#
# @app.get("/metrics")
# async def metrics():
#     """Endpoint de métricas Prometheus."""
#     return get_metrics()
