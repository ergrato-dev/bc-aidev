"""
Router de Health Checks.

Endpoints para verificar el estado del servicio:
- /health: Health check básico
- /ready: Readiness check (¿puede recibir tráfico?)
- /live: Liveness check (¿está vivo el proceso?)
"""

from app.models.schemas import HealthResponse, ReadinessResponse
from fastapi import APIRouter

# TODO: Importar el servicio del modelo
# from app.services.ml_model import ml_model

router = APIRouter(tags=["health"])


# ============================================
# Health Check
# ============================================


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check básico.

    Usado por load balancers para verificar que el servicio responde.

    TODO: Implementar verificación real del modelo
    """
    # TODO: Verificar estado real
    # return HealthResponse(
    #     status="ok",
    #     model_loaded=ml_model.model is not None,
    #     model_version=ml_model.version if ml_model.model else None
    # )

    return HealthResponse(
        status="ok",
        model_loaded=True,  # TODO: Verificar realmente
        model_version="1.0.0",
    )


# ============================================
# Readiness Check
# ============================================


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check():
    """
    Readiness check.

    Verifica que el servicio está listo para recibir tráfico.
    Incluye verificación de dependencias.

    TODO: Implementar verificaciones reales
    """
    # TODO: Verificar dependencias
    # checks = {
    #     "model_loaded": ml_model.model is not None,
    #     "model_healthy": ml_model.is_healthy(),
    # }
    #
    # all_ready = all(checks.values())

    checks = {
        "model_loaded": True,  # TODO: Verificar realmente
    }

    return ReadinessResponse(ready=all(checks.values()), checks=checks)


# ============================================
# Liveness Check
# ============================================


@router.get("/live")
async def liveness_check():
    """
    Liveness check.

    Verifica que el proceso está vivo.
    Si falla, Kubernetes reiniciará el contenedor.
    """
    return {"status": "alive"}
