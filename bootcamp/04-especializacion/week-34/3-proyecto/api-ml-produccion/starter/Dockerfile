# Dockerfile para API ML en Producción
#
# Imagen optimizada con multi-stage build
# Construir: docker build -t ml-api-prod:1.0.0 .

# ============================================
# STAGE 1: Builder
# ============================================
# TODO: Implementar stage de builder
# - Usar python:3.11-slim como base
# - Crear virtualenv
# - Instalar dependencias

# FROM python:3.11-slim as builder
# WORKDIR /app
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt


# ============================================
# STAGE 2: Production
# ============================================
# TODO: Implementar stage de producción
# - Crear usuario no-root
# - Copiar virtualenv desde builder
# - Copiar aplicación
# - Configurar health check
# - Definir CMD

# FROM python:3.11-slim as production

# Crear usuario no-root para seguridad
# RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# WORKDIR /app

# Instalar curl para healthcheck
# RUN apt-get update && apt-get install -y --no-install-recommends curl \
#     && rm -rf /var/lib/apt/lists/*

# Copiar virtualenv desde builder
# COPY --from=builder /opt/venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# Copiar aplicación
# COPY --chown=appuser:appgroup app/ ./app/
# COPY --chown=appuser:appgroup ml_models/ ./ml_models/

# Cambiar a usuario no-root
# USER appuser

# EXPOSE 8000

# HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
