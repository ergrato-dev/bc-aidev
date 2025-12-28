# Dockerfile para ML API
#
# Este Dockerfile crea una imagen optimizada para servir
# un modelo de Machine Learning con FastAPI.
#
# Construir: docker build -t ml-api:1.0.0 .
# Ejecutar: docker run -p 8000:8000 ml-api:1.0.0


# ============================================
# PASO 1: Imagen Base
# ============================================
# Descomenta la imagen base
# Usamos python:3.11-slim para menor tamaño

# FROM python:3.11-slim


# ============================================
# PASO 2: Metadata (opcional)
# ============================================
# Descomenta las etiquetas

# LABEL maintainer="tu-email@example.com"
# LABEL version="1.0.0"
# LABEL description="ML API for Iris classification"


# ============================================
# PASO 3: Variables de Entorno
# ============================================
# Descomenta las variables de entorno
# Estas optimizan Python para contenedores

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PIP_NO_CACHE_DIR=1 \
#     PIP_DISABLE_PIP_VERSION_CHECK=1


# ============================================
# PASO 4: Directorio de Trabajo
# ============================================
# Descomenta para establecer el directorio de trabajo

# WORKDIR /app


# ============================================
# PASO 5: Instalar Dependencias
# ============================================
# Copiamos requirements.txt primero para aprovechar
# el cache de Docker (las dependencias cambian poco)

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt


# ============================================
# PASO 6: Copiar Aplicación
# ============================================
# Copiamos el resto del código
# Esto se hace después de instalar dependencias
# para aprovechar el cache si solo cambia el código

# COPY app/ ./app/
# COPY models/ ./models/


# ============================================
# PASO 7: Exponer Puerto
# ============================================
# Documentamos qué puerto usa la aplicación

# EXPOSE 8000


# ============================================
# PASO 8: Health Check
# ============================================
# Docker verificará periódicamente que el servicio
# está funcionando correctamente

# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1


# ============================================
# PASO 9: Comando de Ejecución
# ============================================
# Comando que se ejecuta al iniciar el contenedor

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
