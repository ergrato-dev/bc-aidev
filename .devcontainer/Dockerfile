# =============================================================================
# AI Bootcamp - Zero to Hero
# Dev Container con Python 3.13
# =============================================================================

# Python 3.13 - Última versión estable
FROM python:3.13-slim-bookworm

# Metadata
LABEL maintainer="ergrato-dev"
LABEL description="AI Bootcamp Development Environment"
LABEL python.version="3.13"

# Evitar prompts interactivos durante instalación
ENV DEBIAN_FRONTEND=noninteractive

# Configuración de Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Herramientas básicas
    git \
    curl \
    wget \
    vim \
    nano \
    # Compilación (necesario para algunas librerías)
    build-essential \
    gcc \
    g++ \
    # Librerías para data science
    libopenblas-dev \
    liblapack-dev \
    # Librerías para visualización
    libfreetype6-dev \
    libpng-dev \
    # Librerías para procesamiento de imágenes
    libjpeg-dev \
    zlib1g-dev \
    # SQLite para bases de datos locales
    sqlite3 \
    libsqlite3-dev \
    # Limpieza
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para desarrollo
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /workspace

# Copiar requirements e instalar dependencias Python
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Cambiar a usuario no-root
USER $USERNAME

# Configurar shell
ENV SHELL=/bin/bash

# Puerto por defecto para Jupyter
EXPOSE 8888

# Comando por defecto
CMD ["bash"]
