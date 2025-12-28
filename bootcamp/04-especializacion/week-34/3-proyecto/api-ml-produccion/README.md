# 🚀 Proyecto: API ML en Producción

## 🎯 Objetivo

Crear una API completa de Machine Learning lista para producción, incluyendo containerización con Docker y monitoreo con Prometheus/Grafana.

---

## 📋 Descripción

Este proyecto integra todos los conceptos de la semana:

1. **FastAPI** para servir el modelo
2. **Pydantic** para validación de datos
3. **Docker** para containerización
4. **Docker Compose** para orquestación
5. **Prometheus** para métricas
6. **Grafana** para visualización

---

## ⏱️ Duración

**2 horas**

---

## 📁 Estructura del Proyecto

```
api-ml-produccion/
├── README.md
└── starter/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py              # Aplicación FastAPI principal
    │   ├── config.py            # Configuración con Pydantic Settings
    │   ├── models/
    │   │   ├── __init__.py
    │   │   └── schemas.py       # Schemas Pydantic
    │   ├── routers/
    │   │   ├── __init__.py
    │   │   ├── health.py        # Endpoints de salud
    │   │   └── predict.py       # Endpoints de predicción
    │   ├── services/
    │   │   ├── __init__.py
    │   │   └── ml_model.py      # Servicio del modelo ML
    │   └── monitoring/
    │       ├── __init__.py
    │       └── metrics.py       # Métricas Prometheus
    ├── ml_models/
    │   └── .gitkeep             # Carpeta para modelos
    ├── monitoring/
    │   ├── prometheus.yml       # Config Prometheus
    │   └── grafana/
    │       └── provisioning/
    │           └── datasources/
    │               └── prometheus.yml
    ├── Dockerfile
    ├── docker-compose.yml
    ├── .dockerignore
    ├── .env.example
    └── requirements.txt
```

---

## 🔧 Requisitos

- Python 3.11+
- Docker y Docker Compose
- Ejercicios 01-03 completados

---

## 📝 Tareas

### Parte 1: Configuración (20 min)

1. Revisar la estructura del proyecto
2. Completar `app/config.py` con Pydantic Settings
3. Configurar variables de entorno en `.env`

### Parte 2: API FastAPI (30 min)

1. Completar `app/services/ml_model.py` - servicio del modelo
2. Completar `app/routers/predict.py` - endpoint de predicción
3. Completar `app/routers/health.py` - health checks
4. Integrar routers en `app/main.py`

### Parte 3: Métricas (20 min)

1. Completar `app/monitoring/metrics.py`
2. Integrar métricas en los endpoints
3. Agregar endpoint `/metrics`

### Parte 4: Docker (20 min)

1. Completar el `Dockerfile`
2. Completar `docker-compose.yml` con todos los servicios
3. Configurar Prometheus y Grafana

### Parte 5: Testing y Validación (30 min)

1. Construir y ejecutar el stack
2. Probar todos los endpoints
3. Verificar métricas en Prometheus
4. Crear dashboard básico en Grafana

---

## 🚀 Ejecución

```bash
# Copiar variables de entorno
cp .env.example .env

# Construir e iniciar
docker compose up --build

# Endpoints disponibles:
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

---

## ✅ Criterios de Evaluación

### Funcionalidad (40%)
- [ ] API inicia correctamente
- [ ] Endpoint `/predict` funciona
- [ ] Validación de datos correcta
- [ ] Health checks implementados

### Docker (30%)
- [ ] Dockerfile optimizado
- [ ] Docker Compose con todos los servicios
- [ ] Variables de entorno configuradas
- [ ] Health checks en Docker

### Monitoreo (20%)
- [ ] Métricas expuestas en `/metrics`
- [ ] Prometheus scraping correctamente
- [ ] Métricas de requests y modelo
- [ ] Grafana accesible

### Código (10%)
- [ ] Código limpio y documentado
- [ ] Estructura de proyecto clara
- [ ] Manejo de errores apropiado

---

## 🔗 Recursos

- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [Docker Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
