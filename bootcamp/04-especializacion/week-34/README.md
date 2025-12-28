# 🚀 Semana 34: MLOps - Deployment de Modelos

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, serás capaz de:

- ✅ Crear APIs REST para modelos ML con FastAPI
- ✅ Containerizar aplicaciones ML con Docker
- ✅ Implementar pipelines de CI/CD para ML
- ✅ Monitorear modelos en producción
- ✅ Gestionar versiones de modelos y datos

---

## 📋 Contenido

### ¿Qué es MLOps?

**MLOps** (Machine Learning Operations) es la práctica de aplicar principios DevOps al ciclo de vida de Machine Learning:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CICLO MLOps                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│    │   Data   │───▶│  Train   │───▶│  Deploy  │                │
│    │  Prep    │    │  Model   │    │  & Serve │                │
│    └────┬─────┘    └────┬─────┘    └────┬─────┘                │
│         │               │               │                       │
│         ▼               ▼               ▼                       │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│    │ Version  │    │ Evaluate │    │ Monitor  │                │
│    │  Data    │    │ & Track  │    │ & Alert  │                │
│    └──────────┘    └──────────┘    └──────────┘                │
│                                                                 │
│              ◀────── Feedback Loop ──────▶                     │
└─────────────────────────────────────────────────────────────────┘
```

### Componentes Clave

| Componente | Herramienta | Función |
|------------|-------------|---------|
| **API** | FastAPI | Servir predicciones |
| **Container** | Docker | Empaquetar aplicación |
| **Registry** | Docker Hub | Almacenar imágenes |
| **CI/CD** | GitHub Actions | Automatizar deploy |
| **Monitoring** | Prometheus/Grafana | Observabilidad |

---

## 📚 Requisitos Previos

- Python avanzado
- Módulos 1-3 completados
- Conocimientos básicos de terminal/bash
- (Opcional) Cuenta en Docker Hub

---

## 🗂️ Estructura de la Semana

```
week-34/
├── README.md
├── rubrica-evaluacion.md
├── 0-assets/
│   ├── 01-mlops-lifecycle.svg
│   ├── 02-fastapi-architecture.svg
│   ├── 03-docker-layers.svg
│   └── 04-monitoring-stack.svg
├── 1-teoria/
│   ├── 01-introduccion-mlops.md
│   ├── 02-fastapi-ml.md
│   ├── 03-docker-containerization.md
│   └── 04-monitoring-production.md
├── 2-practicas/
│   ├── ejercicio-01-fastapi-basico/
│   ├── ejercicio-02-docker-ml/
│   └── ejercicio-03-monitoring/
├── 3-proyecto/
│   └── api-ml-produccion/
├── 4-recursos/
│   └── README.md
└── 5-glosario/
    └── README.md
```

---

## 📝 Contenidos

### 📖 Teoría (1.5 horas)

| # | Tema | Archivo | Duración |
|---|------|---------|----------|
| 1 | Introducción a MLOps | [01-introduccion-mlops.md](1-teoria/01-introduccion-mlops.md) | 20 min |
| 2 | FastAPI para ML | [02-fastapi-ml.md](1-teoria/02-fastapi-ml.md) | 25 min |
| 3 | Docker y Containerización | [03-docker-containerization.md](1-teoria/03-docker-containerization.md) | 25 min |
| 4 | Monitoring en Producción | [04-monitoring-production.md](1-teoria/04-monitoring-production.md) | 20 min |

### 💻 Prácticas (2.5 horas)

| # | Ejercicio | Carpeta | Duración |
|---|-----------|---------|----------|
| 1 | API con FastAPI | [ejercicio-01-fastapi-basico/](2-practicas/ejercicio-01-fastapi-basico/) | 50 min |
| 2 | Docker para ML | [ejercicio-02-docker-ml/](2-practicas/ejercicio-02-docker-ml/) | 50 min |
| 3 | Monitoring Básico | [ejercicio-03-monitoring/](2-practicas/ejercicio-03-monitoring/) | 50 min |

### 📦 Proyecto (2 horas)

| Proyecto | Descripción | Carpeta |
|----------|-------------|---------|
| API ML Producción | API completa con Docker y monitoring | [api-ml-produccion/](3-proyecto/api-ml-produccion/) |

---

## ⏱️ Distribución del Tiempo

```
Total: 6 horas

┌─────────────────────────────────────────────────────────────────┐
│  📖 Teoría      │████████░░░░░░░░░░░░░░░░│  1.5h (25%)         │
│  💻 Prácticas   │████████████████░░░░░░░░│  2.5h (42%)         │
│  📦 Proyecto    │████████████░░░░░░░░░░░░│  2.0h (33%)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Stack Tecnológico

| Tecnología | Versión | Uso |
|------------|---------|-----|
| FastAPI | 0.109+ | Framework API |
| Uvicorn | Latest | ASGI Server |
| Docker | Latest | Containerización |
| Pydantic | 2.0+ | Validación datos |
| Prometheus | Latest | Métricas |

---

## 📌 Entregables

1. **Ejercicios completados** (2-practicas/)
   - [ ] API FastAPI funcionando
   - [ ] Dockerfile válido
   - [ ] Métricas básicas implementadas

2. **Proyecto semanal** (3-proyecto/)
   - [ ] API ML desplegable
   - [ ] Docker Compose configurado
   - [ ] Documentación de endpoints

---

## 🔗 Navegación

| ⬅️ Anterior | 🏠 Módulo | Siguiente ➡️ |
|-------------|-----------|--------------|
| [Semana 33](../week-33/README.md) | [Especialización](../README.md) | [Proyecto Final](../../05-proyecto-final/README.md) |

---

## 💡 Tips para esta Semana

> 🎯 **Consejo**: MLOps es un campo amplio. Esta semana nos enfocamos en lo esencial: servir modelos via API y containerizarlos. El resto viene con la experiencia.

- **Practica localmente**: Docker y FastAPI funcionan perfectamente en tu máquina
- **Lee los logs**: Son tu mejor amigo para debugging
- **Empieza simple**: Un endpoint, un modelo, un container
- **Itera**: Agrega complejidad gradualmente

---

_Semana 34 de 36 | Módulo: Especialización | Bootcamp IA: Zero to Hero_
