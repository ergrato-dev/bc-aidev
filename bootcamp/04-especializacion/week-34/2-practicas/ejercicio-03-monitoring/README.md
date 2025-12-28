# 📊 Ejercicio 03: Monitoreo con Prometheus

## 🎯 Objetivo

Implementar métricas de Prometheus en una API FastAPI y visualizarlas.

---

## 📋 Descripción

En este ejercicio aprenderás a:

1. Agregar métricas de Prometheus a FastAPI
2. Crear métricas personalizadas para ML
3. Configurar Prometheus para scraping
4. Consultar métricas con PromQL

---

## ⏱️ Duración

**45 minutos**

---

## 📁 Estructura

```
ejercicio-03-monitoring/
├── README.md
└── starter/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py           # API con métricas
    │   ├── model.py          # Modelo ML
    │   ├── schemas.py        # Schemas
    │   └── metrics.py        # Métricas Prometheus
    ├── monitoring/
    │   └── prometheus.yml    # Config Prometheus
    ├── docker-compose.yml    # Stack completo
    └── requirements.txt
```

---

## 🔧 Requisitos Previos

- Docker y Docker Compose
- Ejercicio 02 completado

---

## 📝 Instrucciones

### Paso 1: Revisar la Configuración de Métricas

**Abre `starter/app/metrics.py`** y observa:

1. Tipos de métricas (Counter, Histogram, Gauge)
2. Etiquetas (labels) para dimensiones
3. Buckets para histogramas

### Paso 2: Completar el Módulo de Métricas

**En `starter/app/metrics.py`** descomenta:

1. Definición de métricas
2. Función para obtener métricas
3. Métricas específicas de ML

### Paso 3: Integrar Métricas en la API

**En `starter/app/main.py`** descomenta:

1. Import de métricas
2. Endpoint `/metrics`
3. Instrumentación de predicciones

### Paso 4: Iniciar el Stack

```bash
cd starter

# Iniciar todos los servicios
docker compose up --build -d

# Verificar servicios
docker compose ps
```

### Paso 5: Generar Tráfico

```bash
# Health check
curl http://localhost:8000/health

# Varias predicciones
for i in {1..20}; do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d "{\"sepal_length\": $((RANDOM % 3 + 4)).$((RANDOM % 10)), \"sepal_width\": $((RANDOM % 2 + 2)).$((RANDOM % 10)), \"petal_length\": $((RANDOM % 5 + 1)).$((RANDOM % 10)), \"petal_width\": $((RANDOM % 2)).$((RANDOM % 10))}"
done

# Ver métricas raw
curl http://localhost:8000/metrics
```

### Paso 6: Explorar en Prometheus

1. Abrir http://localhost:9090
2. En el campo de query, probar:

```promql
# Total de requests
ml_api_requests_total

# Requests por segundo
rate(ml_api_requests_total[1m])

# Latencia P95
histogram_quantile(0.95, rate(ml_api_request_duration_seconds_bucket[5m]))

# Predicciones por clase
ml_model_predictions_total

# Confianza promedio
avg(ml_model_prediction_confidence)
```

### Paso 7: Detener Servicios

```bash
docker compose down
```

---

## ✅ Criterios de Éxito

- [ ] Endpoint `/metrics` expone métricas Prometheus
- [ ] Métricas de requests (total, latencia)
- [ ] Métricas de modelo (predicciones, confianza)
- [ ] Prometheus scrape correctamente
- [ ] Queries PromQL funcionan

---

## 🔗 Recursos

- [prometheus_client Python](https://github.com/prometheus/client_python)
- [PromQL Basics](https://prometheus.io/docs/prometheus/latest/querying/basics/)
