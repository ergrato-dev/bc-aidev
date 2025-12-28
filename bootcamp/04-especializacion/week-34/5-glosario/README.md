# 📖 Glosario - Semana 34: MLOps

Términos clave de MLOps, APIs y deployment ordenados alfabéticamente.

---

## A

### API (Application Programming Interface)
Interfaz que permite la comunicación entre diferentes aplicaciones. En ML, las APIs exponen modelos para que otros sistemas puedan consumir predicciones.

### Artifact
Cualquier archivo o dato generado durante el ciclo de ML: modelos entrenados, datasets procesados, métricas, logs.

---

## B

### Blue-Green Deployment
Estrategia de deployment donde se mantienen dos entornos idénticos (blue y green) para minimizar downtime durante actualizaciones.

### Build
Proceso de crear una imagen Docker a partir de un Dockerfile y el código fuente.

---

## C

### Canary Release
Estrategia de deployment gradual donde una nueva versión se expone a un pequeño porcentaje de usuarios antes del rollout completo.

### CI/CD (Continuous Integration/Continuous Deployment)
Prácticas de automatización para integrar código frecuentemente (CI) y desplegarlo automáticamente (CD).

### Container
Unidad de software que empaqueta código y dependencias para ejecutarse de forma aislada y consistente.

### Counter (Prometheus)
Tipo de métrica que solo puede incrementar. Útil para contar requests, errores, predicciones.

---

## D

### Data Drift
Cambio en la distribución de los datos de entrada respecto a los datos de entrenamiento.

### Docker
Plataforma para desarrollar, enviar y ejecutar aplicaciones en contenedores.

### Docker Compose
Herramienta para definir y ejecutar aplicaciones multi-contenedor con un archivo YAML.

### Dockerfile
Archivo de texto con instrucciones para construir una imagen Docker.

---

## E

### Endpoint
URL específica de una API donde se pueden realizar operaciones. Ejemplo: `/api/v1/predict`.

### Environment Variables
Variables de configuración definidas fuera del código, usadas para configurar aplicaciones sin modificar código.

---

## F

### FastAPI
Framework moderno de Python para crear APIs, con validación automática y documentación OpenAPI.

### Feature Store
Sistema centralizado para almacenar, versionar y servir features de ML.

---

## G

### Gauge (Prometheus)
Tipo de métrica que puede subir o bajar. Útil para valores actuales como memoria usada o requests activos.

### Grafana
Plataforma de visualización y monitoreo que se integra con múltiples fuentes de datos como Prometheus.

---

## H

### Health Check
Endpoint que verifica si un servicio está funcionando correctamente. Usado por orquestadores para detectar fallos.

### Histogram (Prometheus)
Tipo de métrica que muestrea observaciones y las cuenta en buckets configurables. Útil para latencias.

---

## I

### Image (Docker)
Template de solo lectura con instrucciones para crear un contenedor Docker.

### Inference
Proceso de usar un modelo entrenado para hacer predicciones sobre nuevos datos.

---

## K

### Kubernetes (K8s)
Plataforma de orquestación de contenedores para automatizar deployment, escalado y gestión.

---

## L

### Latency
Tiempo que tarda una operación, como el tiempo de respuesta de una API o el tiempo de inferencia de un modelo.

### Liveness Probe
Verificación periódica para determinar si un contenedor está vivo. Si falla, el contenedor se reinicia.

### Load Balancer
Componente que distribuye tráfico entre múltiples instancias de un servicio.

---

## M

### Metric
Medición cuantitativa del comportamiento de un sistema. Ejemplo: requests por segundo, latencia, errores.

### MLOps
Conjunto de prácticas para desplegar y mantener modelos de ML en producción de manera confiable y eficiente.

### Model Registry
Sistema para versionar, almacenar y gestionar modelos de ML.

### Model Serving
Proceso de hacer disponible un modelo entrenado para realizar predicciones en producción.

### Multi-stage Build
Técnica en Docker para crear imágenes optimizadas usando múltiples etapas de construcción.

---

## O

### OpenAPI (Swagger)
Especificación para describir APIs REST. FastAPI genera documentación OpenAPI automáticamente.

### Orchestration
Automatización de la configuración, coordinación y gestión de contenedores y servicios.

---

## P

### Pipeline
Secuencia automatizada de pasos para entrenar, validar y desplegar modelos de ML.

### Prometheus
Sistema de monitoreo y alertas de código abierto, diseñado para sistemas distribuidos.

### PromQL
Lenguaje de consultas de Prometheus para seleccionar y agregar datos de series temporales.

### Pydantic
Biblioteca de Python para validación de datos usando anotaciones de tipos.

---

## R

### Readiness Probe
Verificación para determinar si un contenedor está listo para recibir tráfico.

### REST (Representational State Transfer)
Estilo arquitectónico para diseñar APIs web usando métodos HTTP estándar.

### Rollback
Proceso de revertir a una versión anterior de una aplicación o modelo cuando hay problemas.

---

## S

### Scaling
Ajustar la capacidad de un sistema. Horizontal: más instancias. Vertical: más recursos por instancia.

### Schema
Definición de la estructura de datos esperada. En APIs, define el formato de requests y responses.

### Scraping (Prometheus)
Proceso donde Prometheus obtiene métricas de los endpoints `/metrics` de los servicios.

### Service Discovery
Mecanismo para detectar automáticamente servicios disponibles en una red.

---

## T

### Throughput
Cantidad de operaciones procesadas por unidad de tiempo. Ejemplo: predicciones por segundo.

---

## U

### Uvicorn
Servidor ASGI de alto rendimiento para aplicaciones Python como FastAPI.

---

## V

### Validation
Proceso de verificar que los datos cumplen con el esquema y reglas definidas antes de procesarlos.

### Volume (Docker)
Mecanismo para persistir datos generados por contenedores Docker.

---

## Fórmulas y Métricas Comunes

### Disponibilidad (Availability)
$$\text{Availability} = \frac{\text{Uptime}}{\text{Uptime} + \text{Downtime}} \times 100\%$$

### Latencia Percentil
$$P_{99} = \text{Valor donde el 99\% de las requests son más rápidas}$$

### Error Rate
$$\text{Error Rate} = \frac{\text{Requests Fallidos}}{\text{Total Requests}} \times 100\%$$

### Throughput
$$\text{Throughput} = \frac{\text{Requests Exitosos}}{\text{Tiempo}}$$

---

## PromQL Ejemplos

```promql
# Rate de requests por segundo (últimos 5 min)
rate(ml_api_requests_total[5m])

# Latencia percentil 99
histogram_quantile(0.99, rate(ml_api_request_duration_seconds_bucket[5m]))

# Tasa de errores
sum(rate(ml_api_requests_total{status_code=~"5.."}[5m])) / sum(rate(ml_api_requests_total[5m]))

# Predicciones por clase
sum by (predicted_class) (rate(ml_model_predictions_total[1h]))
```
