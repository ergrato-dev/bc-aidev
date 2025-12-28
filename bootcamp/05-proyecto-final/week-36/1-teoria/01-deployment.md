# 🚀 Guía de Deployment

## 1. Preparación para Deploy

### 1.1 Checklist Pre-Deploy

Antes de desplegar, verifica:

```
✅ Código funciona localmente
✅ Tests pasan
✅ requirements.txt actualizado
✅ Variables de entorno documentadas
✅ Secrets NO están en el código
✅ .gitignore configurado
✅ Dockerfile funciona localmente
```

### 1.2 Dockerfile de Producción

```dockerfile
# Dockerfile optimizado para producción
FROM python:3.11-slim as builder

WORKDIR /app

# Instalar dependencias en capa separada
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Producción
FROM python:3.11-slim

WORKDIR /app

# Crear usuario no-root
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copiar dependencias instaladas
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar aplicación
COPY --chown=appuser:appgroup . .

USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 2. Opciones de Deploy

### 2.1 Hugging Face Spaces (Recomendado para demos)

**Ideal para**: Proyectos con Gradio/Streamlit

```bash
# 1. Crear Space en huggingface.co/new-space

# 2. Estructura para Gradio
mi-space/
├── app.py          # Aplicación Gradio
├── requirements.txt
└── README.md       # Con YAML metadata

# 3. README.md con metadata
---
title: Mi Proyecto ML
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---
```

**Ejemplo app.py con Gradio**:
```python
import gradio as gr
import joblib

model = joblib.load("model.pkl")

def predict(feature1, feature2, feature3, feature4):
    features = [feature1, feature2, feature3, feature4]
    prediction = model.predict([features])[0]
    proba = model.predict_proba([features])[0]
    return f"Predicción: {prediction}", dict(zip(model.classes_, proba))

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Feature 1"),
        gr.Number(label="Feature 2"),
        gr.Number(label="Feature 3"),
        gr.Number(label="Feature 4"),
    ],
    outputs=[
        gr.Textbox(label="Predicción"),
        gr.Label(label="Probabilidades")
    ],
    title="Mi Modelo ML",
    description="Clasificador entrenado con scikit-learn"
)

demo.launch()
```

---

### 2.2 Railway (Recomendado para APIs)

**Ideal para**: FastAPI, Docker

```bash
# 1. Instalar Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Inicializar proyecto
railway init

# 4. Configurar variables de entorno
railway variables set KEY=value

# 5. Deploy
railway up
```

**railway.json** (opcional):
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

---

### 2.3 Render

**Ideal para**: Web services, APIs

```yaml
# render.yaml
services:
  - type: web
    name: ml-api
    env: docker
    dockerfilePath: ./Dockerfile
    healthCheckPath: /health
    envVars:
      - key: MODEL_PATH
        value: models/model.pkl
```

**Pasos**:
1. Conectar repositorio GitHub
2. Crear nuevo Web Service
3. Seleccionar Docker
4. Configurar variables de entorno
5. Deploy automático en cada push

---

### 2.4 Google Cloud Run

**Ideal para**: Escalabilidad automática

```bash
# 1. Instalar gcloud CLI

# 2. Configurar proyecto
gcloud config set project MI_PROYECTO

# 3. Build y push imagen
gcloud builds submit --tag gcr.io/MI_PROYECTO/ml-api

# 4. Deploy
gcloud run deploy ml-api \
  --image gcr.io/MI_PROYECTO/ml-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 3. Configuración de Variables de Entorno

### 3.1 En el código

```python
# config.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = os.getenv("MODEL_PATH", "models/model.pkl")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    model_config = {"env_file": ".env"}

settings = Settings()
```

### 3.2 .env.example

```bash
# .env.example - copiar a .env y ajustar

# App
APP_NAME=Mi Proyecto ML
DEBUG=false

# Model
MODEL_PATH=models/model.pkl

# Server
PORT=8000
```

---

## 4. Monitoreo Post-Deploy

### 4.1 Health Check

```python
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }
```

### 4.2 Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.post("/predict")
def predict(data: Input):
    logger.info(f"Prediction request received")
    result = model.predict(data)
    logger.info(f"Prediction: {result}")
    return result
```

---

## 5. Troubleshooting

### Problemas Comunes

| Problema | Causa | Solución |
|----------|-------|----------|
| Container no inicia | Puerto incorrecto | Usar `$PORT` variable |
| Modelo no carga | Path incorrecto | Usar paths relativos |
| Timeout | Modelo muy grande | Lazy loading, model caching |
| Memory error | Modelo muy grande | Usar instancia más grande |

### Debug en producción

```python
# Endpoint de debug (solo en desarrollo)
@app.get("/debug")
def debug():
    import sys
    return {
        "python_version": sys.version,
        "working_dir": os.getcwd(),
        "files": os.listdir("."),
        "env_vars": dict(os.environ)  # ¡Cuidado con secrets!
    }
```

---

## 6. Checklist Post-Deploy

- [ ] URL accesible públicamente
- [ ] /health responde 200
- [ ] /predict funciona correctamente
- [ ] /docs (Swagger) accesible
- [ ] Logs funcionando
- [ ] Variables de entorno configuradas
- [ ] HTTPS habilitado
- [ ] Performance aceptable (<2s)
