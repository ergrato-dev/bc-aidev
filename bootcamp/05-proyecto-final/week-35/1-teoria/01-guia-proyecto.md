# 📘 Guía del Proyecto Final

## 🎯 Objetivo

Desarrollar un proyecto completo de Inteligencia Artificial que integre los conocimientos adquiridos durante el bootcamp, desde la preparación de datos hasta el deployment en producción.

---

## 1. Visión General del Proyecto

### 1.1 ¿Qué es un Proyecto End-to-End?

Un proyecto end-to-end incluye todas las fases del ciclo de vida de ML:

```
┌─────────────────────────────────────────────────────────────┐
│                 PROYECTO END-TO-END                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐     │
│  │ Data │──▶│Model │──▶│ API  │──▶│Deploy│──▶│ Demo │     │
│  └──────┘   └──────┘   └──────┘   └──────┘   └──────┘     │
│                                                             │
│  Semana 35:          Semana 36:                            │
│  • Datos             • Deployment                          │
│  • Modelo            • Documentación                       │
│  • API               • Presentación                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Componentes Requeridos

| Componente | Descripción | Entregable |
|------------|-------------|------------|
| **Datos** | Dataset preparado y procesado | Notebook o script |
| **Modelo** | Modelo entrenado y evaluado | Archivo .pkl/.pt/.h5 |
| **API** | Endpoints para predicción | FastAPI app |
| **Docker** | Containerización | Dockerfile |
| **Deploy** | Aplicación en la nube | URL pública |
| **Docs** | Documentación completa | README.md |

---

## 2. Fases del Proyecto

### 2.1 Fase 1: Definición (30 min)

```python
# Preguntas a responder
proyecto = {
    "problema": "¿Qué problema resuelve?",
    "usuario": "¿Quién lo usará?",
    "valor": "¿Qué valor aporta?",
    "alcance": "¿Qué incluye el MVP?",
    "metricas": "¿Cómo medimos el éxito?"
}
```

**Checklist de Definición:**
- [ ] Problema claramente definido
- [ ] Usuario objetivo identificado
- [ ] Alcance del MVP establecido
- [ ] Métricas de éxito definidas
- [ ] Stack tecnológico seleccionado

### 2.2 Fase 2: Datos (1 hora)

```python
# Pipeline típico de datos
"""
1. Obtención
   - Descargar dataset (Kaggle, Hugging Face, API)
   - Verificar licencia y términos de uso
   
2. Exploración
   - Análisis exploratorio (EDA)
   - Identificar problemas de calidad
   
3. Preparación
   - Limpieza
   - Transformaciones
   - Split train/val/test
   
4. Versionado
   - Guardar datasets procesados
   - Documentar transformaciones
"""
```

**Fuentes de Datos Recomendadas:**

| Fuente | Tipo | URL |
|--------|------|-----|
| Kaggle | General | kaggle.com/datasets |
| Hugging Face | NLP/CV | huggingface.co/datasets |
| UCI ML | Clásicos | archive.ics.uci.edu |
| Papers With Code | Benchmarks | paperswithcode.com |

### 2.3 Fase 3: Modelo (1.5 horas)

```python
# Estructura típica de entrenamiento

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Pipeline de entrenamiento.
    
    Returns:
        model: Modelo entrenado
        metrics: Diccionario de métricas
    """
    # 1. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 2. Entrenar
    model = create_model()
    model.fit(X_train, y_train)
    
    # 3. Evaluar
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred)
    }
    
    # 4. Guardar
    joblib.dump(model, "models/model.pkl")
    
    return model, metrics
```

**Checklist del Modelo:**
- [ ] Baseline establecido
- [ ] Modelo principal entrenado
- [ ] Hiperparámetros ajustados
- [ ] Métricas documentadas
- [ ] Modelo guardado correctamente

### 2.4 Fase 4: API (1 hora)

```python
# Estructura mínima de API

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI(title="Mi Proyecto ML")

# Cargar modelo al inicio
model = joblib.load("models/model.pkl")

class PredictionInput(BaseModel):
    features: list[float]

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        prediction = model.predict([input_data.features])[0]
        confidence = model.predict_proba([input_data.features]).max()
        return PredictionOutput(
            prediction=str(prediction),
            confidence=float(confidence)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 3. Estructura de Proyecto Recomendada

```
mi-proyecto-final/
├── README.md               # Documentación principal
├── requirements.txt        # Dependencias
├── Dockerfile             # Containerización
├── docker-compose.yml     # Orquestación
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/               # Datos originales
│   ├── processed/         # Datos procesados
│   └── README.md          # Documentación de datos
│
├── notebooks/
│   ├── 01_eda.ipynb       # Análisis exploratorio
│   ├── 02_training.ipynb  # Entrenamiento
│   └── 03_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/              # Módulo de datos
│   │   ├── __init__.py
│   │   ├── load.py
│   │   └── preprocess.py
│   ├── models/            # Módulo de modelos
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app
│   ├── schemas.py         # Pydantic models
│   └── config.py          # Configuración
│
├── models/                # Modelos guardados
│   └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_model.py
│
└── docs/
    ├── architecture.md
    └── api.md
```

---

## 4. README Profesional

Tu README debe incluir:

```markdown
# 🚀 Nombre del Proyecto

Descripción breve y clara del proyecto.

![Demo](docs/demo.gif)

## 🎯 Problema

¿Qué problema resuelve?

## ✨ Features

- Feature 1
- Feature 2
- Feature 3

## 🛠️ Stack Tecnológico

- Python 3.11+
- FastAPI
- TensorFlow/PyTorch
- Docker

## 🚀 Quick Start

\`\`\`bash
# Clonar
git clone https://github.com/usuario/proyecto.git
cd proyecto

# Instalar
pip install -r requirements.txt

# Ejecutar
uvicorn app.main:app --reload
\`\`\`

## 📊 Resultados

| Métrica | Valor |
|---------|-------|
| Accuracy | 95% |
| F1-Score | 0.94 |

## 📖 API

### POST /predict

\`\`\`bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
\`\`\`

## 🐳 Docker

\`\`\`bash
docker compose up --build
\`\`\`

## 📝 Licencia

MIT
```

---

## 5. Errores Comunes a Evitar

### ❌ No hagas esto:

```python
# 1. Hardcoded paths
model = load("C:/Users/mi_usuario/Desktop/model.pkl")  # ❌

# 2. Sin manejo de errores
prediction = model.predict(data)  # ❌ ¿Y si falla?

# 3. Secrets en código
API_KEY = "sk-abc123"  # ❌

# 4. Sin validación de entrada
def predict(data):
    return model.predict(data)  # ❌ ¿Qué es data?
```

### ✅ Hazlo así:

```python
# 1. Paths relativos o configurables
from pathlib import Path
MODEL_PATH = Path(__file__).parent / "models" / "model.pkl"

# 2. Con manejo de errores
try:
    prediction = model.predict(data)
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    raise HTTPException(status_code=500, detail="Prediction error")

# 3. Variables de entorno
import os
API_KEY = os.getenv("API_KEY")

# 4. Con validación (Pydantic)
class InputData(BaseModel):
    features: list[float] = Field(..., min_length=4, max_length=4)
```

---

## 6. Timeline Sugerido

### Semana 35 (6 horas)

| Día | Actividad | Tiempo |
|-----|-----------|--------|
| 1 | Leer guías, elegir proyecto | 1h |
| 2 | Obtener y explorar datos | 1.5h |
| 3 | Desarrollar modelo | 2h |
| 4 | Crear API básica | 1.5h |

### Semana 36 (6 horas)

| Día | Actividad | Tiempo |
|-----|-----------|--------|
| 1 | Dockerizar y desplegar | 2h |
| 2 | Documentación y README | 2h |
| 3 | Preparar demo y presentación | 2h |

---

## ✅ Checklist Final

- [ ] Proyecto funciona localmente
- [ ] Código está en GitHub
- [ ] README completo
- [ ] API documentada
- [ ] Docker funciona
- [ ] Deploy en cloud
- [ ] Demo lista

---

## 📚 Recursos

- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [ML Project Template](https://github.com/drivendata/cookiecutter-data-science)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Railway](https://railway.app/)
