# 🏗️ Arquitectura y Diseño

## 1. Patrones de Arquitectura ML

### 1.1 Arquitectura Monolítica Simple

Ideal para proyectos pequeños y MVPs.

```
┌─────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA MONOLÍTICA                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                     FastAPI App                      │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐            │   │
│  │  │ Routes  │  │ Services│  │  Model  │            │   │
│  │  │         │──│         │──│         │            │   │
│  │  │ /predict│  │ ML Logic│  │ .pkl    │            │   │
│  │  │ /health │  │         │  │         │            │   │
│  │  └─────────┘  └─────────┘  └─────────┘            │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                     ┌─────┴─────┐                          │
│                     │  Docker   │                          │
│                     └───────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
# Estructura monolítica
"""
proyecto/
├── app/
│   ├── main.py        # FastAPI + routes + model
│   ├── schemas.py     # Pydantic models
│   └── model.pkl      # Modelo cargado
├── Dockerfile
└── requirements.txt
"""
```

**Pros**: Simple, fácil de desarrollar y desplegar
**Contras**: Difícil de escalar, todo acoplado

---

### 1.2 Arquitectura por Capas

Mejor separación de responsabilidades.

```
┌─────────────────────────────────────────────────────────────┐
│                   ARQUITECTURA POR CAPAS                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   PRESENTACIÓN                       │   │
│  │            FastAPI / Gradio / Streamlit              │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                     SERVICIOS                        │   │
│  │         Lógica de negocio, Orquestación             │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                       MODELO                         │   │
│  │            ML Model, Preprocessing                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                       DATOS                          │   │
│  │           Base de datos, Archivos                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
# Estructura por capas
"""
proyecto/
├── app/
│   ├── main.py           # Presentación
│   ├── routers/          # Endpoints
│   │   ├── predict.py
│   │   └── health.py
│   ├── services/         # Lógica de negocio
│   │   └── prediction.py
│   ├── models/           # Schemas
│   │   └── schemas.py
│   └── ml/               # ML específico
│       ├── model.py
│       └── preprocess.py
├── data/
├── Dockerfile
└── requirements.txt
"""
```

---

## 2. Diseño de API

### 2.1 Endpoints Esenciales

```python
# Endpoints mínimos para un proyecto ML
from fastapi import FastAPI

app = FastAPI(
    title="Mi Proyecto ML",
    version="1.0.0",
    description="API de predicción"
)

# 1. Health check - OBLIGATORIO
@app.get("/health")
def health():
    """Verificar que el servicio está activo."""
    return {"status": "ok", "model_loaded": True}

# 2. Información del modelo
@app.get("/model/info")
def model_info():
    """Información sobre el modelo."""
    return {
        "name": "Clasificador de Imágenes",
        "version": "1.0.0",
        "accuracy": 0.95,
        "classes": ["cat", "dog"]
    }

# 3. Predicción - OBLIGATORIO
@app.post("/predict")
def predict(data: InputSchema) -> OutputSchema:
    """Realizar predicción."""
    result = model.predict(data)
    return result

# 4. Predicción batch (opcional)
@app.post("/predict/batch")
def predict_batch(data: list[InputSchema]) -> list[OutputSchema]:
    """Predicción en lote."""
    return [model.predict(d) for d in data]
```

### 2.2 Diseño de Schemas

```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

# Input schema - validación de entrada
class PredictionInput(BaseModel):
    """Schema de entrada para predicción."""
    
    # Con validación
    feature_1: float = Field(..., ge=0, le=100, description="Feature 1")
    feature_2: float = Field(..., ge=0, le=100, description="Feature 2")
    
    # Ejemplo para documentación
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"feature_1": 5.1, "feature_2": 3.5}
            ]
        }
    }

# Output schema - respuesta estructurada
class PredictionOutput(BaseModel):
    """Schema de salida."""
    
    prediction: str = Field(..., description="Clase predicha")
    confidence: float = Field(..., ge=0, le=1, description="Confianza")
    probabilities: dict[str, float] = Field(..., description="Probabilidades")
    model_version: str = Field(default="1.0.0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Error schema
class ErrorResponse(BaseModel):
    """Schema para errores."""
    
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

---

## 3. Diseño del Modelo ML

### 3.1 Wrapper de Modelo

Encapsula el modelo para facilitar su uso:

```python
# ml/model.py
from pathlib import Path
import joblib
import numpy as np
from typing import Optional

class MLModel:
    """Wrapper para el modelo de ML."""
    
    def __init__(self, model_path: str = "models/model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.version = "1.0.0"
        self.classes: list[str] = []
    
    def load(self) -> None:
        """Cargar modelo desde disco."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        self.classes = list(self.model.classes_)
    
    def predict(self, features: list[float]) -> dict:
        """Realizar predicción."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        X = np.array(features).reshape(1, -1)
        
        # Predicción
        pred_idx = self.model.predict(X)[0]
        pred_class = self.classes[pred_idx]
        
        # Probabilidades
        probs = self.model.predict_proba(X)[0]
        prob_dict = {c: float(p) for c, p in zip(self.classes, probs)}
        
        return {
            "prediction": pred_class,
            "confidence": float(max(probs)),
            "probabilities": prob_dict
        }
    
    def is_loaded(self) -> bool:
        """Verificar si el modelo está cargado."""
        return self.model is not None

# Singleton
ml_model = MLModel()
```

### 3.2 Preprocessing Pipeline

```python
# ml/preprocess.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

class Preprocessor:
    """Pipeline de preprocessing."""
    
    def __init__(self):
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'Preprocessor':
        """Ajustar el pipeline."""
        self.pipeline.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transformar datos."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted")
        return self.pipeline.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Ajustar y transformar."""
        return self.fit(X).transform(X)
```

---

## 4. Patrones de Diseño Útiles

### 4.1 Singleton para Modelo

```python
# Garantiza una única instancia del modelo
from functools import lru_cache

@lru_cache()
def get_model() -> MLModel:
    """Obtener instancia del modelo (singleton)."""
    model = MLModel()
    model.load()
    return model

# Uso en FastAPI
from fastapi import Depends

@app.post("/predict")
def predict(
    data: PredictionInput,
    model: MLModel = Depends(get_model)
):
    return model.predict(data.features)
```

### 4.2 Factory para Modelos

```python
# Crear diferentes tipos de modelos
class ModelFactory:
    """Factory para crear modelos."""
    
    @staticmethod
    def create(model_type: str) -> MLModel:
        if model_type == "classification":
            return ClassificationModel()
        elif model_type == "regression":
            return RegressionModel()
        elif model_type == "nlp":
            return NLPModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
```

### 4.3 Strategy para Preprocessing

```python
# Diferentes estrategias de preprocessing
from abc import ABC, abstractmethod

class PreprocessStrategy(ABC):
    @abstractmethod
    def process(self, data):
        pass

class ImagePreprocess(PreprocessStrategy):
    def process(self, data):
        # Resize, normalize, etc.
        return processed_image

class TextPreprocess(PreprocessStrategy):
    def process(self, data):
        # Tokenize, clean, etc.
        return processed_text

class TabularPreprocess(PreprocessStrategy):
    def process(self, data):
        # Scale, encode, etc.
        return processed_tabular
```

---

## 5. Configuración

### 5.1 Pydantic Settings

```python
# config.py
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Configuración de la aplicación."""
    
    # App
    app_name: str = "ML Project"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Model
    model_path: Path = Path("models/model.pkl")
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

settings = Settings()
```

### 5.2 Archivo .env

```bash
# .env.example
APP_NAME=Mi Proyecto ML
APP_VERSION=1.0.0
DEBUG=false

MODEL_PATH=models/model.pkl

HOST=0.0.0.0
PORT=8000
```

---

## 6. Checklist de Arquitectura

Antes de empezar a codear:

- [ ] Arquitectura elegida (monolítica/capas)
- [ ] Estructura de carpetas definida
- [ ] Endpoints diseñados
- [ ] Schemas de entrada/salida definidos
- [ ] Wrapper de modelo planificado
- [ ] Configuración externalizada
- [ ] Manejo de errores considerado

---

## 💡 Recomendación

> Para el proyecto final, recomendamos **Arquitectura Monolítica** o **Por Capas simple**. No intentes microservicios - es overkill para un MVP de 12 horas.

**Prioriza**:
1. ✅ Funcionalidad sobre arquitectura perfecta
2. ✅ Código limpio y legible
3. ✅ Documentación clara
4. ✅ Tests básicos
