# 🚀 Ejercicio 01: API Básica con FastAPI

## 🎯 Objetivo

Crear una API REST básica con FastAPI que sirva un modelo de clasificación simple.

---

## 📋 Descripción

En este ejercicio aprenderás a:

1. Crear una aplicación FastAPI desde cero
2. Definir modelos Pydantic para validación
3. Implementar endpoints de predicción y health check
4. Probar la API con Swagger UI

---

## ⏱️ Duración

**45 minutos**

---

## 📁 Estructura

```
ejercicio-01-fastapi-basico/
├── README.md
└── starter/
    ├── main.py              # API FastAPI
    ├── schemas.py           # Modelos Pydantic
    ├── model.py             # Modelo ML simulado
    └── requirements.txt     # Dependencias
```

---

## 🔧 Requisitos Previos

- Python 3.11+
- Entorno virtual configurado

---

## 📝 Instrucciones

### Paso 1: Configurar Entorno

Abre una terminal en la carpeta `starter/`:

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 2: Revisar el Modelo Simulado

**Abre `starter/model.py`** y observa cómo se simula un modelo ML:

```python
# El modelo simula predicciones basadas en reglas simples
# En producción, cargaríamos un modelo real con joblib
```

### Paso 3: Definir Schemas Pydantic

**Abre `starter/schemas.py`** y descomenta las secciones indicadas:

1. Define `IrisFeatures` con validación de campos
2. Define `PredictionResponse` con la estructura de respuesta
3. Agrega validadores personalizados

### Paso 4: Implementar Endpoints

**Abre `starter/main.py`** y descomenta las secciones indicadas:

1. Crea la instancia de FastAPI
2. Implementa el endpoint `/health`
3. Implementa el endpoint `/predict`
4. Agrega manejo de errores

### Paso 5: Ejecutar y Probar

```bash
# Ejecutar servidor
uvicorn main:app --reload --port 8000

# Abrir en navegador
# http://localhost:8000/docs  (Swagger UI)
# http://localhost:8000/health
```

### Paso 6: Probar con curl

```bash
# Health check
curl http://localhost:8000/health

# Predicción
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

---

## ✅ Criterios de Éxito

- [ ] API inicia sin errores
- [ ] `/health` retorna `{"status": "ok"}`
- [ ] `/predict` valida entrada correctamente
- [ ] `/predict` retorna predicción con confianza
- [ ] Swagger UI funciona en `/docs`
- [ ] Errores de validación retornan código 422

---

## 🔗 Recursos

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
