# 📊 Template: Proyecto Tabular

Template para proyectos con datos tabulares (ML clásico).

## 📁 Estructura

```
proyecto-tabular/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build.py
│   └── models/
│       ├── __init__.py
│       └── train.py
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── model.py
│
├── models/
│   └── .gitkeep
│
└── tests/
    ├── __init__.py
    └── test_api.py
```

## 🚀 Ideas de Proyecto Tabular

1. **Predictor de Precios** - Regresión para precios de casas/autos
2. **Clasificador de Churn** - Predecir abandono de clientes
3. **Sistema de Recomendación** - Recomendar productos
4. **Detector de Fraude** - Clasificación de anomalías
5. **Predictor de Series Temporales** - Forecasting

## 🛠️ Stack Sugerido

- Scikit-learn
- Pandas
- XGBoost/LightGBM
- FastAPI
- Streamlit
