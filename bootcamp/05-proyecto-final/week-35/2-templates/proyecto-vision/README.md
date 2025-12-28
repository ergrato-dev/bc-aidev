# 🖼️ Template: Proyecto Computer Vision

Template para proyectos de Visión por Computadora.

## 📁 Estructura

```
proyecto-vision/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── test/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── transforms.py
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

## 🚀 Ideas de Proyecto CV

1. **Clasificador de Imágenes** - Categorizar imágenes
2. **Detector de Objetos** - Encontrar objetos con YOLO
3. **Segmentación** - Segmentar regiones de imágenes
4. **OCR** - Extraer texto de imágenes
5. **Reconocimiento Facial** - Detectar/reconocer caras

## 🛠️ Stack Sugerido

- TensorFlow/Keras o PyTorch
- Ultralytics (YOLO)
- FastAPI
- Gradio
- OpenCV
