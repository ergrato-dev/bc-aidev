# 🗣️ Template: Proyecto NLP

Template para proyectos de Procesamiento de Lenguaje Natural.

## 📁 Estructura

```
proyecto-nlp/
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
│   └── 02_training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocess.py
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

## 🚀 Ideas de Proyecto NLP

1. **Clasificador de Sentimiento** - Analizar opiniones/reviews
2. **Chatbot RAG** - Q&A sobre documentos
3. **Detector de Spam** - Clasificar emails/mensajes
4. **Extractor de Entidades** - NER para textos específicos
5. **Resumidor de Textos** - Resúmenes automáticos

## 🛠️ Stack Sugerido

- Hugging Face Transformers
- LangChain (para RAG)
- FastAPI
- Gradio/Streamlit
