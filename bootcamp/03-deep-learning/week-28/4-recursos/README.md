# 📚 Recursos Adicionales - Semana 28

## 🎯 Proyecto Final de Deep Learning

Recursos para completar el proyecto integrador de Computer Vision o NLP.

---

## 📖 Documentación Oficial

### PyTorch / TorchVision
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

### Hugging Face
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Datasets Library](https://huggingface.co/docs/datasets)
- [Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Fine-tuning Tutorial](https://huggingface.co/docs/transformers/training)

---

## 🎥 Videografía

### Computer Vision
- [Transfer Learning with PyTorch](https://www.youtube.com/watch?v=K0lWSB2QoIQ) - PyTorch Official
- [Image Classification with ResNet](https://www.youtube.com/watch?v=dKU9SfRX2Wg) - Sentdex
- [Data Augmentation Techniques](https://www.youtube.com/watch?v=mTVf7BN7S8w) - Aladdin Persson

### NLP con Transformers
- [Hugging Face Course](https://www.youtube.com/playlist?list=PLo2EIpI_JMQvWfQndUesu0nPBAtZ9gP1o) - Hugging Face
- [Fine-tuning BERT](https://www.youtube.com/watch?v=x66kkDnbzi4) - James Briggs
- [Sentiment Analysis with Transformers](https://www.youtube.com/watch?v=QpzMWQvxXWk) - Venelin Valkov

### Proyecto End-to-End
- [ML Project Structure](https://www.youtube.com/watch?v=pxk1Fr33-L4) - MLOps
- [Model Deployment Basics](https://www.youtube.com/watch?v=SZF4RGWgVjk) - Patrick Loeber

---

## 📄 Papers Fundamentales

### Transfer Learning
- [ImageNet Classification with Deep CNNs (AlexNet)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)
- [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)

### Transformers y NLP
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [DistilBERT: A Distilled Version of BERT](https://arxiv.org/abs/1910.01108)
- [RoBERTa: A Robustly Optimized BERT](https://arxiv.org/abs/1907.11692)

---

## 🛠️ Herramientas Útiles

### Visualización y Debugging
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [Weights & Biases](https://wandb.ai/)
- [Netron - Model Visualizer](https://netron.app/)

### Deployment
- [Gradio](https://gradio.app/) - Demos rápidos
- [Streamlit](https://streamlit.io/) - Apps de datos
- [FastAPI](https://fastapi.tiangolo.com/) - APIs REST
- [Hugging Face Spaces](https://huggingface.co/spaces) - Hosting gratuito

### Datasets
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [TorchVision Datasets](https://pytorch.org/vision/stable/datasets.html)

---

## 📊 Datasets para Proyectos

### Computer Vision
| Dataset | Descripción | Clases | Tamaño |
|---------|-------------|--------|--------|
| CIFAR-10 | Imágenes naturales | 10 | 60K |
| CIFAR-100 | Imágenes naturales | 100 | 60K |
| Flowers-102 | Flores | 102 | 8K |
| Food-101 | Comidas | 101 | 101K |
| Stanford Dogs | Razas de perros | 120 | 20K |

### NLP
| Dataset | Descripción | Clases | Tamaño |
|---------|-------------|--------|--------|
| IMDB | Reviews de películas | 2 | 50K |
| SST-2 | Sentimiento | 2 | 70K |
| AG News | Noticias | 4 | 120K |
| Yelp Reviews | Reviews de negocios | 5 | 650K |
| Amazon Reviews | Reviews de productos | 5 | 3M |

---

## 💡 Mejores Prácticas

### Estructura de Proyecto
```
proyecto/
├── data/               # Datos (no subir a git)
├── models/             # Modelos guardados
├── notebooks/          # Exploración
├── src/
│   ├── data.py        # Carga de datos
│   ├── model.py       # Definición del modelo
│   ├── train.py       # Entrenamiento
│   └── evaluate.py    # Evaluación
├── config.yaml         # Configuración
├── requirements.txt    # Dependencias
└── README.md          # Documentación
```

### Checklist de Proyecto ML
- [ ] Reproducibilidad (seeds fijos)
- [ ] Versionado de datos y modelos
- [ ] Logging de experimentos
- [ ] Validación cruzada o hold-out
- [ ] Análisis de errores
- [ ] Documentación clara

---

## 🔗 Enlaces Rápidos

- [Semana 28 README](../README.md)
- [Guía CV](../1-teoria/01-guia-proyecto-cv.md)
- [Guía NLP](../1-teoria/02-guia-proyecto-nlp.md)
- [Proyecto CV](../3-proyecto/opcion-a-clasificador-imagenes/)
- [Proyecto NLP](../3-proyecto/opcion-b-clasificador-texto/)

---

_Recursos Semana 28 - Proyecto Final Deep Learning_
