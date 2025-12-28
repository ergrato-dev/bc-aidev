# 📚 Recursos - Semana 25: Transformers

## 📖 Documentación Oficial

- [PyTorch nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- [PyTorch nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

## 📄 Papers Fundamentales

### Attention Is All You Need (2017)
- **Autores**: Vaswani et al. (Google Brain)
- **Link**: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **Importancia**: Paper original que introdujo la arquitectura Transformer

### BERT (2018)
- **Autores**: Devlin et al. (Google AI)
- **Link**: [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- **Importancia**: Pre-training bidireccional con Transformers

### GPT (2018)
- **Autores**: Radford et al. (OpenAI)
- **Link**: [OpenAI Blog](https://openai.com/research/language-unsupervised)
- **Importancia**: Generative Pre-Training con decoder-only

---

## 🎥 Videografía

### Tutoriales Recomendados

| Video | Canal | Duración | Enlace |
|-------|-------|----------|--------|
| Attention in Transformers | StatQuest | 20 min | [YouTube](https://www.youtube.com/watch?v=PSs6nxngL6k) |
| Transformers from Scratch | Andrej Karpathy | 2h | [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| The Illustrated Transformer | Jay Alammar | - | [Blog](https://jalammar.github.io/illustrated-transformer/) |

### Cursos Online

- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/)
- [Hugging Face Course](https://huggingface.co/course)
- [fast.ai NLP Course](https://www.fast.ai/)

---

## 🌐 Webgrafía

### Blogs y Tutoriales

- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) - Harvard NLP
- [Transformers from Scratch](https://peterbloem.nl/blog/transformers) - Peter Bloem
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) - Lilian Weng

### Visualizaciones Interactivas

- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [BertViz](https://github.com/jessevig/bertviz) - Visualizar atención de BERT
- [Attention Visualization](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

---

## 📦 Librerías Útiles

```bash
# Transformers de Hugging Face
pip install transformers

# Datasets
pip install datasets

# Visualización de atención
pip install bertviz

# Einops para operaciones de tensores
pip install einops
```

---

## 🔧 Herramientas

| Herramienta | Uso | Link |
|-------------|-----|------|
| Weights & Biases | Tracking experimentos | [wandb.ai](https://wandb.ai) |
| TensorBoard | Visualización | [tensorflow.org](https://www.tensorflow.org/tensorboard) |
| Netron | Visualizar arquitecturas | [netron.app](https://netron.app) |

---

## 📊 Datasets para Práctica

- **IMDB Reviews**: Clasificación de sentimientos
- **SST-2**: Stanford Sentiment Treebank
- **AG News**: Clasificación de noticias
- **MNLI**: Natural Language Inference

```python
from datasets import load_dataset

# Cargar IMDB
imdb = load_dataset("imdb")

# Cargar SST-2
sst2 = load_dataset("glue", "sst2")
```

---

## 📚 Libros Recomendados

1. **"Natural Language Processing with Transformers"** - Lewis Tunstall et al.
   - Editorial: O'Reilly
   - [Libro](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/)

2. **"Deep Learning for Natural Language Processing"** - Palash Goyal et al.
   - Editorial: Apress

---

_Última actualización: Diciembre 2024_
