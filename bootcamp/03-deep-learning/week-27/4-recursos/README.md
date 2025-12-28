# 📚 Recursos Adicionales - Semana 27

## 📖 Documentación Oficial

### PyTorch

- [torch.optim - Optimizers](https://pytorch.org/docs/stable/optim.html)
- [Learning Rate Schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [torch.nn.init - Initialization](https://pytorch.org/docs/stable/nn.init.html)
- [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

---

## 📄 Papers Fundamentales

### Optimizadores

| Paper | Año | Conceptos Clave |
|-------|-----|-----------------|
| [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) | 2014 | Adam optimizer, momentos adaptativos |
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | 2017 | AdamW, weight decay correcto |
| [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265) | 2019 | RAdam, warmup automático |

### Learning Rate

| Paper | Año | Conceptos Clave |
|-------|-----|-----------------|
| [Cyclical Learning Rates](https://arxiv.org/abs/1506.01186) | 2015 | CLR, LR range test |
| [Super-Convergence](https://arxiv.org/abs/1708.07120) | 2017 | OneCycleLR, entrenamiento rápido |
| [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) | 2016 | Cosine annealing con restarts |

### Inicialización

| Paper | Año | Conceptos Clave |
|-------|-----|-----------------|
| [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html) | 2010 | Xavier/Glorot initialization |
| [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) | 2015 | He/Kaiming initialization para ReLU |

---

## 🎥 Videos Recomendados

### Optimizadores

- [But what is a neural network? - 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk) - Fundamentos visuales
- [How Optimization for Machine Learning Works - StatQuest](https://www.youtube.com/watch?v=x6f5JOPhci0) - SGD explicado
- [Adam Optimizer - DeepLearning.AI](https://www.youtube.com/watch?v=JXQT_vxqwIs) - Andrew Ng explica Adam

### Learning Rate

- [Finding good learning rate - fast.ai](https://www.youtube.com/watch?v=W6NJ2P3cqkk) - LR finder
- [1cycle policy - fast.ai](https://www.youtube.com/watch?v=dxpyg3mP_rU) - Super-convergencia

---

## 📘 Libros y Capítulos

### Deep Learning (Goodfellow et al.)

- **Capítulo 8**: Optimization for Training Deep Models
  - 8.1: Learning vs Pure Optimization
  - 8.3: Exponentially Weighted Moving Averages
  - 8.5: Algorithms with Adaptive Learning Rates

### Dive into Deep Learning

- [Chapter 12: Optimization Algorithms](https://d2l.ai/chapter_optimization/index.html) - Gratuito online
  - 12.4: Momentum
  - 12.6: Adam
  - 12.10: Learning Rate Scheduling

---

## 🔧 Herramientas Útiles

### Visualización de Optimizadores

- [Optimizer Visualization](https://github.com/Jaewan-Yun/optimizer-visualization) - Comparación visual
- [Loss Landscape Visualization](https://losslandscape.com/) - Paisajes de pérdida 3D

### Debugging y Profiling

- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html) - Logging de métricas

---

## 🌐 Blogs y Artículos

### Optimizadores

- [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/) - Sebastian Ruder
- [Why Momentum Really Works](https://distill.pub/2017/momentum/) - Distill.pub (interactivo)

### Learning Rate

- [The 1cycle Policy](https://sgugger.github.io/the-1cycle-policy.html) - Sylvain Gugger
- [A disciplined approach to neural network hyper-parameters](https://arxiv.org/abs/1803.09820) - Leslie Smith

### Inicialización

- [Weight Initialization in Neural Networks](https://www.deeplearning.ai/ai-notes/initialization/) - DeepLearning.AI

---

## 💻 Código de Referencia

### Repositorios

- [pytorch/examples](https://github.com/pytorch/examples) - Ejemplos oficiales
- [fastai/fastai](https://github.com/fastai/fastai) - Implementación de 1cycle
- [huggingface/transformers](https://github.com/huggingface/transformers) - AdamW con warmup

### Notebooks

- [PyTorch Optimizer Benchmark](https://www.kaggle.com/code/residentmario/pytorch-optimizers-benchmark) - Kaggle
- [Learning Rate Finder](https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling) - Kaggle

---

## 🎓 Cursos Online

| Curso | Plataforma | Relevancia |
|-------|------------|------------|
| [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) | Coursera | Curso 2: Optimization |
| [Practical Deep Learning](https://course.fast.ai/) | fast.ai | 1cycle, AdamW |
| [Full Stack Deep Learning](https://fullstackdeeplearning.com/) | FSDL | MLOps, checkpoints |

---

## 📌 Cheatsheets

### Selección de Optimizador

```
¿Cuál optimizador usar?

Empezando → Adam (lr=0.001)
    ↓
¿Overfitting? → AdamW (weight_decay=0.01)
    ↓
¿Quieres mejor generalización? → SGD + Momentum (lr=0.1, momentum=0.9)
    ↓
¿Transfer Learning? → AdamW con lr pequeño (1e-5 a 1e-4)
```

### Selección de Scheduler

```
¿Cuál scheduler usar?

Entrenamiento rápido → OneCycleLR
    ↓
Entrenamiento largo → CosineAnnealingLR
    ↓
No sabes cuántas épocas → ReduceLROnPlateau
    ↓
Fine-tuning → LinearWarmup + CosineDecay
```

---

_Última actualización: Diciembre 2024_
