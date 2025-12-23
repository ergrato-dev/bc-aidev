![Bootcamp de Inteligencia Artificial: Zero to Hero](_assets/bootcamp-header.svg)

[![License MIT](https://img.shields.io/badge/License-MIT-0969DA?style=flat&logo=opensourceinitiative&logoColor=white)](LICENSE)
![36 Semanas](https://img.shields.io/badge/Duración-36%20Semanas-1F6FEB?style=flat)
![216 Horas](https://img.shields.io/badge/Total-216%20Horas-FF6F00?style=flat)
![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat&logo=python&logoColor=white)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-238636?style=flat&logo=git&logoColor=white)](CONTRIBUTING.md)

<div align="center">

[![🇺🇸 English Version](https://img.shields.io/badge/🇺🇸_English-Version-blue?style=for-the-badge)](README_EN.md)

</div>

---

## 📋 Descripción

Bootcamp intensivo de **36 semanas (9 meses)** diseñado para llevar a estudiantes de cero a desarrollador IA/ML Junior. Cubre desde fundamentos de Python hasta Large Language Models (LLMs) y despliegue de modelos en producción.

### 🎯 Objetivos

Al finalizar el bootcamp, los estudiantes serán capaces de:

- ✅ Dominar Python para ciencia de datos e IA
- ✅ Comprender fundamentos matemáticos (álgebra lineal, estadística, cálculo)
- ✅ Manipular y visualizar datos con NumPy, Pandas, Matplotlib
- ✅ Implementar algoritmos de Machine Learning con Scikit-learn
- ✅ Construir redes neuronales con TensorFlow/PyTorch
- ✅ Desarrollar modelos de Deep Learning (CNNs, RNNs, Transformers)
- ✅ Trabajar con NLP y LLMs usando Hugging Face
- ✅ Desplegar modelos en producción (MLOps básico)

### 🚀 ¿Por qué este Bootcamp?

> **De Zero a Hero** - Un camino estructurado desde los fundamentos hasta aplicaciones avanzadas de IA.

Este bootcamp se enfoca en el aprendizaje práctico con proyectos del mundo real. Cada semana incluye teoría, ejercicios guiados y un proyecto integrador que consolida el conocimiento adquirido.

---

## 🗓️ Estructura del Bootcamp

| Módulo               | Semanas | Horas | Contenido                                         |
| -------------------- | ------- | ----- | ------------------------------------------------- |
| **Fundamentos**      | 1-8     | 48h   | Python, Matemáticas, NumPy, Pandas, Visualización |
| **Machine Learning** | 9-18    | 60h   | Scikit-learn, Algoritmos ML, Feature Engineering  |
| **Deep Learning**    | 19-28   | 60h   | TensorFlow, PyTorch, CNNs, RNNs, Transformers     |
| **Especialización**  | 29-34   | 36h   | NLP, LLMs, Computer Vision, MLOps                 |
| **Proyecto Final**   | 35-36   | 12h   | Proyecto end-to-end en producción                 |

**Total: 36 semanas | 216 horas de formación intensiva**

---

## 📚 Contenido por Semana

Cada semana incluye:

```
bootcamp/week-XX/
├── README.md                 # Descripción y objetivos
├── rubrica-evaluacion.md     # Criterios de evaluación
├── 0-assets/                 # Imágenes y diagramas
├── 1-teoria/                 # Material teórico
├── 2-practicas/              # Ejercicios guiados
├── 3-proyecto/               # Proyecto semanal
├── 4-recursos/               # Recursos adicionales
│   ├── ebooks-free/
│   ├── videografia/
│   └── webgrafia/
└── 5-glosario/               # Términos clave
```

### 🔑 Componentes Clave

- 📖 **Teoría**: Conceptos fundamentales con ejemplos del mundo real
- 💻 **Práctica**: Ejercicios progresivos y proyectos hands-on
- 📝 **Evaluación**: Evidencias de conocimiento, desempeño y producto
- 🎓 **Recursos**: Glosarios, referencias y material complementario

---

## 🛠️ Stack Tecnológico

| Tecnología         | Versión | Uso                    |
| ------------------ | ------- | ---------------------- |
| Python             | 3.11+   | Lenguaje principal     |
| NumPy              | 1.26+   | Computación numérica   |
| Pandas             | 2.0+    | Manipulación de datos  |
| Matplotlib/Seaborn | Latest  | Visualización          |
| Scikit-learn       | 1.4+    | Machine Learning       |
| TensorFlow         | 2.15+   | Deep Learning          |
| PyTorch            | 2.1+    | Deep Learning          |
| Hugging Face       | Latest  | NLP y LLMs             |
| Docker             | Latest  | Entornos reproducibles |
| pytest             | 8+      | Testing                |

**Gestores de entorno**: `venv`, `conda`, o `poetry`

---

## 🚀 Inicio Rápido

### Prerrequisitos

- Python 3.11+
- Git para control de versiones
- Docker (recomendado para entornos reproducibles)
- VS Code (recomendado) con extensiones incluidas

### 1. Clonar el Repositorio

```bash
git clone https://github.com/epti-dev/bc-aidev.git
cd bc-aidev
```

### 2. Configurar Entorno

**Opción A: venv + pip (simple)**

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

**Opción B: conda (recomendado para Deep Learning)**

```bash
conda create -n ai-bootcamp python=3.11
conda activate ai-bootcamp
pip install -r requirements.txt
```

**Opción C: Docker (entorno reproducible)**

```bash
docker compose up --build
```

### 3. Instalar Extensiones de VS Code

```bash
# Abrir en VS Code
code .
# Las extensiones recomendadas aparecerán automáticamente
```

### 4. Navegar a la Semana Actual

```bash
cd bootcamp/week-01
```

### 5. Seguir las Instrucciones

Cada semana contiene un `README.md` con instrucciones detalladas.

---

## 📊 Metodología de Aprendizaje

### Estrategias Didácticas

- 🎯 **Aprendizaje Basado en Proyectos (ABP)**
- 🧩 **Práctica Deliberada**
- 🏆 **Kaggle Challenges**
- 👥 **Code Review entre pares**
- 📄 **Paper Reading**

### Distribución del Tiempo (6h/semana)

- **Teoría**: 1.5 horas
- **Prácticas**: 2.5 horas
- **Proyecto**: 2 horas

### Evaluación

Cada semana incluye tres tipos de evidencias:

1. **Conocimiento 🧠** (30%): Cuestionarios y evaluaciones teóricas
2. **Desempeño 💪** (40%): Ejercicios prácticos completados
3. **Producto 📦** (30%): Entregables evaluables (proyectos funcionales)

**Criterio de aprobación**: Mínimo 70% en cada tipo de evidencia

---

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Este es un proyecto educativo de código abierto.

### Cómo Contribuir

1. Lee la [Guía de Contribución](CONTRIBUTING.md)
2. Revisa el [Código de Conducta](CODE_OF_CONDUCT.md)
3. Fork del repositorio
4. Crea tu rama (`git checkout -b feat/nueva-funcionalidad`)
5. Commit con [Conventional Commits](https://www.conventionalcommits.org/) (`git commit -m 'feat: add new exercise'`)
6. Push a la rama (`git push origin feat/nueva-funcionalidad`)
7. Abre un Pull Request

### 📋 Áreas de Contribución

- ✨ Ejercicios adicionales
- 📚 Mejoras en documentación
- 🐛 Corrección de errores
- 🎨 Recursos visuales (diagramas SVG)
- 🌐 Traducciones
- 📹 Videos tutoriales

---

## 📞 Soporte

- 💬 **Discussions**: [GitHub Discussions](https://github.com/epti-dev/bc-aidev/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/epti-dev/bc-aidev/issues)

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🏆 Agradecimientos

- [Python Documentation](https://docs.python.org/3/) - Documentación oficial
- [Scikit-learn](https://scikit-learn.org/) - Por excelentes tutoriales de ML
- [TensorFlow](https://www.tensorflow.org/) - Por recursos educativos
- [PyTorch](https://pytorch.org/) - Por documentación clara
- [Hugging Face](https://huggingface.co/) - Por democratizar los LLMs
- [Kaggle](https://www.kaggle.com/) - Por datasets y competiciones
- Comunidad de IA/ML - Por los recursos y ejemplos
- Todos los contribuidores

---

## 📚 Documentación Adicional

- [🤖 Instrucciones de Copilot](.github/copilot-instructions.md)
- [🤝 Guía de Contribución](CONTRIBUTING.md)
- [📜 Código de Conducta](CODE_OF_CONDUCT.md)
- [🔒 Política de Seguridad](SECURITY.md)

---

<div align="center">

**🎓 Bootcamp de Inteligencia Artificial: Zero to Hero**

_De cero a desarrollador IA/ML Junior en 9 meses_

[Comenzar Semana 1](bootcamp/week-01) • [Ver Documentación](_docs) • [Reportar Issue](https://github.com/epti-dev/bc-aidev/issues) • [Contribuir](CONTRIBUTING.md)

---

Hecho con ❤️ para la comunidad de desarrolladores

</div>
