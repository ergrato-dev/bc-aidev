![Artificial Intelligence Bootcamp: Zero to Hero](_assets/bootcamp-header.svg)

[![License MIT](https://img.shields.io/badge/License-MIT-0969DA?style=flat&logo=opensourceinitiative&logoColor=white)](LICENSE)
![36 Weeks](https://img.shields.io/badge/Duration-36%20Weeks-1F6FEB?style=flat)
![216 Hours](https://img.shields.io/badge/Total-216%20Hours-FF6F00?style=flat)
![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat&logo=python&logoColor=white)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-238636?style=flat&logo=git&logoColor=white)](CONTRIBUTING.md)

<div align="center">

[![🇪🇸 Versión en Español](https://img.shields.io/badge/🇪🇸_Español-Versión-red?style=for-the-badge)](README.md)

</div>

---

## 📋 Description

Intensive **36-week (9-month)** bootcamp designed to take students from zero to Junior AI/ML Developer. Covers everything from Python fundamentals to Large Language Models (LLMs) and model deployment in production.

### 🎯 Objectives

Upon completing the bootcamp, students will be able to:

- ✅ Master Python for data science and AI
- ✅ Understand mathematical foundations (linear algebra, statistics, calculus)
- ✅ Manipulate and visualize data with NumPy, Pandas, Matplotlib
- ✅ Implement Machine Learning algorithms with Scikit-learn
- ✅ Build neural networks with TensorFlow/PyTorch
- ✅ Develop Deep Learning models (CNNs, RNNs, Transformers)
- ✅ Work with NLP and LLMs using Hugging Face
- ✅ Deploy models in production (basic MLOps)

### 🚀 Why This Bootcamp?

> **Zero to Hero** - A structured path from fundamentals to advanced AI applications.

This bootcamp focuses on hands-on learning with real-world projects. Each week includes theory, guided exercises, and an integrative project that consolidates acquired knowledge.

---

## 🗓️ Bootcamp Structure

| Module               | Weeks | Hours | Content                                           |
| -------------------- | ----- | ----- | ------------------------------------------------- |
| **Fundamentals**     | 1-8   | 48h   | Python, Mathematics, NumPy, Pandas, Visualization |
| **Machine Learning** | 9-18  | 60h   | Scikit-learn, ML Algorithms, Feature Engineering  |
| **Deep Learning**    | 19-28 | 60h   | TensorFlow, PyTorch, CNNs, RNNs, Transformers     |
| **Specialization**   | 29-34 | 36h   | NLP, LLMs, Computer Vision, MLOps                 |
| **Final Project**    | 35-36 | 12h   | End-to-end project in production                  |

**Total: 36 weeks | 216 hours of intensive training**

---

## 📚 Weekly Content

Each week includes:

```
bootcamp/week-XX/
├── README.md                 # Description and objectives
├── rubrica-evaluacion.md     # Evaluation criteria
├── 0-assets/                 # Images and diagrams
├── 1-teoria/                 # Theoretical material
├── 2-practicas/              # Guided exercises
├── 3-proyecto/               # Weekly project
├── 4-recursos/               # Additional resources
│   ├── ebooks-free/
│   ├── videografia/
│   └── webgrafia/
└── 5-glosario/               # Key terms
```

### 🔑 Key Components

- 📖 **Theory**: Fundamental concepts with real-world examples
- 💻 **Practice**: Progressive exercises and hands-on projects
- 📝 **Evaluation**: Evidence of knowledge, performance, and product
- 🎓 **Resources**: Glossaries, references, and supplementary material

---

## 🛠️ Tech Stack

| Technology         | Version | Use                       |
| ------------------ | ------- | ------------------------- |
| Python             | 3.11+   | Main language             |
| NumPy              | 1.26+   | Numerical computing       |
| Pandas             | 2.0+    | Data manipulation         |
| Matplotlib/Seaborn | Latest  | Visualization             |
| Scikit-learn       | 1.4+    | Machine Learning          |
| TensorFlow         | 2.15+   | Deep Learning             |
| PyTorch            | 2.1+    | Deep Learning             |
| Hugging Face       | Latest  | NLP and LLMs              |
| Docker             | Latest  | Reproducible environments |
| pytest             | 8+      | Testing                   |

**Environment managers**: `venv`, `conda`, or `poetry`

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git for version control
- Docker (recommended for reproducible environments)
- VS Code (recommended) with included extensions

### 1. Clone the Repository

```bash
git clone https://github.com/epti-dev/bc-aidev.git
cd bc-aidev
```

### 2. Configure Environment

**Option A: venv + pip (simple)**

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

**Option B: conda (recommended for Deep Learning)**

```bash
conda create -n ai-bootcamp python=3.11
conda activate ai-bootcamp
pip install -r requirements.txt
```

**Option C: Docker (reproducible environment)**

```bash
docker compose up --build
```

### 3. Install VS Code Extensions

```bash
# Open in VS Code
code .
# Recommended extensions will appear automatically
```

### 4. Navigate to Current Week

```bash
cd bootcamp/week-01
```

### 5. Follow Instructions

Each week contains a `README.md` with detailed instructions.

---

## 📊 Learning Methodology

### Teaching Strategies

- 🎯 **Project-Based Learning (PBL)**
- 🧩 **Deliberate Practice**
- 🏆 **Kaggle Challenges**
- 👥 **Peer Code Review**
- 📄 **Paper Reading**

### Time Distribution (6h/week)

- **Theory**: 1.5 hours
- **Practice**: 2.5 hours
- **Project**: 2 hours

### Evaluation

Each week includes three types of evidence:

1. **Knowledge 🧠** (30%): Quizzes and theoretical assessments
2. **Performance 💪** (40%): Completed practical exercises
3. **Product 📦** (30%): Evaluable deliverables (functional projects)

**Passing criteria**: Minimum 70% in each type of evidence

---

## 🤝 Contributing

Contributions are welcome! This is an open-source educational project.

### How to Contribute

1. Read the [Contribution Guide](CONTRIBUTING.md)
2. Review the [Code of Conduct](CODE_OF_CONDUCT.md)
3. Fork the repository
4. Create your branch (`git checkout -b feat/new-feature`)
5. Commit with [Conventional Commits](https://www.conventionalcommits.org/) (`git commit -m 'feat: add new exercise'`)
6. Push to branch (`git push origin feat/new-feature`)
7. Open a Pull Request

### 📋 Contribution Areas

- ✨ Additional exercises
- 📚 Documentation improvements
- 🐛 Bug fixes
- 🎨 Visual resources (SVG diagrams)
- 🌐 Translations
- 📹 Video tutorials

---

## 📞 Support

- 💬 **Discussions**: [GitHub Discussions](https://github.com/epti-dev/bc-aidev/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/epti-dev/bc-aidev/issues)

---

## 📄 License

This project is under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🏆 Acknowledgments

- [Python Documentation](https://docs.python.org/3/) - Official documentation
- [Scikit-learn](https://scikit-learn.org/) - For excellent ML tutorials
- [TensorFlow](https://www.tensorflow.org/) - For educational resources
- [PyTorch](https://pytorch.org/) - For clear documentation
- [Hugging Face](https://huggingface.co/) - For democratizing LLMs
- [Kaggle](https://www.kaggle.com/) - For datasets and competitions
- AI/ML Community - For resources and examples
- All contributors

---

## 📚 Additional Documentation

- [🤖 Copilot Instructions](.github/copilot-instructions.md)
- [🤝 Contribution Guide](CONTRIBUTING.md)
- [📜 Code of Conduct](CODE_OF_CONDUCT.md)
- [🔒 Security Policy](SECURITY.md)

---

<div align="center">

**🎓 Artificial Intelligence Bootcamp: Zero to Hero**

_From zero to Junior AI/ML Developer in 9 months_

[Start Week 1](bootcamp/week-01) • [View Documentation](_docs) • [Report Issue](https://github.com/epti-dev/bc-aidev/issues) • [Contribute](CONTRIBUTING.md)

---

Made with ❤️ for the developer community

</div>
