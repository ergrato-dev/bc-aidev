# 🤝 Guía de Contribución

¡Gracias por tu interés en contribuir al **Bootcamp de Inteligencia Artificial: Zero to Hero**!

Este es un proyecto educativo de código abierto y todas las contribuciones son bienvenidas.

---

## 📋 Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [¿Cómo Puedo Contribuir?](#cómo-puedo-contribuir)
- [Configuración del Entorno](#configuración-del-entorno)
- [Flujo de Trabajo](#flujo-de-trabajo)
- [Guía de Estilo](#guía-de-estilo)
- [Commits Convencionales](#commits-convencionales)

---

## 📜 Código de Conducta

Este proyecto sigue un [Código de Conducta](CODE_OF_CONDUCT.md). Al participar, aceptas seguir sus términos.

---

## 🎯 ¿Cómo Puedo Contribuir?

### 🐛 Reportar Bugs

Si encuentras un error:

1. Verifica que no exista un [issue similar](https://github.com/epti-dev/bc-aidev/issues)
2. Crea un nuevo issue usando la plantilla de **Bug Report**
3. Incluye:
   - Descripción clara del problema
   - Pasos para reproducirlo
   - Comportamiento esperado vs actual
   - Capturas de pantalla si aplica
   - Entorno (OS, Python version, etc.)

### 💡 Sugerir Mejoras

Para nuevas características o mejoras:

1. Revisa los [issues existentes](https://github.com/epti-dev/bc-aidev/issues)
2. Crea un issue usando la plantilla de **Feature Request**
3. Describe claramente la mejora y su beneficio

### 📚 Contribuir Contenido

Áreas donde puedes ayudar:

| Área            | Descripción                 |
| --------------- | --------------------------- |
| ✨ Ejercicios   | Nuevos ejercicios prácticos |
| 📖 Teoría       | Mejoras en explicaciones    |
| 🎨 Diagramas    | Assets SVG educativos       |
| 🌐 Traducciones | Versiones en otros idiomas  |
| 📹 Videos       | Tutoriales complementarios  |
| 🐛 Correcciones | Errores en código o texto   |

---

## ⚙️ Configuración del Entorno

### Prerrequisitos

- Python 3.11+
- Git
- Docker (opcional, recomendado)
- VS Code (recomendado)

### Instalación

```bash
# 1. Fork del repositorio en GitHub

# 2. Clonar tu fork
git clone https://github.com/TU-USUARIO/bc-aidev.git
cd bc-aidev

# 3. Agregar upstream
git remote add upstream https://github.com/epti-dev/bc-aidev.git

# 4. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 5. Instalar dependencias de desarrollo
pip install -r requirements-dev.txt
```

### Con Docker

```bash
docker compose up --build
```

---

## 🔄 Flujo de Trabajo

### 1. Sincronizar con upstream

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

### 2. Crear rama para tu contribución

```bash
# Usar prefijos según el tipo de cambio
git checkout -b feat/nueva-funcionalidad
git checkout -b fix/corregir-error
git checkout -b docs/mejorar-documentacion
git checkout -b refactor/mejorar-codigo
```

### 3. Hacer cambios

- Sigue la [Guía de Estilo](#guía-de-estilo)
- Prueba tu código
- Actualiza documentación si es necesario

### 4. Commit con mensaje convencional

```bash
git add .
git commit -m "feat(week-05): add neural network exercise"
```

### 5. Push y Pull Request

```bash
git push origin feat/nueva-funcionalidad
```

Luego crea un Pull Request en GitHub.

---

## 🎨 Guía de Estilo

### Python

- **PEP 8** como base
- **Type hints** en funciones públicas
- **Docstrings** en formato Google
- **snake_case** para variables y funciones
- **PascalCase** para clases

```python
def train_model(X_train: np.ndarray, y_train: np.ndarray, epochs: int = 100) -> Model:
    """
    Entrena un modelo de clasificación.

    Args:
        X_train: Features de entrenamiento.
        y_train: Labels de entrenamiento.
        epochs: Número de épocas.

    Returns:
        Modelo entrenado.
    """
    pass
```

### Markdown

- Usar encabezados jerárquicos (`#`, `##`, `###`)
- Incluir tabla de contenidos en documentos largos
- Usar emojis con moderación para mejorar legibilidad
- Código con syntax highlighting

### Assets SVG

- Tema dark obligatorio
- Sin degradados
- Fuentes sans-serif
- Padding mínimo 8px en textos
- Nombrar con números: `01-diagrama.svg`

---

## 📝 Commits Convencionales

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

```
<tipo>(<alcance>): <descripción>

[cuerpo opcional]

[footer opcional]
```

### Tipos

| Tipo       | Descripción                     |
| ---------- | ------------------------------- |
| `feat`     | Nueva característica            |
| `fix`      | Corrección de bug               |
| `docs`     | Cambios en documentación        |
| `style`    | Formato (sin cambios de código) |
| `refactor` | Refactorización                 |
| `test`     | Añadir o modificar tests        |
| `chore`    | Tareas de mantenimiento         |

### Ejemplos

```bash
feat(week-03): add pandas dataframe exercises
fix(week-01): correct typo in variable names
docs(readme): update installation instructions
refactor(week-05): simplify neural network implementation
chore(deps): update tensorflow to 2.15
```

### Alcance (scope)

- `week-XX` - Cambios en semana específica
- `docs` - Documentación general
- `config` - Configuración del proyecto
- `deps` - Dependencias

---

## ✅ Checklist antes del PR

- [ ] Código sigue la guía de estilo
- [ ] Tests pasan (si aplica)
- [ ] Documentación actualizada
- [ ] Commits siguen convención
- [ ] Branch actualizada con main
- [ ] PR describe claramente los cambios

---

## 🙏 Reconocimiento

Todos los contribuidores serán reconocidos en el README del proyecto.

¡Gracias por hacer este bootcamp mejor para todos! 🚀
