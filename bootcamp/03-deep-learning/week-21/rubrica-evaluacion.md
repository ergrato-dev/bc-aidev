# 📋 Rúbrica de Evaluación - Semana 21

## 🔥 PyTorch Fundamentals

---

## 📊 Distribución de Puntos

| Tipo de Evidencia | Porcentaje | Puntos |
| ----------------- | ---------- | ------ |
| 🧠 Conocimiento   | 30%        | 30     |
| 💪 Desempeño      | 35%        | 35     |
| 📦 Producto       | 35%        | 35     |
| **Total**         | **100%**   | **100** |

---

## 🧠 Evidencia de Conocimiento (30 puntos)

### Cuestionario Teórico

| Criterio                                  | Excelente (10) | Bueno (7) | Suficiente (5) | Insuficiente (0-4) |
| ----------------------------------------- | -------------- | --------- | -------------- | ------------------ |
| Comprensión de tensores PyTorch           | Domina creación, operaciones, shapes, dtype y device | Entiende operaciones básicas | Conocimiento superficial | No comprende tensores |
| Entendimiento de autograd                 | Explica grafo computacional, requires_grad, backward | Usa autograd correctamente | Conocimiento básico | No entiende autograd |
| Arquitectura nn.Module                    | Domina herencia, forward, parámetros, módulos anidados | Implementa redes simples | Conocimiento limitado | No comprende nn.Module |

**Puntuación Conocimiento: ___ / 30**

---

## 💪 Evidencia de Desempeño (35 puntos)

### Ejercicio 1: Tensores PyTorch (10 puntos)

| Criterio                        | Completo (10) | Parcial (5-9) | Mínimo (1-4) | No realizado (0) |
| ------------------------------- | ------------- | ------------- | ------------ | ---------------- |
| Creación de tensores            | Múltiples métodos dominados | Métodos básicos | Solo torch.tensor() | No realizado |
| Operaciones matemáticas         | Broadcasting, indexing avanzado | Operaciones básicas | Suma y resta | No realizado |
| Conversión NumPy ↔ PyTorch      | Ambas direcciones, entiende memoria compartida | Una dirección | Con errores | No realizado |
| Manejo de dispositivos (CPU/GPU)| .to(), .cuda(), .cpu() dominados | Uso básico | Con ayuda | No realizado |

### Ejercicio 2: Autograd y Gradientes (12 puntos)

| Criterio                        | Completo (12) | Parcial (6-11) | Mínimo (1-5) | No realizado (0) |
| ------------------------------- | ------------- | -------------- | ------------ | ---------------- |
| requires_grad configuración     | Entiende cuándo y por qué | Uso correcto básico | Con errores | No realizado |
| Cálculo de gradientes           | backward() con escalares y tensores | backward() básico | Con ayuda | No realizado |
| Grafo computacional             | Entiende construcción dinámica | Uso básico | Confusión | No comprende |
| torch.no_grad() y detach()      | Uso apropiado para inferencia | Conoce uno | Con errores | No conoce |

### Ejercicio 3: Red Neuronal Manual (13 puntos)

| Criterio                        | Completo (13) | Parcial (7-12) | Mínimo (1-6) | No realizado (0) |
| ------------------------------- | ------------- | -------------- | ------------ | ---------------- |
| Clase nn.Module correcta        | __init__ y forward impecables | Funciona con errores menores | Estructura incorrecta | No realizado |
| Capas definidas correctamente   | nn.Linear, activaciones apropiadas | Capas básicas | Errores en dimensiones | No realizado |
| Training loop completo          | forward, loss, backward, step, zero_grad | Loop básico funcional | Incompleto | No realizado |
| Evaluación del modelo           | model.eval(), torch.no_grad() | Evaluación básica | Con errores | No realizado |

**Puntuación Desempeño: ___ / 35**

---

## 📦 Evidencia de Producto (35 puntos)

### Proyecto: Clasificador Fashion-MNIST

#### Funcionalidad (15 puntos)

| Criterio                          | Excelente (15) | Bueno (10-14) | Suficiente (5-9) | Insuficiente (0-4) |
| --------------------------------- | -------------- | ------------- | ---------------- | ------------------ |
| Accuracy en test set              | ≥90%           | 88-89%        | 85-87%           | <85%               |
| Carga de datos con DataLoader     | Correcta con batches y shuffle | Funciona básico | Con errores | No implementado |
| Training loop completo            | Epochs, batches, métricas por epoch | Loop funcional | Parcial | No funciona |
| Guardado y carga de modelo        | torch.save/load state_dict | Guarda modelo | Con errores | No implementado |

#### Calidad del Código (10 puntos)

| Criterio                          | Excelente (10) | Bueno (7-9) | Suficiente (4-6) | Insuficiente (0-3) |
| --------------------------------- | -------------- | ----------- | ---------------- | ------------------ |
| Organización y estructura         | Código modular, funciones claras | Estructura aceptable | Código desordenado | Difícil de leer |
| Documentación                     | Docstrings, comentarios útiles | Documentación básica | Comentarios escasos | Sin documentación |
| Manejo de dispositivos            | CPU/GPU automático con device | Hardcoded pero funciona | Errores de device | No considera GPU |
| Uso idiomático de PyTorch         | Patrones y convenciones correctas | Mayormente correcto | Algunos antipatterns | Código no pythónico |

#### Visualización y Análisis (10 puntos)

| Criterio                          | Excelente (10) | Bueno (7-9) | Suficiente (4-6) | Insuficiente (0-3) |
| --------------------------------- | -------------- | ----------- | ---------------- | ------------------ |
| Gráfica de pérdida                | Train y validation loss por epoch | Solo train loss | Gráfica básica | Sin gráfica |
| Gráfica de accuracy               | Train y validation accuracy | Solo una métrica | Gráfica básica | Sin gráfica |
| Matriz de confusión               | Implementada con análisis | Implementada | Parcial | No implementada |
| Ejemplos de predicciones          | Muestra imágenes con predicciones | Algunas predicciones | Básico | No muestra |

**Puntuación Producto: ___ / 35**

---

## 📈 Resumen de Puntuación

| Sección            | Puntos Obtenidos | Puntos Máximos |
| ------------------ | ---------------- | -------------- |
| 🧠 Conocimiento    |                  | 30             |
| 💪 Desempeño       |                  | 35             |
| 📦 Producto        |                  | 35             |
| **Total**          |                  | **100**        |

---

## 🎯 Escala de Calificación

| Rango     | Calificación | Descripción                              |
| --------- | ------------ | ---------------------------------------- |
| 90-100    | A            | Excelente dominio de PyTorch             |
| 80-89     | B            | Buen manejo, errores menores             |
| 70-79     | C            | Competencia básica alcanzada             |
| 60-69     | D            | Necesita práctica adicional              |
| <60       | F            | No alcanza competencias mínimas          |

---

## 📝 Retroalimentación

### Fortalezas:
_[Espacio para comentarios positivos]_

### Áreas de Mejora:
_[Espacio para sugerencias específicas]_

### Recursos Recomendados:
_[Links a tutoriales o documentación relevante]_

---

_Bootcamp IA: Zero to Hero | Módulo 3: Deep Learning | Semana 21_
