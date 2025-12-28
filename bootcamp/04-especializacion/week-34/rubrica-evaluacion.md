# 📋 Rúbrica de Evaluación - Semana 34

## 🎯 MLOps: Deployment de Modelos ML

---

## 📊 Distribución de Puntos

| Componente | Peso | Puntos |
|------------|------|--------|
| 🧠 Conocimiento | 30% | 30 pts |
| 💪 Desempeño | 40% | 40 pts |
| 📦 Producto | 30% | 30 pts |
| **Total** | **100%** | **100 pts** |

---

## 🧠 Conocimiento (30 puntos)

### Conceptos MLOps (10 pts)

| Criterio | Excelente (10) | Bueno (7) | Regular (4) | Insuficiente (0) |
|----------|----------------|-----------|-------------|------------------|
| Ciclo MLOps | Explica todos los componentes y su interacción | Conoce componentes principales | Conocimiento parcial | No comprende el ciclo |

### APIs REST (10 pts)

| Criterio | Excelente (10) | Bueno (7) | Regular (4) | Insuficiente (0) |
|----------|----------------|-----------|-------------|------------------|
| HTTP/REST | Domina métodos, status codes, headers | Conoce GET/POST básico | Confunde conceptos | No entiende REST |

### Containerización (10 pts)

| Criterio | Excelente (10) | Bueno (7) | Regular (4) | Insuficiente (0) |
|----------|----------------|-----------|-------------|------------------|
| Docker | Entiende imágenes, containers, layers, volumes | Conoce comandos básicos | Uso limitado | No comprende Docker |

---

## 💪 Desempeño (40 puntos)

### Ejercicio 1: FastAPI Básico (15 pts)

| Criterio | Excelente (15) | Bueno (11) | Regular (7) | Insuficiente (0) |
|----------|----------------|------------|-------------|------------------|
| Implementación | API funcional con validación Pydantic, docs automáticos | Endpoints funcionan correctamente | Endpoints básicos con errores | No implementa |

### Ejercicio 2: Docker ML (15 pts)

| Criterio | Excelente (15) | Bueno (11) | Regular (7) | Insuficiente (0) |
|----------|----------------|------------|-------------|------------------|
| Containerización | Dockerfile optimizado, multi-stage, .dockerignore | Dockerfile funcional | Container construye con warnings | No funciona |

### Ejercicio 3: Monitoring (10 pts)

| Criterio | Excelente (10) | Bueno (7) | Regular (4) | Insuficiente (0) |
|----------|----------------|-----------|-------------|------------------|
| Métricas | Implementa métricas custom, histogramas, counters | Métricas básicas funcionando | Solo health check | Sin monitoring |

---

## 📦 Producto (30 puntos)

### API ML Producción

| Criterio | Excelente (10) | Bueno (7) | Regular (4) | Insuficiente (0) |
|----------|----------------|-----------|-------------|------------------|
| **Funcionalidad** | API completa con todos los endpoints documentados | Endpoints principales funcionan | Funcionalidad parcial | No funciona |
| **Docker** | Compose con múltiples servicios, volúmenes, networks | Docker compose básico funcional | Solo Dockerfile | Sin containerización |
| **Calidad** | Código limpio, tipado, manejo errores, logging | Código organizado | Código funcional desordenado | Código no funcional |

---

## 🎯 Criterios de Aprobación

- **Mínimo para aprobar**: 70 puntos
- **Todos los ejercicios** deben estar completados
- **El proyecto** debe ejecutarse sin errores críticos

---

## 📝 Checklist de Entrega

### Ejercicios
- [ ] API FastAPI responde en `/docs`
- [ ] Dockerfile construye correctamente
- [ ] Métricas expuestas en `/metrics`

### Proyecto
- [ ] `docker compose up` levanta todos los servicios
- [ ] Endpoint `/predict` funciona correctamente
- [ ] Documentación de API disponible
- [ ] Health check implementado

---

## 🏆 Niveles de Logro

| Rango | Nivel | Descripción |
|-------|-------|-------------|
| 90-100 | ⭐ Excepcional | Dominio completo de MLOps básico |
| 80-89 | 🌟 Sobresaliente | Muy buen manejo de herramientas |
| 70-79 | ✅ Aprobado | Cumple con los objetivos mínimos |
| 60-69 | ⚠️ En desarrollo | Necesita reforzar conceptos |
| <60 | ❌ No aprobado | Debe repetir la semana |

---

## 📚 Recursos de Apoyo

Si tienes dificultades:

1. Revisa la [documentación de FastAPI](https://fastapi.tiangolo.com/)
2. Consulta [Docker Docs](https://docs.docker.com/)
3. Practica con ejemplos más simples
4. Pide ayuda en las discusiones del curso

---

_Rúbrica Semana 34 | Módulo: Especialización_
