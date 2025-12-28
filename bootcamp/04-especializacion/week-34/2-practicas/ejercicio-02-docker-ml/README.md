# 🐳 Ejercicio 02: Docker para ML

## 🎯 Objetivo

Containerizar una aplicación ML con Docker y Docker Compose.

---

## 📋 Descripción

En este ejercicio aprenderás a:

1. Crear un Dockerfile optimizado para ML
2. Usar Docker Compose para orquestar servicios
3. Configurar volúmenes y variables de entorno
4. Implementar health checks en Docker

---

## ⏱️ Duración

**50 minutos**

---

## 📁 Estructura

```
ejercicio-02-docker-ml/
├── README.md
└── starter/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py          # API FastAPI
    │   ├── model.py         # Modelo ML
    │   └── schemas.py       # Pydantic schemas
    ├── models/
    │   └── .gitkeep         # Carpeta para modelo
    ├── Dockerfile           # Para completar
    ├── docker-compose.yml   # Para completar
    ├── .dockerignore        # Archivos a ignorar
    └── requirements.txt     # Dependencias
```

---

## 🔧 Requisitos Previos

- Docker instalado
- Docker Compose instalado
- Ejercicio 01 completado (o usar el código proporcionado)

---

## 📝 Instrucciones

### Paso 1: Revisar la Aplicación

La carpeta `app/` contiene una API FastAPI lista. Revisa los archivos para entender la estructura.

### Paso 2: Completar el Dockerfile

**Abre `starter/Dockerfile`** y descomenta las secciones indicadas:

1. Imagen base
2. Variables de entorno
3. Directorio de trabajo
4. Instalación de dependencias
5. Copia de aplicación
6. Health check
7. Comando de ejecución

### Paso 3: Completar docker-compose.yml

**Abre `starter/docker-compose.yml`** y descomenta:

1. Servicio de la API
2. Configuración de puertos
3. Variables de entorno
4. Volúmenes
5. Health check

### Paso 4: Construir la Imagen

```bash
# Construir imagen
docker build -t ml-api:1.0.0 .

# Ver imagen creada
docker images | grep ml-api
```

### Paso 5: Ejecutar con Docker

```bash
# Ejecutar contenedor
docker run -d -p 8000:8000 --name ml-api ml-api:1.0.0

# Ver logs
docker logs ml-api

# Probar API
curl http://localhost:8000/health

# Detener
docker stop ml-api && docker rm ml-api
```

### Paso 6: Ejecutar con Docker Compose

```bash
# Iniciar servicios
docker compose up --build

# En otra terminal, probar
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Detener
docker compose down
```

---

## ✅ Criterios de Éxito

- [ ] Dockerfile construye sin errores
- [ ] Imagen tiene tamaño razonable (< 500MB)
- [ ] Contenedor inicia correctamente
- [ ] Health check funciona
- [ ] API responde en puerto 8000
- [ ] Docker Compose orquesta el servicio

---

## 🔗 Recursos

- [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
