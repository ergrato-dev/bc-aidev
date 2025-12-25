# 🔧 Ejercicio 03: Manejo de Missing Data

## 🎯 Objetivo

Practicar estrategias de imputación para valores faltantes usando SimpleImputer, KNNImputer y técnicas avanzadas.

## 📋 Instrucciones

En este ejercicio aprenderás a:

1. Diagnosticar patrones de missing data
2. Aplicar SimpleImputer con diferentes estrategias
3. Usar KNNImputer para imputación basada en vecinos
4. Crear indicadores de missing como features
5. Manejar missing en pipelines

## 📁 Archivos

```
ejercicio-03-missing-data/
├── README.md          # Este archivo
└── starter/
    └── main.py        # Código para completar
```

## ⏱️ Tiempo Estimado

45 minutos

## 🚀 Pasos

### Paso 1: Crear Dataset con Missing Values

Simulamos un dataset realista con diferentes patrones de missing data.

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

### Paso 2: Diagnóstico de Missing Data

Analizamos cuántos valores faltan y sus patrones.

### Paso 3: SimpleImputer - Estrategias Básicas

Aplicamos imputación con media, mediana y moda.

### Paso 4: KNNImputer

Usamos vecinos cercanos para imputar valores más precisos.

### Paso 5: Missing Indicator

Creamos features que indican si un valor estaba faltante.

### Paso 6: Pipeline Completo con Imputación

Integramos imputación en un pipeline de preprocesamiento.

## ✅ Criterios de Éxito

- [ ] Identificas correctamente el porcentaje de missing
- [ ] SimpleImputer elimina todos los NaN
- [ ] KNNImputer produce valores más realistas que la media
- [ ] El Missing Indicator crea columnas binarias
- [ ] El pipeline maneja datos nuevos sin errores
