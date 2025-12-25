# 🎯 Ejercicio 04: Selección de Características

## 🎯 Objetivo

Practicar métodos Filter, Wrapper y Embedded para seleccionar las características más relevantes.

## 📋 Instrucciones

En este ejercicio aprenderás a:

1. Aplicar SelectKBest con diferentes métricas
2. Usar RFE (Recursive Feature Elimination)
3. Implementar SelectFromModel con importancias de árboles
4. Comparar métodos de selección
5. Integrar selección en pipelines

## 📁 Archivos

```
ejercicio-04-feature-selection/
├── README.md          # Este archivo
└── starter/
    └── main.py        # Código para completar
```

## ⏱️ Tiempo Estimado

45 minutos

## 🚀 Pasos

### Paso 1: Crear Dataset con Features Irrelevantes

Creamos un dataset donde algunas features son ruido.

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

### Paso 2: Variance Threshold

Eliminamos features con varianza muy baja.

### Paso 3: SelectKBest (Método Filter)

Seleccionamos las K mejores features según una métrica.

### Paso 4: RFE - Recursive Feature Elimination (Método Wrapper)

Eliminamos features recursivamente basándonos en un modelo.

### Paso 5: SelectFromModel (Método Embedded)

Seleccionamos features basándonos en importancias del modelo.

### Paso 6: Comparar Métodos

Evaluamos qué método selecciona mejores features.

## ✅ Criterios de Éxito

- [ ] VarianceThreshold elimina features constantes
- [ ] SelectKBest selecciona las K features con mejores scores
- [ ] RFE identifica features importantes iterativamente
- [ ] SelectFromModel usa importancias de Random Forest
- [ ] Puedes comparar la efectividad de cada método
