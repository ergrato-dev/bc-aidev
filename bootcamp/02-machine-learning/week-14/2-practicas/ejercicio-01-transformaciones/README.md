# 📊 Ejercicio 01: Transformaciones Numéricas

## 🎯 Objetivo

Practicar el uso de StandardScaler, MinMaxScaler, RobustScaler y PowerTransformer para escalar y transformar variables numéricas.

## 📋 Instrucciones

En este ejercicio aprenderás a:

1. Comparar diferentes escaladores
2. Manejar outliers con RobustScaler
3. Normalizar distribuciones sesgadas
4. Visualizar el efecto de cada transformación

## 📁 Archivos

```
ejercicio-01-transformaciones/
├── README.md          # Este archivo
└── starter/
    └── main.py        # Código para completar
```

## ⏱️ Tiempo Estimado

30 minutos

## 🚀 Pasos

### Paso 1: Crear Datos de Ejemplo

Creamos un dataset con diferentes características para ver cómo afectan los escaladores.

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

### Paso 2: Aplicar StandardScaler

El StandardScaler centra los datos con media 0 y desviación estándar 1.

### Paso 3: Aplicar MinMaxScaler

El MinMaxScaler escala los datos al rango [0, 1].

### Paso 4: Aplicar RobustScaler

El RobustScaler usa la mediana y el IQR, siendo robusto a outliers.

### Paso 5: Comparar Escaladores Visualmente

Visualizamos las distribuciones antes y después de escalar.

### Paso 6: PowerTransformer para Distribuciones Sesgadas

Aplicamos Box-Cox o Yeo-Johnson para normalizar distribuciones.

### Paso 7: Principio Fit on Train

Demostramos por qué es crucial ajustar solo en datos de entrenamiento.

## ✅ Criterios de Éxito

- [ ] Los datos escalados con StandardScaler tienen media ≈ 0 y std ≈ 1
- [ ] Los datos con MinMaxScaler están en rango [0, 1]
- [ ] RobustScaler maneja mejor los outliers
- [ ] PowerTransformer reduce el sesgo de la distribución
- [ ] Aplicas fit solo en train y transform en ambos
