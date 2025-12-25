# 🏷️ Ejercicio 02: Codificación de Categóricas

## 🎯 Objetivo

Practicar OneHotEncoder, OrdinalEncoder y TargetEncoder para convertir variables categóricas en numéricas.

## 📋 Instrucciones

En este ejercicio aprenderás a:

1. Aplicar OneHotEncoder para variables nominales
2. Usar OrdinalEncoder para variables ordinales
3. Implementar TargetEncoder para alta cardinalidad
4. Manejar categorías desconocidas en test

## 📁 Archivos

```
ejercicio-02-categoricas/
├── README.md          # Este archivo
└── starter/
    └── main.py        # Código para completar
```

## ⏱️ Tiempo Estimado

45 minutos

## 🚀 Pasos

### Paso 1: Crear Dataset de Ejemplo

Creamos un dataset con diferentes tipos de variables categóricas.

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

### Paso 2: OneHotEncoder Básico

Aplicamos one-hot encoding a variables nominales (sin orden).

### Paso 3: OneHotEncoder con drop='first'

Evitamos multicolinealidad eliminando una columna de referencia.

### Paso 4: OrdinalEncoder para Variables Ordinales

Codificamos variables con orden natural respetando la jerarquía.

### Paso 5: Manejar Categorías Desconocidas

Configuramos los encoders para manejar categorías nuevas en test.

### Paso 6: TargetEncoder para Alta Cardinalidad

Usamos mean encoding cuando hay muchas categorías únicas.

### Paso 7: Comparación pd.get_dummies vs OneHotEncoder

Entendemos cuándo usar cada aproximación.

## ✅ Criterios de Éxito

- [ ] OneHotEncoder genera columnas binarias correctamente
- [ ] OrdinalEncoder preserva el orden de las categorías
- [ ] El encoder maneja categorías desconocidas sin errores
- [ ] TargetEncoder reduce dimensionalidad en alta cardinalidad
- [ ] Entiendes la diferencia entre get_dummies y OneHotEncoder
