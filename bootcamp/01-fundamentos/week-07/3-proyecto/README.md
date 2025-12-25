# 🎯 Proyecto: Analizador de Imágenes con NumPy

## 📋 Descripción

En este proyecto construirás un **analizador de imágenes** usando NumPy. Aprenderás que las imágenes digitales son simplemente arrays multidimensionales y aplicarás operaciones NumPy para procesarlas.

### ¿Qué construirás?

Un sistema que puede:

- Cargar y representar imágenes como arrays NumPy
- Calcular estadísticas de brillo y contraste
- Aplicar transformaciones (rotación, volteo, crop)
- Modificar brillo y contraste
- Convertir a escala de grises
- Aplicar filtros básicos

---

## 🎯 Objetivos de Aprendizaje

Al completar este proyecto serás capaz de:

- ✅ Entender imágenes como arrays 3D (alto × ancho × canales)
- ✅ Aplicar indexing y slicing para manipular regiones
- ✅ Usar operaciones vectorizadas para transformaciones
- ✅ Calcular estadísticas por canal de color
- ✅ Implementar filtros usando broadcasting

---

## 📚 Conceptos Clave

### Representación de Imágenes

```
Imagen RGB = Array 3D (height, width, channels)

┌─────────────────────────────────────┐
│  Canal R (Rojo)    [0-255]          │
│  Canal G (Verde)   [0-255]          │
│  Canal B (Azul)    [0-255]          │
└─────────────────────────────────────┘

Ejemplo: imagen 100x100 RGB
- Shape: (100, 100, 3)
- dtype: uint8 (0-255)
- Tamaño: 100 × 100 × 3 = 30,000 valores
```

### Píxel

Un píxel es un array de 3 valores [R, G, B]:

- `[255, 0, 0]` → Rojo puro
- `[0, 255, 0]` → Verde puro
- `[0, 0, 255]` → Azul puro
- `[255, 255, 255]` → Blanco
- `[0, 0, 0]` → Negro
- `[128, 128, 128]` → Gris

---

## 🗂️ Estructura del Proyecto

```
3-proyecto/
├── README.md              # Este archivo
├── 0-assets/
│   └── sample_image.npy   # Imagen de ejemplo
└── starter/
    └── main.py            # Código a completar
```

---

## 📝 Instrucciones

### 1. Preparación

```bash
cd bootcamp/week-07/3-proyecto/starter
```

### 2. Completar las funciones

Abre `starter/main.py` y completa las funciones marcadas con `TODO`.

### 3. Ejecutar y probar

```bash
python main.py
```

---

## 🔧 Funciones a Implementar

### Nivel Básico

| Función                   | Descripción                                  |
| ------------------------- | -------------------------------------------- |
| `create_gradient_image()` | Crear imagen con gradiente de colores        |
| `get_image_stats()`       | Calcular estadísticas (media, std, min, max) |
| `to_grayscale()`          | Convertir RGB a escala de grises             |

### Nivel Intermedio

| Función               | Descripción                  |
| --------------------- | ---------------------------- |
| `adjust_brightness()` | Aumentar/reducir brillo      |
| `adjust_contrast()`   | Modificar contraste          |
| `crop_image()`        | Recortar región de la imagen |

### Nivel Avanzado

| Función          | Descripción                          |
| ---------------- | ------------------------------------ |
| `flip_image()`   | Voltear horizontal/vertical          |
| `rotate_90()`    | Rotar 90 grados                      |
| `apply_filter()` | Aplicar filtro de convolución simple |

---

## 💡 Pistas

### Escala de Grises

Fórmula de luminosidad (ponderada por percepción humana):

```
Gray = 0.299 × R + 0.587 × G + 0.114 × B
```

### Ajuste de Brillo

```python
# Sumar valor a todos los píxeles
bright_image = image + brightness_value
# Asegurar rango [0, 255]
bright_image = np.clip(bright_image, 0, 255)
```

### Ajuste de Contraste

```python
# factor > 1: más contraste, factor < 1: menos contraste
mean = image.mean()
contrast_image = (image - mean) * factor + mean
```

### Voltear Imagen

```python
# Voltear horizontalmente
flipped = image[:, ::-1, :]
# Voltear verticalmente
flipped = image[::-1, :, :]
```

---

## ✅ Criterios de Evaluación

| Criterio                            | Peso |
| ----------------------------------- | ---- |
| Funciones básicas implementadas     | 30%  |
| Funciones intermedias implementadas | 30%  |
| Funciones avanzadas implementadas   | 25%  |
| Código limpio y documentado         | 15%  |

### Rúbrica Detallada

- **Excelente (90-100%)**: Todas las funciones implementadas correctamente, código optimizado con operaciones vectorizadas
- **Bueno (70-89%)**: Funciones básicas e intermedias completas, código funcional
- **Suficiente (50-69%)**: Funciones básicas completas, algunas intermedias
- **Insuficiente (<50%)**: Menos de 3 funciones implementadas

---

## 🚀 Extensiones Opcionales

Si terminas antes, intenta:

1. **Detección de bordes**: Implementar filtro Sobel
2. **Histograma**: Calcular y visualizar distribución de colores
3. **Blend**: Mezclar dos imágenes con transparencia
4. **Thumbnail**: Reducir tamaño de imagen (downsampling)

---

## ⏱️ Tiempo Estimado

| Actividad             | Tiempo      |
| --------------------- | ----------- |
| Leer instrucciones    | 10 min      |
| Funciones básicas     | 30 min      |
| Funciones intermedias | 30 min      |
| Funciones avanzadas   | 30 min      |
| Testing y ajustes     | 20 min      |
| **Total**             | **2 horas** |

---

## 📚 Recursos

- [NumPy Image Basics](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Image Processing with NumPy](https://realpython.com/numpy-tutorial/#image-processing)
