# 🎭 Ejercicio 03: Segmentación de Imágenes

## 🎯 Objetivo

Aprender a realizar segmentación de imágenes usando YOLOv8-seg y Segment Anything Model (SAM).

## 📋 Conceptos Clave

- **Segmentación semántica**: Clasificar cada píxel por categoría
- **Segmentación de instancias**: Separar objetos individuales
- **Máscara binaria**: Imagen donde 1=objeto, 0=fondo
- **Polígonos**: Contornos que delimitan objetos

## ⏱️ Tiempo Estimado

60 minutos

---

## 📝 Instrucciones

### Paso 1: Instalar Dependencias

```bash
pip install ultralytics opencv-python pillow matplotlib
```

### Paso 2: Cargar Modelo de Segmentación

```python
from ultralytics import YOLO

# Cargar YOLOv8-seg (segmentación)
model = YOLO('yolov8n-seg.pt')  # nano con segmentación

# Variantes:
# yolov8n-seg.pt - Nano
# yolov8s-seg.pt - Small
# yolov8m-seg.pt - Medium
# yolov8l-seg.pt - Large
# yolov8x-seg.pt - Extra Large

print(f"Modelo: {model.task}")  # 'segment'
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

### Paso 3: Ejecutar Segmentación

```python
# Segmentar imagen
results = model('https://ultralytics.com/images/bus.jpg')

# Acceder a máscaras
result = results[0]
masks = result.masks  # Máscaras de segmentación

print(f"Objetos segmentados: {len(masks)}")

# Cada máscara tiene:
# - masks.data: Tensor de máscaras binarias
# - masks.xy: Polígonos (coordenadas de contorno)
```

### Paso 4: Visualizar Máscaras

```python
import matplotlib.pyplot as plt
import numpy as np

# Imagen anotada con máscaras
annotated = result.plot()

plt.figure(figsize=(12, 8))
plt.imshow(annotated[..., ::-1])
plt.axis('off')
plt.title('Segmentación YOLOv8')
plt.savefig('segmentation_result.png')
plt.show()
```

### Paso 5: Extraer Máscaras Individuales

```python
# Obtener máscara de un objeto específico
for i, (mask, box) in enumerate(zip(result.masks.data, result.boxes)):
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = box.conf[0].item()
    
    # Máscara binaria (H x W)
    mask_binary = mask.cpu().numpy()
    
    print(f"Objeto {i}: {class_name} ({confidence:.2%})")
    print(f"  Máscara shape: {mask_binary.shape}")
    print(f"  Píxeles objeto: {mask_binary.sum():.0f}")
```

### Paso 6: Crear Máscara Combinada

```python
import numpy as np

# Combinar todas las máscaras en una sola
combined_mask = np.zeros(result.orig_shape[:2], dtype=np.uint8)

for i, (mask, box) in enumerate(zip(result.masks.data, result.boxes)):
    mask_np = mask.cpu().numpy().astype(np.uint8)
    # Asignar ID único a cada objeto
    combined_mask[mask_np > 0] = i + 1

# Visualizar
plt.figure(figsize=(10, 8))
plt.imshow(combined_mask, cmap='tab20')
plt.colorbar(label='Object ID')
plt.title('Máscaras combinadas')
plt.show()
```

### Paso 7: Aplicar Máscara a Imagen

```python
import cv2

# Cargar imagen original
original = result.orig_img.copy()

# Extraer solo el primer objeto
if len(result.masks) > 0:
    mask = result.masks.data[0].cpu().numpy()
    
    # Redimensionar máscara si es necesario
    if mask.shape != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
    
    # Crear imagen con solo el objeto
    mask_3d = np.stack([mask] * 3, axis=-1)
    extracted = (original * mask_3d).astype(np.uint8)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(original[..., ::-1])
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(extracted[..., ::-1])
    plt.title('Objeto extraído')
    plt.axis('off')
    
    plt.show()
```

### Paso 8: Usar Segment Anything (SAM)

```python
from ultralytics import SAM

# Cargar SAM
sam_model = SAM('sam_b.pt')  # base model

# Segmentar todo
results = sam_model('image.jpg')

# Segmentar con prompts
results = sam_model(
    'image.jpg',
    points=[[500, 375]],    # Punto dentro del objeto
    labels=[1]               # 1=foreground, 0=background
)

# Segmentar con bounding box
results = sam_model(
    'image.jpg',
    bboxes=[[100, 100, 400, 400]]  # [x1, y1, x2, y2]
)
```

---

## 🔍 Análisis

### Diferencia YOLO-seg vs SAM

| Aspecto | YOLOv8-seg | SAM |
|---------|------------|-----|
| Velocidad | Rápido | Lento |
| Clases | 80 (COCO) | Sin clases |
| Uso | Detección + segmentación | Segmentación interactiva |
| Prompts | No | Puntos, cajas, texto |

### Métricas de Segmentación

- **IoU** (Intersection over Union): Superposición máscara predicha vs real
- **Dice**: 2 × Intersección / (Pred + GT)
- **mIoU**: Promedio IoU por clase

---

## ✅ Checklist de Verificación

- [ ] Modelo YOLOv8-seg cargado correctamente
- [ ] Segmentación ejecutada en imagen de ejemplo
- [ ] Máscaras individuales extraídas
- [ ] Visualización con máscaras coloreadas
- [ ] Objeto extraído usando máscara

---

## 📚 Recursos

- [YOLOv8 Segmentation](https://docs.ultralytics.com/tasks/segment/)
- [Segment Anything Model](https://segment-anything.com/)
- [Ultralytics SAM](https://docs.ultralytics.com/models/sam/)
