# 🎯 Ejercicio 02: Detección de Objetos con YOLOv8

## 🎯 Objetivo

Aprender a detectar objetos en imágenes y videos usando Ultralytics YOLOv8.

## 📋 Conceptos Clave

- **Detección de objetos**: Localizar y clasificar múltiples objetos
- **Bounding boxes**: Coordenadas [x, y, width, height] del objeto
- **Confianza**: Probabilidad de que la detección sea correcta
- **NMS**: Non-Maximum Suppression para eliminar duplicados

## ⏱️ Tiempo Estimado

60 minutos

---

## 📝 Instrucciones

### Paso 1: Instalar Ultralytics

```bash
pip install ultralytics opencv-python pillow
```

### Paso 2: Cargar Modelo YOLOv8

```python
from ultralytics import YOLO

# Cargar modelo pre-entrenado (descarga automática)
model = YOLO('yolov8n.pt')  # nano - más rápido

# Variantes disponibles:
# yolov8n.pt - Nano (3.2M params, más rápido)
# yolov8s.pt - Small (11.2M params)
# yolov8m.pt - Medium (25.9M params)
# yolov8l.pt - Large (43.7M params)
# yolov8x.pt - Extra Large (68.2M params, más preciso)

print(model.info())  # Ver información del modelo
```

**Abre `starter/main.py`** y descomenta la sección correspondiente.

### Paso 3: Detectar en una Imagen

```python
# Detectar objetos
results = model('https://ultralytics.com/images/bus.jpg')

# Acceder a resultados
for result in results:
    boxes = result.boxes  # Detecciones
    print(f"Objetos detectados: {len(boxes)}")
    
    for box in boxes:
        # Coordenadas [x1, y1, x2, y2]
        coords = box.xyxy[0].tolist()
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        
        print(f"  {class_name}: {confidence:.2%} @ {coords}")
```

### Paso 4: Visualizar Resultados

```python
# Mostrar imagen con detecciones
result = results[0]
annotated_img = result.plot()  # Imagen con bboxes dibujados

# Guardar resultado
result.save(filename='result.jpg')

# Mostrar con matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.imshow(annotated_img[..., ::-1])  # BGR -> RGB
plt.axis('off')
plt.savefig('detection_result.png')
plt.show()
```

### Paso 5: Configurar Parámetros de Detección

```python
# Detección con parámetros personalizados
results = model.predict(
    source='image.jpg',
    conf=0.25,        # Umbral de confianza mínimo
    iou=0.45,         # Umbral IoU para NMS
    max_det=100,      # Máximo detecciones
    classes=[0, 2],   # Solo personas (0) y coches (2)
    verbose=False     # Sin logs
)
```

### Paso 6: Detección en Video

```python
# Procesar video
results = model('video.mp4', stream=True)

for frame_results in results:
    # Procesar cada frame
    annotated_frame = frame_results.plot()
    # Aquí podrías mostrar o guardar el frame
```

### Paso 7: Detección en Tiempo Real (Webcam)

```python
import cv2

# Abrir webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Detectar
    results = model(frame, verbose=False)
    
    # Visualizar
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🔍 Análisis de Resultados

### Estructura de una Detección

```python
box = result.boxes[0]

# Coordenadas
box.xyxy    # [x1, y1, x2, y2] esquinas
box.xywh    # [x_center, y_center, width, height]
box.xyxyn   # Normalizado [0-1]

# Metadata
box.conf    # Confianza [0-1]
box.cls     # ID de clase
box.id      # ID de tracking (si está activo)
```

### Clases COCO (80 clases)

Las más comunes:
- 0: person
- 1: bicycle
- 2: car
- 3: motorcycle
- 5: bus
- 7: truck
- 16: dog
- 17: cat

---

## ✅ Checklist de Verificación

- [ ] YOLOv8 instalado y modelo cargado
- [ ] Detección ejecutada en imagen de ejemplo
- [ ] Resultados visualizados con bounding boxes
- [ ] Parámetros de detección ajustados (conf, iou)
- [ ] Probado con al menos 3 imágenes diferentes

---

## 📚 Recursos

- [Ultralytics Docs](https://docs.ultralytics.com/)
- [YOLOv8 Detection](https://docs.ultralytics.com/tasks/detect/)
- [COCO Dataset Classes](https://cocodataset.org/#explore)
