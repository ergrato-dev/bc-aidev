# 📖 Glosario - Semana 33: Computer Vision

Términos clave de Computer Vision, ordenados alfabéticamente.

---

## A

### Anchor Box
Cajas de referencia predefinidas con diferentes aspectos y escalas que el modelo usa como base para predecir bounding boxes. YOLO genera predicciones ajustando estas anclas.

### Annotation
Proceso de etiquetar datos para entrenamiento. En detección incluye bounding boxes y clases; en segmentación, máscaras pixel a pixel.

### Augmentation (Data Augmentation)
Técnicas para aumentar artificialmente el dataset aplicando transformaciones (rotación, flip, cambio de brillo, etc.) a las imágenes originales.

---

## B

### Backbone
Red neuronal que extrae características de la imagen. En YOLO, típicamente CSPDarknet. Procesa la imagen y genera mapas de características.

```
Input Image → [BACKBONE] → Feature Maps
```

### Bounding Box (BBox)
Rectángulo que enmarca un objeto en una imagen. Se define con coordenadas:
- **xyxy**: [x1, y1, x2, y2] (esquinas)
- **xywh**: [x_center, y_center, width, height]

---

## C

### Class (Clase)
Categoría o etiqueta que identifica el tipo de objeto detectado (person, car, dog, etc.). COCO tiene 80 clases; ImageNet tiene 1000.

### Confidence Score
Probabilidad de que una detección sea correcta, típicamente entre 0 y 1. Combina la probabilidad de que exista un objeto y la precisión de la clasificación.

### Convolutional Neural Network (CNN)
Red neuronal especializada en procesar imágenes usando operaciones de convolución que detectan patrones locales (bordes, texturas, formas).

### COCO Dataset
Common Objects in Context. Dataset estándar para detección y segmentación con 80 clases de objetos cotidianos y 330K+ imágenes.

---

## D

### Detection
Tarea de localizar y clasificar objetos en una imagen, produciendo bounding boxes con etiquetas y scores de confianza.

### Dice Coefficient
Métrica de similitud para segmentación:

$$Dice = \frac{2|A \cap B|}{|A| + |B|}$$

Rango [0, 1], donde 1 es coincidencia perfecta.

---

## E

### Epoch
Una pasada completa por todo el dataset de entrenamiento. El modelo típicamente se entrena durante múltiples epochs.

---

## F

### False Positive (FP)
Detección incorrecta: el modelo predice un objeto donde no existe.

### False Negative (FN)
Objeto no detectado: el modelo falla en encontrar un objeto que sí existe.

### Feature Pyramid Network (FPN)
Arquitectura que combina características de múltiples escalas para detectar objetos de diferentes tamaños.

```
    P5 (small features, large objects)
     ↑
    P4
     ↑
    P3 (large features, small objects)
```

### Fine-tuning
Proceso de ajustar un modelo pre-entrenado con datos específicos de tu dominio, manteniendo el conocimiento previo.

---

## G

### Ground Truth
Etiquetas correctas creadas manualmente que se usan para entrenar y evaluar el modelo.

---

## H

### Head
Parte final de la red que produce las predicciones (clases, bounding boxes, máscaras). En YOLO, genera las detecciones finales.

---

## I

### ImageNet
Dataset masivo con 14M+ imágenes y 1000 clases. Estándar para pre-entrenamiento de modelos de clasificación.

### Inference
Proceso de usar un modelo entrenado para hacer predicciones en nuevos datos.

### Instance Segmentation
Segmentación que distingue objetos individuales de la misma clase. Cada persona se segmenta por separado.

### IoU (Intersection over Union)
Métrica fundamental que mide el solapamiento entre predicción y ground truth:

$$IoU = \frac{\text{Área Intersección}}{\text{Área Unión}}$$

- IoU = 1: Coincidencia perfecta
- IoU > 0.5: Típicamente considerado "correcto"

---

## M

### mAP (mean Average Precision)
Métrica estándar para evaluar detectores. Promedio de AP (Average Precision) sobre todas las clases.

- **mAP@0.5**: IoU threshold = 0.5
- **mAP@0.5:0.95**: Promedio sobre múltiples thresholds

### Mask
Imagen binaria donde cada píxel indica pertenencia a un objeto (1) o fondo (0). Usada en segmentación.

### Multi-scale Detection
Detección de objetos a diferentes escalas/tamaños usando feature maps de diferentes resoluciones.

---

## N

### Neck
Componente entre backbone y head que fusiona características de múltiples escalas. En YOLO: FPN + PAN.

### NMS (Non-Maximum Suppression)
Algoritmo que elimina detecciones duplicadas, manteniendo solo la de mayor confianza cuando múltiples boxes se solapan.

```python
# Pseudo-código NMS
while boxes:
    best = max(boxes, key=confidence)
    keep(best)
    remove boxes with IoU(best, box) > threshold
```

---

## O

### Object Detection
Tarea de identificar y localizar objetos en imágenes, proporcionando clase, posición y confianza para cada objeto.

### One-Stage Detector
Detector que predice clases y posiciones en una sola pasada (YOLO, SSD). Más rápido que two-stage.

---

## P

### Panoptic Segmentation
Combinación de segmentación semántica (stuff: cielo, suelo) e instancias (things: personas, coches). Todo píxel clasificado.

### Precision
Proporción de detecciones correctas sobre el total de detecciones:

$$Precision = \frac{TP}{TP + FP}$$

### Pre-trained Model
Modelo entrenado en un dataset grande (ImageNet, COCO) que se puede usar directamente o como punto de partida para fine-tuning.

---

## R

### R-CNN (Region-based CNN)
Familia de detectores two-stage: R-CNN → Fast R-CNN → Faster R-CNN. Proponen regiones y luego clasifican.

### Recall
Proporción de objetos reales que fueron detectados:

$$Recall = \frac{TP}{TP + FN}$$

### Region Proposal
Proceso de generar candidatos de regiones que podrían contener objetos.

### ROI (Region of Interest)
Área específica de una imagen donde se enfoca el análisis.

---

## S

### SAM (Segment Anything Model)
Modelo de segmentación de Meta AI que puede segmentar cualquier objeto sin entrenamiento específico, usando prompts (puntos, boxes, texto).

### Semantic Segmentation
Clasificación de cada píxel por categoría, sin distinguir instancias individuales. Todas las personas comparten el mismo color.

### Stride
Paso de desplazamiento en operaciones de convolución. Stride mayor reduce el tamaño del feature map.

---

## T

### Threshold
Umbral para filtrar predicciones:
- **Confidence threshold**: Mínima confianza aceptada
- **IoU threshold**: Para NMS y evaluación

### Tracking
Seguimiento de objetos a través del tiempo en video, manteniendo identidades consistentes.

### Transfer Learning
Técnica de usar conocimiento de un modelo pre-entrenado para una nueva tarea, acelerando el entrenamiento.

### True Positive (TP)
Detección correcta: el modelo predice correctamente un objeto que existe.

### Two-Stage Detector
Detector que primero propone regiones y luego las clasifica (R-CNN family). Más preciso pero más lento.

---

## U

### U-Net
Arquitectura encoder-decoder para segmentación con conexiones skip, muy usada en imágenes médicas.

```
Encoder (↓) → Bottleneck → Decoder (↑)
     ↘----- Skip Connections -----↗
```

---

## Y

### YOLO (You Only Look Once)
Familia de detectores one-stage que procesan la imagen completa en una pasada. Conocido por su velocidad y balance precisión/rendimiento.

Versiones: YOLOv1 (2016) → YOLOv8 (2023)

### YOLOv8
Última versión de YOLO por Ultralytics. Soporta:
- Detección (yolov8n.pt)
- Segmentación (yolov8n-seg.pt)
- Clasificación (yolov8n-cls.pt)
- Pose estimation (yolov8n-pose.pt)

---

## Fórmulas Clave

| Métrica | Fórmula |
|---------|---------|
| **IoU** | $\frac{Intersección}{Unión}$ |
| **Precision** | $\frac{TP}{TP + FP}$ |
| **Recall** | $\frac{TP}{TP + FN}$ |
| **F1-Score** | $\frac{2 \times P \times R}{P + R}$ |
| **Dice** | $\frac{2 \times Intersección}{Pred + GT}$ |

---

## Comparativa de Tareas CV

| Tarea | Input | Output | Ejemplo |
|-------|-------|--------|---------|
| Clasificación | Imagen | 1 etiqueta | "Es un gato" |
| Detección | Imagen | N × (bbox, clase, conf) | "Gato en [x,y,w,h]" |
| Segmentación Semántica | Imagen | Máscara H×W (clases) | Cada píxel → clase |
| Segmentación Instancias | Imagen | N máscaras | Cada objeto → máscara |
| Panóptica | Imagen | Mapa completo | Todo clasificado |

---

_Glosario actualizado: Enero 2025_
