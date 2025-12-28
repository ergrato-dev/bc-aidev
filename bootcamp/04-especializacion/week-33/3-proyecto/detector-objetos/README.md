# 🎯 Proyecto: Detector de Objetos en Tiempo Real

## 📋 Descripción

Construye un sistema completo de detección de objetos que funcione en tiempo real, capaz de procesar imágenes, videos y transmisiones de webcam usando YOLOv8.

## 🎯 Objetivos de Aprendizaje

- Implementar un detector de objetos modular y reutilizable
- Procesar diferentes fuentes de entrada (imagen, video, webcam)
- Aplicar filtros y configuraciones de detección
- Generar estadísticas y reportes de detección
- Optimizar el rendimiento para tiempo real

## ⏱️ Tiempo Estimado

2 horas

---

## 📋 Requisitos del Proyecto

### Funcionalidades Obligatorias

1. **Clase `ObjectDetector`**
   - Cargar diferentes modelos YOLOv8 (n, s, m, l, x)
   - Detectar en imagen, video y webcam
   - Configurar umbral de confianza e IoU
   - Filtrar por clases específicas

2. **Procesamiento de Resultados**
   - Extraer información de cada detección
   - Contar objetos por clase
   - Calcular estadísticas (confianza promedio, objetos/frame)
   - Generar reporte en formato JSON

3. **Visualización**
   - Dibujar bounding boxes con etiquetas
   - Mostrar contador de objetos
   - Indicar FPS en tiempo real
   - Guardar resultados anotados

4. **Pipeline Completo**
   - Función para procesar batch de imágenes
   - Exportar resultados a CSV
   - Generar video con detecciones

### Funcionalidades Opcionales (Bonus)

- Tracking de objetos entre frames
- Zonas de interés (ROI)
- Alertas cuando se detecta clase específica
- Dashboard con estadísticas en vivo

---

## 🗂️ Estructura del Proyecto

```
detector-objetos/
├── README.md
├── starter/
│   └── main.py           # Plantilla para completar
└── solution/
    └── main.py           # Solución de referencia
```

---

## 📝 Especificaciones Técnicas

### Clase `ObjectDetector`

```python
class ObjectDetector:
    def __init__(
        self,
        model_name: str = 'yolov8n.pt',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list = None
    ):
        """
        Inicializa el detector de objetos.
        
        Args:
            model_name: Nombre del modelo YOLOv8
            conf_threshold: Umbral de confianza mínimo
            iou_threshold: Umbral IoU para NMS
            classes: Lista de IDs de clase a detectar (None=todas)
        """
        pass
    
    def detect_image(self, source: str) -> dict:
        """Detecta objetos en una imagen."""
        pass
    
    def detect_video(self, source: str, output_path: str = None) -> dict:
        """Detecta objetos en un video."""
        pass
    
    def detect_webcam(self, camera_id: int = 0) -> None:
        """Detecta objetos en tiempo real desde webcam."""
        pass
    
    def get_statistics(self, detections: list) -> dict:
        """Calcula estadísticas de las detecciones."""
        pass
```

### Formato de Salida

```python
{
    "source": "image.jpg",
    "total_objects": 5,
    "detections": [
        {
            "class": "person",
            "confidence": 0.92,
            "bbox": [100, 150, 300, 450],
            "area": 60000
        }
    ],
    "statistics": {
        "objects_per_class": {"person": 3, "car": 2},
        "avg_confidence": 0.87,
        "processing_time_ms": 45.2
    }
}
```

---

## 📊 Criterios de Evaluación

| Criterio | Peso | Descripción |
|----------|------|-------------|
| Funcionalidad | 40% | Todas las funciones implementadas correctamente |
| Código limpio | 20% | Type hints, docstrings, nombres descriptivos |
| Manejo errores | 15% | Excepciones controladas, validaciones |
| Documentación | 15% | README, comentarios, ejemplos de uso |
| Bonus | 10% | Funcionalidades adicionales |

---

## 🚀 Instrucciones

### 1. Configura el Entorno

```bash
pip install ultralytics opencv-python matplotlib pandas
```

### 2. Implementa la Clase `ObjectDetector`

Abre `starter/main.py` y completa las funciones marcadas con `TODO`.

### 3. Prueba con Diferentes Fuentes

```python
detector = ObjectDetector(model_name='yolov8s.pt', conf_threshold=0.3)

# Imagen
result = detector.detect_image('test_image.jpg')
print(result)

# Video
result = detector.detect_video('test_video.mp4', output_path='output.mp4')

# Webcam
detector.detect_webcam()
```

### 4. Genera un Reporte

```python
# Procesar múltiples imágenes
results = detector.process_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# Exportar a CSV
detector.export_to_csv(results, 'detections_report.csv')
```

---

## 📚 Recursos

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [OpenCV Python](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [COCO Dataset Classes](https://cocodataset.org/#explore)

---

## ✅ Checklist de Entrega

- [ ] Clase `ObjectDetector` implementada
- [ ] Detección en imagen funcionando
- [ ] Detección en video funcionando
- [ ] Detección en webcam funcionando
- [ ] Estadísticas calculadas correctamente
- [ ] Código documentado con docstrings
- [ ] Al menos 5 imágenes de prueba procesadas
- [ ] Reporte de detecciones generado
