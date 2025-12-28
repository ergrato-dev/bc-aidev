"""
Proyecto: Detector de Objetos en Tiempo Real
============================================

Sistema completo de detección de objetos usando YOLOv8.

Instrucciones:
1. Completa las funciones marcadas con TODO
2. Sigue las especificaciones del README.md
3. Prueba con diferentes fuentes (imagen, video, webcam)
"""

import json
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================
# CLASE PRINCIPAL: ObjectDetector
# ============================================


class ObjectDetector:
    """
    Detector de objetos usando YOLOv8.

    Soporta detección en imágenes, videos y webcam.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: Optional[list] = None,
    ):
        """
        Inicializa el detector de objetos.

        Args:
            model_name: Nombre del modelo YOLOv8 (yolov8n/s/m/l/x.pt)
            conf_threshold: Umbral de confianza mínimo [0-1]
            iou_threshold: Umbral IoU para NMS [0-1]
            classes: Lista de IDs de clase a detectar (None=todas)
        """
        # TODO: Cargar el modelo YOLOv8
        # Hint: self.model = YOLO(model_name)
        self.model = None

        # TODO: Guardar parámetros de configuración
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes

        # TODO: Obtener nombres de clases del modelo
        # Hint: self.class_names = self.model.names
        self.class_names = {}

    def detect_image(self, source: Union[str, np.ndarray]) -> dict:
        """
        Detecta objetos en una imagen.

        Args:
            source: Path a imagen o array numpy (BGR)

        Returns:
            dict con detecciones y metadata
        """
        # TODO: Implementar detección en imagen
        # 1. Ejecutar model.predict() con los parámetros configurados
        # 2. Extraer información de cada detección (clase, confianza, bbox)
        # 3. Retornar diccionario con formato especificado

        result = {
            "source": str(source) if isinstance(source, str) else "numpy_array",
            "total_objects": 0,
            "detections": [],
            "processing_time_ms": 0,
        }

        # Tu código aquí...

        return result

    def detect_video(
        self, source: str, output_path: Optional[str] = None, show: bool = False
    ) -> dict:
        """
        Detecta objetos en un video.

        Args:
            source: Path al video
            output_path: Path para guardar video anotado (opcional)
            show: Mostrar video en tiempo real

        Returns:
            dict con estadísticas del video
        """
        # TODO: Implementar detección en video
        # 1. Abrir video con cv2.VideoCapture
        # 2. Procesar cada frame con detect_image()
        # 3. Si output_path, guardar video anotado
        # 4. Calcular estadísticas totales

        result = {
            "source": source,
            "total_frames": 0,
            "total_objects": 0,
            "objects_per_frame": [],
            "avg_fps": 0,
        }

        # Tu código aquí...

        return result

    def detect_webcam(
        self, camera_id: int = 0, window_name: str = "YOLOv8 Detection"
    ) -> None:
        """
        Detecta objetos en tiempo real desde webcam.

        Args:
            camera_id: ID de la cámara (0 = default)
            window_name: Nombre de la ventana

        Presiona 'q' para salir.
        """
        # TODO: Implementar detección en webcam
        # 1. Abrir webcam con cv2.VideoCapture(camera_id)
        # 2. Loop infinito procesando frames
        # 3. Mostrar FPS en pantalla
        # 4. Salir con tecla 'q'

        print("Detección en webcam no implementada")
        print("Presiona 'q' para salir")

        # Tu código aquí...

    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """
        Dibuja las detecciones en una imagen.

        Args:
            image: Imagen BGR
            detections: Lista de detecciones

        Returns:
            Imagen con bounding boxes dibujados
        """
        # TODO: Dibujar bounding boxes y etiquetas
        # 1. Para cada detección, dibujar rectángulo
        # 2. Añadir texto con clase y confianza
        # 3. Usar colores diferentes por clase

        annotated = image.copy()

        # Tu código aquí...

        return annotated

    def get_statistics(self, detections: list) -> dict:
        """
        Calcula estadísticas de las detecciones.

        Args:
            detections: Lista de detecciones

        Returns:
            dict con estadísticas
        """
        # TODO: Calcular estadísticas
        # - objects_per_class: Conteo por clase
        # - avg_confidence: Confianza promedio
        # - total_area: Área total cubierta

        stats = {"objects_per_class": {}, "avg_confidence": 0, "total_objects": 0}

        # Tu código aquí...

        return stats

    def process_batch(self, image_paths: list) -> list:
        """
        Procesa múltiples imágenes.

        Args:
            image_paths: Lista de paths a imágenes

        Returns:
            Lista de resultados de detección
        """
        # TODO: Procesar batch de imágenes
        # 1. Iterar sobre cada imagen
        # 2. Llamar detect_image()
        # 3. Agregar resultados a lista

        results = []

        # Tu código aquí...

        return results

    def export_to_json(self, results: list, output_path: str) -> None:
        """
        Exporta resultados a JSON.

        Args:
            results: Lista de resultados
            output_path: Path del archivo JSON
        """
        # TODO: Guardar resultados en JSON

        pass

    def export_to_csv(self, results: list, output_path: str) -> None:
        """
        Exporta resultados a CSV.

        Args:
            results: Lista de resultados
            output_path: Path del archivo CSV
        """
        # TODO: Guardar resultados en CSV
        # Columnas: source, class, confidence, x1, y1, x2, y2

        pass


# ============================================
# FUNCIONES AUXILIARES
# ============================================


def generate_colors(num_classes: int) -> dict:
    """
    Genera colores únicos para cada clase.

    Args:
        num_classes: Número de clases

    Returns:
        dict: {class_id: (B, G, R)}
    """
    # TODO: Generar paleta de colores
    # Hint: Usar HSV y convertir a BGR

    colors = {}

    # Tu código aquí...

    return colors


def calculate_fps(prev_time: float) -> tuple:
    """
    Calcula FPS.

    Args:
        prev_time: Tiempo del frame anterior

    Returns:
        tuple: (fps, current_time)
    """
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    return fps, current_time


# ============================================
# PROGRAMA PRINCIPAL
# ============================================


def main():
    """Función principal para probar el detector."""

    print("=" * 50)
    print("🎯 DETECTOR DE OBJETOS - YOLOv8")
    print("=" * 50)

    # TODO: Crear instancia del detector
    # detector = ObjectDetector(
    #     model_name='yolov8n.pt',
    #     conf_threshold=0.25,
    #     classes=None  # Todas las clases
    # )

    # TODO: Probar detección en imagen
    # print("\n📷 Detectando en imagen...")
    # result = detector.detect_image("https://ultralytics.com/images/bus.jpg")
    # print(f"   Objetos detectados: {result['total_objects']}")

    # TODO: Mostrar estadísticas
    # stats = detector.get_statistics(result['detections'])
    # print(f"   Objetos por clase: {stats['objects_per_class']}")

    # TODO: Probar con webcam (opcional)
    # print("\n📹 Iniciando detección en webcam...")
    # print("   Presiona 'q' para salir")
    # detector.detect_webcam()

    print("\n✅ Proyecto completado")
    print("   Implementa las funciones TODO y prueba el detector")


if __name__ == "__main__":
    main()
