"""
Ejercicio 02: Detección de Objetos con YOLOv8
=============================================

Aprende a detectar objetos usando Ultralytics YOLOv8.
"""

# ============================================
# PASO 1: Importar Librerías
# ============================================
print("--- Paso 1: Importar Librerías ---")

# Descomenta las siguientes líneas:
# from ultralytics import YOLO
# import cv2
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# print("✅ Librerías importadas correctamente")

print()


# ============================================
# PASO 2: Cargar Modelo YOLOv8
# ============================================
print("--- Paso 2: Cargar Modelo YOLOv8 ---")

# Descomenta las siguientes líneas:
# # Cargar YOLOv8 nano (más rápido para aprendizaje)
# model = YOLO('yolov8n.pt')
#
# # Ver información del modelo
# print("✅ Modelo YOLOv8n cargado")
# print(f"   Tarea: {model.task}")
# print(f"   Clases: {len(model.names)} (COCO dataset)")
#
# # Mostrar algunas clases
# print("\n   Primeras 10 clases:")
# for i in range(10):
#     print(f"     {i}: {model.names[i]}")

print()


# ============================================
# PASO 3: Detectar en Imagen de Ejemplo
# ============================================
print("--- Paso 3: Detectar en Imagen de Ejemplo ---")

# Descomenta las siguientes líneas:
# # URL de imagen de ejemplo (bus con personas)
# image_url = "https://ultralytics.com/images/bus.jpg"
#
# # Ejecutar detección
# print(f"   Procesando: {image_url}")
# results = model(image_url)
#
# # Obtener resultados
# result = results[0]
# print(f"\n✅ Detección completada")
# print(f"   Imagen shape: {result.orig_shape}")
# print(f"   Objetos detectados: {len(result.boxes)}")

print()


# ============================================
# PASO 4: Analizar Detecciones
# ============================================
print("--- Paso 4: Analizar Detecciones ---")

# Descomenta las siguientes líneas:
# print("\n📦 DETECCIONES:")
# print("-" * 70)
# print(f"{'#':<4}{'Clase':<15}{'Confianza':<12}{'Coordenadas (x1,y1,x2,y2)':<35}")
# print("-" * 70)
#
# for i, box in enumerate(result.boxes):
#     # Extraer información
#     class_id = int(box.cls[0].item())
#     class_name = model.names[class_id]
#     confidence = box.conf[0].item()
#     coords = box.xyxy[0].tolist()
#
#     # Formatear coordenadas
#     coords_str = f"({coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f})"
#
#     print(f"{i+1:<4}{class_name:<15}{confidence:<12.2%}{coords_str:<35}")
#
# print("-" * 70)

print()


# ============================================
# PASO 5: Visualizar Resultados
# ============================================
print("--- Paso 5: Visualizar Resultados ---")

# Descomenta las siguientes líneas:
# # Generar imagen anotada
# annotated_img = result.plot()  # BGR format
#
# # Convertir BGR -> RGB para matplotlib
# annotated_img_rgb = annotated_img[..., ::-1]
#
# # Mostrar y guardar
# plt.figure(figsize=(12, 8))
# plt.imshow(annotated_img_rgb)
# plt.axis('off')
# plt.title(f'YOLOv8 - {len(result.boxes)} objetos detectados')
# plt.tight_layout()
#
# # Guardar resultado
# output_path = Path('detection_result.png')
# plt.savefig(output_path, dpi=150, bbox_inches='tight')
# print(f"✅ Imagen guardada: {output_path}")
#
# plt.show()

print()


# ============================================
# PASO 6: Configurar Parámetros de Detección
# ============================================
print("--- Paso 6: Configurar Parámetros de Detección ---")

# Descomenta las siguientes líneas:
# # Detección con parámetros personalizados
# results_custom = model.predict(
#     source=image_url,
#     conf=0.5,         # Solo detecciones con >50% confianza
#     iou=0.45,         # Umbral IoU para NMS
#     max_det=50,       # Máximo 50 detecciones
#     classes=[0, 5],   # Solo personas (0) y buses (5)
#     verbose=False     # Sin logs detallados
# )
#
# result_custom = results_custom[0]
# print(f"✅ Detección con filtros:")
# print(f"   Umbral confianza: 0.5")
# print(f"   Clases filtradas: persona, bus")
# print(f"   Objetos encontrados: {len(result_custom.boxes)}")
#
# # Mostrar solo las clases filtradas
# for box in result_custom.boxes:
#     class_id = int(box.cls[0].item())
#     class_name = model.names[class_id]
#     conf = box.conf[0].item()
#     print(f"   → {class_name}: {conf:.2%}")

print()


# ============================================
# PASO 7: Diferentes Formatos de Coordenadas
# ============================================
print("--- Paso 7: Diferentes Formatos de Coordenadas ---")

# Descomenta las siguientes líneas:
# print("\n🔢 Formatos de coordenadas disponibles:")
#
# if len(result.boxes) > 0:
#     box = result.boxes[0]  # Primera detección
#
#     print(f"\n   Detección: {model.names[int(box.cls[0])]}")
#     print(f"   xyxy (esquinas):     {box.xyxy[0].tolist()}")
#     print(f"   xywh (centro+size):  {box.xywh[0].tolist()}")
#     print(f"   xyxyn (normalizado): {box.xyxyn[0].tolist()}")
#     print(f"   xywhn (norm+center): {box.xywhn[0].tolist()}")

print()


# ============================================
# PASO 8: Procesar Múltiples Imágenes
# ============================================
print("--- Paso 8: Procesar Múltiples Imágenes ---")

# Descomenta las siguientes líneas:
# # Lista de imágenes de ejemplo
# test_images = [
#     "https://ultralytics.com/images/zidane.jpg",
#     "https://ultralytics.com/images/bus.jpg"
# ]
#
# print("📷 Procesando múltiples imágenes:")
#
# for img_url in test_images:
#     results = model(img_url, verbose=False)
#     result = results[0]
#
#     # Contar por clase
#     class_counts = {}
#     for box in result.boxes:
#         class_name = model.names[int(box.cls[0])]
#         class_counts[class_name] = class_counts.get(class_name, 0) + 1
#
#     print(f"\n   {img_url.split('/')[-1]}:")
#     for cls, count in sorted(class_counts.items()):
#         print(f"     - {cls}: {count}")

print()


# ============================================
# PASO 9: Comparar Modelos YOLOv8
# ============================================
print("--- Paso 9: Comparar Modelos YOLOv8 ---")

# Descomenta las siguientes líneas:
# import time
#
# # Comparar velocidad (solo nano y small para ser rápido)
# models_to_compare = ['yolov8n.pt', 'yolov8s.pt']
#
# print("\n⏱️ Comparación de velocidad:")
# print("-" * 50)
#
# for model_name in models_to_compare:
#     model_test = YOLO(model_name)
#
#     # Warmup
#     _ = model_test(image_url, verbose=False)
#
#     # Benchmark (3 iteraciones)
#     times = []
#     for _ in range(3):
#         start = time.time()
#         _ = model_test(image_url, verbose=False)
#         times.append(time.time() - start)
#
#     avg_time = sum(times) / len(times)
#     fps = 1 / avg_time
#
#     print(f"   {model_name:<12}: {avg_time*1000:.1f}ms ({fps:.1f} FPS)")
#
# print("-" * 50)

print()


# ============================================
# PASO 10: Función de Detección Reutilizable
# ============================================
print("--- Paso 10: Función de Detección Reutilizable ---")

# Descomenta las siguientes líneas:
# def detect_objects(
#     source,
#     model_name='yolov8n.pt',
#     conf_threshold=0.25,
#     classes=None,
#     save=False,
#     show=True
# ):
#     """
#     Detecta objetos en una imagen o video.
#
#     Args:
#         source: Path o URL de la imagen/video
#         model_name: Modelo a usar (yolov8n/s/m/l/x.pt)
#         conf_threshold: Umbral de confianza mínimo
#         classes: Lista de IDs de clase a detectar (None = todas)
#         save: Guardar imagen anotada
#         show: Mostrar imagen con matplotlib
#
#     Returns:
#         dict con detecciones por clase
#     """
#     # Cargar modelo
#     model = YOLO(model_name)
#
#     # Detectar
#     results = model.predict(
#         source=source,
#         conf=conf_threshold,
#         classes=classes,
#         verbose=False
#     )
#
#     result = results[0]
#
#     # Contar por clase
#     detections = {}
#     for box in result.boxes:
#         class_name = model.names[int(box.cls[0])]
#         conf = box.conf[0].item()
#
#         if class_name not in detections:
#             detections[class_name] = []
#
#         detections[class_name].append({
#             'confidence': conf,
#             'bbox': box.xyxy[0].tolist()
#         })
#
#     # Visualizar
#     if show or save:
#         annotated = result.plot()[..., ::-1]
#
#         if show:
#             plt.figure(figsize=(10, 8))
#             plt.imshow(annotated)
#             plt.axis('off')
#             plt.show()
#
#         if save:
#             result.save(filename='detection_output.jpg')
#
#     return detections
#
# # Ejemplo de uso
# print("✅ Función detect_objects() definida")
# print("\n   Ejemplo de uso:")
# print('   results = detect_objects("image.jpg", conf_threshold=0.5)')

print()


# ============================================
# RESUMEN
# ============================================
print("=" * 50)
print("📊 RESUMEN DEL EJERCICIO")
print("=" * 50)
print(
    """
✅ Aprendiste a:
   1. Cargar modelos YOLOv8 (nano, small, medium...)
   2. Ejecutar detección en imágenes
   3. Interpretar coordenadas (xyxy, xywh, normalizadas)
   4. Ajustar parámetros (conf, iou, classes)
   5. Visualizar y guardar resultados

🔑 Parámetros clave:
   - conf: Umbral de confianza mínima
   - iou: Umbral para Non-Maximum Suppression
   - classes: Filtrar por IDs de clase COCO
   - max_det: Máximo de detecciones

📦 Modelos YOLOv8:
   - yolov8n: Más rápido (edge/mobile)
   - yolov8s: Balance velocidad/precisión
   - yolov8m/l/x: Mayor precisión

🏷️ Clases COCO más comunes:
   0: person, 1: bicycle, 2: car, 5: bus
   16: dog, 17: cat, 67: cell phone
"""
)
