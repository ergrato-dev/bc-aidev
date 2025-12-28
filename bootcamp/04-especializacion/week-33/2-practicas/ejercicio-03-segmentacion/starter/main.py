"""
Ejercicio 03: Segmentación de Imágenes
======================================

Aprende a segmentar objetos usando YOLOv8-seg y conceptos de segmentación.
"""

# ============================================
# PASO 1: Importar Librerías
# ============================================
print("--- Paso 1: Importar Librerías ---")

# Descomenta las siguientes líneas:
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# print("✅ Librerías importadas correctamente")

print()


# ============================================
# PASO 2: Cargar Modelo de Segmentación
# ============================================
print("--- Paso 2: Cargar Modelo de Segmentación ---")

# Descomenta las siguientes líneas:
# # Cargar YOLOv8 con segmentación (nota el sufijo -seg)
# model = YOLO('yolov8n-seg.pt')
#
# print("✅ Modelo YOLOv8n-seg cargado")
# print(f"   Tarea: {model.task}")  # 'segment'
# print(f"   Clases: {len(model.names)}")

print()


# ============================================
# PASO 3: Ejecutar Segmentación
# ============================================
print("--- Paso 3: Ejecutar Segmentación ---")

# Descomenta las siguientes líneas:
# # URL de imagen de ejemplo
# image_url = "https://ultralytics.com/images/bus.jpg"
#
# # Segmentar imagen
# results = model(image_url)
# result = results[0]
#
# print(f"✅ Segmentación completada")
# print(f"   Objetos segmentados: {len(result.masks) if result.masks else 0}")
# print(f"   Imagen original: {result.orig_shape}")

print()


# ============================================
# PASO 4: Analizar Máscaras
# ============================================
print("--- Paso 4: Analizar Máscaras ---")

# Descomenta las siguientes líneas:
# print("\n🎭 MÁSCARAS DETECTADAS:")
# print("-" * 70)
# print(f"{'#':<4}{'Clase':<15}{'Confianza':<12}{'Píxeles':<15}{'% Imagen':<12}")
# print("-" * 70)
#
# total_pixels = result.orig_shape[0] * result.orig_shape[1]
#
# for i, (mask, box) in enumerate(zip(result.masks.data, result.boxes)):
#     class_id = int(box.cls[0])
#     class_name = model.names[class_id]
#     confidence = box.conf[0].item()
#
#     # Contar píxeles en la máscara
#     mask_np = mask.cpu().numpy()
#     pixels = mask_np.sum()
#     percentage = (pixels / total_pixels) * 100
#
#     print(f"{i+1:<4}{class_name:<15}{confidence:<12.2%}{pixels:<15.0f}{percentage:<12.2f}%")
#
# print("-" * 70)

print()


# ============================================
# PASO 5: Visualizar Segmentación
# ============================================
print("--- Paso 5: Visualizar Segmentación ---")

# Descomenta las siguientes líneas:
# # Generar imagen anotada con máscaras
# annotated = result.plot()  # Incluye máscaras coloreadas
#
# # Convertir BGR -> RGB
# annotated_rgb = annotated[..., ::-1]
#
# # Mostrar
# plt.figure(figsize=(14, 8))
#
# plt.subplot(121)
# plt.imshow(result.orig_img[..., ::-1])
# plt.title('Imagen Original')
# plt.axis('off')
#
# plt.subplot(122)
# plt.imshow(annotated_rgb)
# plt.title(f'Segmentación ({len(result.masks)} objetos)')
# plt.axis('off')
#
# plt.tight_layout()
# plt.savefig('segmentation_comparison.png', dpi=150)
# print("✅ Imagen guardada: segmentation_comparison.png")
# plt.show()

print()


# ============================================
# PASO 6: Extraer Máscara Individual
# ============================================
print("--- Paso 6: Extraer Máscara Individual ---")

# Descomenta las siguientes líneas:
# if result.masks is not None and len(result.masks) > 0:
#     # Tomar el primer objeto (generalmente el más grande/confiable)
#     mask = result.masks.data[0].cpu().numpy()
#     box = result.boxes[0]
#
#     class_name = model.names[int(box.cls[0])]
#
#     # Redimensionar máscara al tamaño original
#     orig_h, orig_w = result.orig_shape
#     mask_resized = cv2.resize(mask, (orig_w, orig_h))
#
#     # Visualizar máscara binaria
#     plt.figure(figsize=(12, 4))
#
#     plt.subplot(131)
#     plt.imshow(result.orig_img[..., ::-1])
#     plt.title('Original')
#     plt.axis('off')
#
#     plt.subplot(132)
#     plt.imshow(mask_resized, cmap='gray')
#     plt.title(f'Máscara: {class_name}')
#     plt.axis('off')
#
#     plt.subplot(133)
#     plt.imshow(mask_resized > 0.5, cmap='Reds', alpha=0.7)
#     plt.imshow(result.orig_img[..., ::-1], alpha=0.3)
#     plt.title('Overlay')
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.savefig('mask_individual.png', dpi=150)
#     print(f"✅ Máscara extraída para: {class_name}")
#     plt.show()

print()


# ============================================
# PASO 7: Combinar Máscaras por Clase
# ============================================
print("--- Paso 7: Combinar Máscaras por Clase ---")

# Descomenta las siguientes líneas:
# if result.masks is not None:
#     orig_h, orig_w = result.orig_shape
#
#     # Crear máscara combinada con colores únicos por instancia
#     combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
#
#     for i, mask in enumerate(result.masks.data):
#         mask_np = mask.cpu().numpy()
#         mask_resized = cv2.resize(mask_np, (orig_w, orig_h))
#         combined_mask[mask_resized > 0.5] = i + 1
#
#     # Visualizar
#     plt.figure(figsize=(12, 5))
#
#     plt.subplot(121)
#     plt.imshow(result.orig_img[..., ::-1])
#     plt.title('Original')
#     plt.axis('off')
#
#     plt.subplot(122)
#     plt.imshow(combined_mask, cmap='tab20')
#     plt.colorbar(label='Instance ID')
#     plt.title('Máscaras combinadas')
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.savefig('masks_combined.png', dpi=150)
#     print(f"✅ {len(result.masks)} máscaras combinadas")
#     plt.show()

print()


# ============================================
# PASO 8: Extraer Objeto de la Imagen
# ============================================
print("--- Paso 8: Extraer Objeto de la Imagen ---")

# Descomenta las siguientes líneas:
# if result.masks is not None and len(result.masks) > 0:
#     # Seleccionar primer objeto
#     mask = result.masks.data[0].cpu().numpy()
#     original = result.orig_img.copy()
#
#     # Redimensionar máscara
#     orig_h, orig_w = result.orig_shape
#     mask_resized = cv2.resize(mask, (orig_w, orig_h))
#
#     # Binarizar
#     mask_binary = (mask_resized > 0.5).astype(np.uint8)
#
#     # Crear imagen con solo el objeto (fondo transparente simulado)
#     mask_3d = np.stack([mask_binary] * 3, axis=-1)
#     extracted = (original * mask_3d).astype(np.uint8)
#
#     # Crear imagen con fondo blanco
#     background_white = np.ones_like(original) * 255
#     extracted_white_bg = np.where(mask_3d, original, background_white).astype(np.uint8)
#
#     # Visualizar
#     plt.figure(figsize=(15, 5))
#
#     plt.subplot(131)
#     plt.imshow(original[..., ::-1])
#     plt.title('Original')
#     plt.axis('off')
#
#     plt.subplot(132)
#     plt.imshow(extracted[..., ::-1])
#     plt.title('Extraído (fondo negro)')
#     plt.axis('off')
#
#     plt.subplot(133)
#     plt.imshow(extracted_white_bg[..., ::-1])
#     plt.title('Extraído (fondo blanco)')
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.savefig('object_extracted.png', dpi=150)
#     print("✅ Objeto extraído con diferentes fondos")
#     plt.show()

print()


# ============================================
# PASO 9: Calcular Métricas de Segmentación
# ============================================
print("--- Paso 9: Calcular Métricas de Segmentación ---")

# Descomenta las siguientes líneas:
# def calculate_iou(mask1, mask2):
#     """Calcula Intersection over Union entre dos máscaras."""
#     intersection = np.logical_and(mask1, mask2).sum()
#     union = np.logical_or(mask1, mask2).sum()
#     return intersection / union if union > 0 else 0
#
# def calculate_dice(mask1, mask2):
#     """Calcula Dice coefficient entre dos máscaras."""
#     intersection = np.logical_and(mask1, mask2).sum()
#     total = mask1.sum() + mask2.sum()
#     return 2 * intersection / total if total > 0 else 0
#
# # Ejemplo: comparar dos máscaras del mismo resultado
# if result.masks is not None and len(result.masks) >= 2:
#     mask1 = result.masks.data[0].cpu().numpy() > 0.5
#     mask2 = result.masks.data[1].cpu().numpy() > 0.5
#
#     # Redimensionar al mismo tamaño
#     target_shape = (100, 100)
#     m1 = cv2.resize(mask1.astype(np.float32), target_shape) > 0.5
#     m2 = cv2.resize(mask2.astype(np.float32), target_shape) > 0.5
#
#     iou = calculate_iou(m1, m2)
#     dice = calculate_dice(m1, m2)
#
#     print("📊 Métricas entre primeros 2 objetos:")
#     print(f"   IoU (solapamiento): {iou:.4f}")
#     print(f"   Dice coefficient: {dice:.4f}")
#     print(f"   (Valores bajos = objetos separados ✓)")
# else:
#     print("   (Necesitas al menos 2 objetos para comparar)")

print()


# ============================================
# PASO 10: Función de Segmentación Reutilizable
# ============================================
print("--- Paso 10: Función de Segmentación Reutilizable ---")

# Descomenta las siguientes líneas:
# def segment_image(
#     source,
#     model_name='yolov8n-seg.pt',
#     conf_threshold=0.25,
#     classes=None,
#     extract_objects=False
# ):
#     """
#     Segmenta objetos en una imagen.
#
#     Args:
#         source: Path o URL de la imagen
#         model_name: Modelo de segmentación
#         conf_threshold: Umbral de confianza
#         classes: Clases a segmentar (None=todas)
#         extract_objects: Si True, retorna imágenes de cada objeto
#
#     Returns:
#         dict con máscaras y metadata
#     """
#     model = YOLO(model_name)
#
#     results = model.predict(
#         source=source,
#         conf=conf_threshold,
#         classes=classes,
#         verbose=False
#     )
#
#     result = results[0]
#
#     output = {
#         'num_objects': len(result.masks) if result.masks else 0,
#         'objects': [],
#         'combined_mask': None
#     }
#
#     if result.masks is None:
#         return output
#
#     orig_h, orig_w = result.orig_shape
#     combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
#
#     for i, (mask, box) in enumerate(zip(result.masks.data, result.boxes)):
#         mask_np = mask.cpu().numpy()
#         mask_resized = cv2.resize(mask_np, (orig_w, orig_h))
#         mask_binary = (mask_resized > 0.5).astype(np.uint8)
#
#         combined[mask_binary > 0] = i + 1
#
#         obj_data = {
#             'class': model.names[int(box.cls[0])],
#             'confidence': box.conf[0].item(),
#             'mask': mask_binary,
#             'bbox': box.xyxy[0].tolist(),
#             'area_pixels': mask_binary.sum()
#         }
#
#         if extract_objects:
#             mask_3d = np.stack([mask_binary] * 3, axis=-1)
#             obj_data['image'] = (result.orig_img * mask_3d).astype(np.uint8)
#
#         output['objects'].append(obj_data)
#
#     output['combined_mask'] = combined
#
#     return output
#
# print("✅ Función segment_image() definida")
# print("\n   Ejemplo de uso:")
# print('   results = segment_image("image.jpg", extract_objects=True)')
# print('   for obj in results["objects"]:')
# print('       print(f"{obj["class"]}: {obj["confidence"]:.2%}")')

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
   1. Cargar modelos YOLOv8-seg
   2. Ejecutar segmentación de instancias
   3. Extraer máscaras individuales
   4. Combinar máscaras por clase/instancia
   5. Extraer objetos de imágenes
   6. Calcular métricas (IoU, Dice)

🔑 Conceptos clave:
   - Máscara binaria: 0=fondo, 1=objeto
   - Segmentación de instancias: Cada objeto separado
   - IoU: Intersection over Union
   - Dice: 2×Intersección / (Pred + GT)

📦 Formatos de máscara:
   - masks.data: Tensor (N, H, W)
   - masks.xy: Polígonos de contorno
   - masks.xyn: Polígonos normalizados

🎯 Aplicaciones:
   - Edición de imágenes (remover fondo)
   - Conteo de objetos
   - Análisis de área/tamaño
   - Realidad aumentada
"""
)
