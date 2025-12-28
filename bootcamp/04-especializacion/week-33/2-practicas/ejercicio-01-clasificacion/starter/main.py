"""
Ejercicio 01: Clasificación de Imágenes con PyTorch
===================================================

Aprende a clasificar imágenes usando modelos pre-entrenados.
"""

# ============================================
# PASO 1: Importar Librerías
# ============================================
print("--- Paso 1: Importar Librerías ---")

# Descomenta las siguientes líneas:
# import torch
# from torchvision import models, transforms
# from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
# from PIL import Image
# import requests
# from io import BytesIO
#
# print("✅ Librerías importadas correctamente")
# print(f"   PyTorch version: {torch.__version__}")
# print(f"   CUDA disponible: {torch.cuda.is_available()}")

print()


# ============================================
# PASO 2: Cargar Modelo Pre-entrenado
# ============================================
print("--- Paso 2: Cargar Modelo Pre-entrenado ---")

# Descomenta las siguientes líneas:
# model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# model.eval()  # Modo evaluación (desactiva dropout, batchnorm)
#
# # Contar parámetros
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# print(f"✅ Modelo: ResNet50")
# print(f"   Parámetros totales: {total_params:,}")
# print(f"   Parámetros entrenables: {trainable_params:,}")

print()


# ============================================
# PASO 3: Definir Preprocesamiento
# ============================================
print("--- Paso 3: Definir Preprocesamiento ---")

# Descomenta las siguientes líneas:
# # Transformaciones estándar para ImageNet
# preprocess = transforms.Compose([
#     transforms.Resize(256),              # Redimensionar lado menor a 256
#     transforms.CenterCrop(224),          # Recortar centro 224x224
#     transforms.ToTensor(),               # Convertir a tensor [0, 1]
#     transforms.Normalize(                # Normalizar con media/std ImageNet
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])
#
# print("✅ Pipeline de preprocesamiento definido:")
# print("   1. Resize(256)")
# print("   2. CenterCrop(224)")
# print("   3. ToTensor()")
# print("   4. Normalize(ImageNet mean/std)")

print()


# ============================================
# PASO 4: Descargar y Preprocesar Imagen
# ============================================
print("--- Paso 4: Descargar y Preprocesar Imagen ---")

# Descomenta las siguientes líneas:
# # URL de imagen de ejemplo (gato)
# url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"
#
# # Descargar imagen
# print(f"   Descargando imagen...")
# response = requests.get(url)
# img = Image.open(BytesIO(response.content)).convert('RGB')
#
# print(f"✅ Imagen cargada:")
# print(f"   Tamaño original: {img.size}")
# print(f"   Modo: {img.mode}")
#
# # Preprocesar
# input_tensor = preprocess(img)
# input_batch = input_tensor.unsqueeze(0)  # Añadir dimensión batch [1, 3, 224, 224]
#
# print(f"   Tensor shape: {input_batch.shape}")
# print(f"   Tensor dtype: {input_batch.dtype}")

print()


# ============================================
# PASO 5: Ejecutar Inferencia
# ============================================
print("--- Paso 5: Ejecutar Inferencia ---")

# Descomenta las siguientes líneas:
# # Mover a GPU si está disponible
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# input_batch = input_batch.to(device)
#
# # Inferencia (sin calcular gradientes)
# with torch.no_grad():
#     output = model(input_batch)
#
# print(f"✅ Inferencia completada")
# print(f"   Output shape: {output.shape}")  # [1, 1000] - 1000 clases ImageNet
# print(f"   Device: {device}")

print()


# ============================================
# PASO 6: Convertir a Probabilidades
# ============================================
print("--- Paso 6: Convertir a Probabilidades ---")

# Descomenta las siguientes líneas:
# # Aplicar softmax para obtener probabilidades
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
#
# print(f"✅ Probabilidades calculadas")
# print(f"   Suma de probabilidades: {probabilities.sum().item():.4f}")  # Debe ser ~1.0
# print(f"   Prob. máxima: {probabilities.max().item()*100:.2f}%")
# print(f"   Prob. mínima: {probabilities.min().item()*100:.6f}%")

print()


# ============================================
# PASO 7: Cargar Etiquetas ImageNet
# ============================================
print("--- Paso 7: Cargar Etiquetas ImageNet ---")

# Descomenta las siguientes líneas:
# # Descargar etiquetas de las 1000 clases
# IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
# labels = requests.get(IMAGENET_LABELS_URL).text.strip().split('\n')
#
# print(f"✅ Etiquetas cargadas: {len(labels)} clases")
# print(f"   Ejemplos: {labels[:5]}")

print()


# ============================================
# PASO 8: Mostrar Top-5 Predicciones
# ============================================
print("--- Paso 8: Mostrar Top-5 Predicciones ---")

# Descomenta las siguientes líneas:
# # Obtener top-5 predicciones
# top5_prob, top5_catid = torch.topk(probabilities, 5)
#
# print("\n" + "="*50)
# print("🎯 TOP-5 PREDICCIONES")
# print("="*50)
#
# for i in range(5):
#     label = labels[top5_catid[i]]
#     prob = top5_prob[i].item() * 100
#     bar = "█" * int(prob / 2)  # Barra visual
#     print(f"  {i+1}. {label:25s} {prob:6.2f}% {bar}")
#
# print("="*50)

print()


# ============================================
# PASO 9: Comparar con EfficientNet
# ============================================
print("--- Paso 9: Comparar con EfficientNet ---")

# Descomenta las siguientes líneas:
# # Cargar EfficientNet-B0
# model_efficient = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
# model_efficient.eval()
# model_efficient = model_efficient.to(device)
#
# # Inferencia
# with torch.no_grad():
#     output_efficient = model_efficient(input_batch)
#     probs_efficient = torch.nn.functional.softmax(output_efficient[0], dim=0)
#
# # Top-5
# top5_prob_eff, top5_catid_eff = torch.topk(probs_efficient, 5)
#
# print("\n🔄 Comparación ResNet50 vs EfficientNet-B0:")
# print("-" * 60)
# print(f"{'Rank':<6}{'ResNet50':<25}{'EfficientNet-B0':<25}")
# print("-" * 60)
#
# for i in range(5):
#     resnet_label = labels[top5_catid[i]][:20]
#     efficient_label = labels[top5_catid_eff[i]][:20]
#     print(f"{i+1:<6}{resnet_label:<25}{efficient_label:<25}")

print()


# ============================================
# PASO 10: Clasificar Imagen Local (Opcional)
# ============================================
print("--- Paso 10: Clasificar Imagen Local ---")

# Descomenta y modifica para usar tu propia imagen:
# def classify_image(image_path_or_url, model, preprocess, labels, device):
#     """Clasifica una imagen desde path local o URL."""
#
#     if image_path_or_url.startswith('http'):
#         response = requests.get(image_path_or_url)
#         img = Image.open(BytesIO(response.content)).convert('RGB')
#     else:
#         img = Image.open(image_path_or_url).convert('RGB')
#
#     # Preprocesar y clasificar
#     input_tensor = preprocess(img).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         output = model(input_tensor)
#         probs = torch.nn.functional.softmax(output[0], dim=0)
#
#     # Top-3
#     top3_prob, top3_idx = torch.topk(probs, 3)
#
#     results = []
#     for prob, idx in zip(top3_prob, top3_idx):
#         results.append({
#             'label': labels[idx],
#             'probability': prob.item()
#         })
#
#     return results
#
# # Ejemplo con otra imagen
# dog_url = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
# results = classify_image(dog_url, model, preprocess, labels, device)
#
# print("\n🐕 Clasificación de imagen de perro:")
# for r in results:
#     print(f"   {r['label']}: {r['probability']*100:.2f}%")

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
   1. Cargar modelos pre-entrenados (ResNet, EfficientNet)
   2. Preprocesar imágenes para ImageNet
   3. Ejecutar inferencia con PyTorch
   4. Interpretar probabilidades softmax
   5. Obtener top-k predicciones

🔑 Conceptos clave:
   - model.eval(): Modo evaluación
   - torch.no_grad(): Sin gradientes (inferencia)
   - softmax: Logits → Probabilidades
   - Transfer learning: Usar conocimiento previo

📦 Modelos comparados:
   - ResNet50: 25.6M params, 80.4% acc
   - EfficientNet-B0: 5.3M params, 77.1% acc
"""
)
