"""
Ejercicio 01: Convolución Manual
Bootcamp IA: Zero to Hero | Semana 22

Implementa convolución 2D desde cero para entender su funcionamiento.
"""

import numpy as np

# ============================================
# PASO 1: Convolución Básica Sin Padding
# ============================================
print("=" * 50)
print("PASO 1: Convolución Básica Sin Padding")
print("=" * 50)

# Descomenta las siguientes líneas:

# def conv2d_basic(image, kernel):
#     """
#     Convolución 2D básica sin padding.
#
#     Args:
#         image: Array 2D (H, W)
#         kernel: Array 2D (Kh, Kw)
#
#     Returns:
#         Feature map resultante
#     """
#     H, W = image.shape
#     Kh, Kw = kernel.shape
#
#     out_h = H - Kh + 1
#     out_w = W - Kw + 1
#
#     output = np.zeros((out_h, out_w))
#
#     for i in range(out_h):
#         for j in range(out_w):
#             region = image[i:i+Kh, j:j+Kw]
#             output[i, j] = np.sum(region * kernel)
#
#     return output
#
#
# # Probar con ejemplo simple
# image = np.array([
#     [1, 2, 3, 4, 5],
#     [6, 7, 8, 9, 0],
#     [1, 2, 3, 4, 5],
#     [6, 7, 8, 9, 0],
#     [1, 2, 3, 4, 5]
# ], dtype=float)
#
# kernel = np.array([
#     [1, 0, -1],
#     [1, 0, -1],
#     [1, 0, -1]
# ], dtype=float)
#
# result = conv2d_basic(image, kernel)
# print(f"Imagen shape: {image.shape}")
# print(f"Kernel shape: {kernel.shape}")
# print(f"Output shape: {result.shape}")
# print(f"Output:\n{result}")

print()

# ============================================
# PASO 2: Convolución con Padding
# ============================================
print("=" * 50)
print("PASO 2: Convolución con Padding")
print("=" * 50)

# Descomenta las siguientes líneas:

# def conv2d_padding(image, kernel, padding=1):
#     """
#     Convolución 2D con zero padding.
#
#     Args:
#         image: Array 2D (H, W)
#         kernel: Array 2D (Kh, Kw)
#         padding: Cantidad de padding
#
#     Returns:
#         Feature map con padding aplicado
#     """
#     padded = np.pad(image, padding, mode='constant', constant_values=0)
#     print(f"Shape con padding: {padded.shape}")
#     return conv2d_basic(padded, kernel)
#
#
# # Probar padding=1 para mantener tamaño
# result_padded = conv2d_padding(image, kernel, padding=1)
# print(f"Input shape: {image.shape}")
# print(f"Output shape (padding=1): {result_padded.shape}")
# print(f"Output:\n{result_padded}")

print()

# ============================================
# PASO 3: Convolución con Stride
# ============================================
print("=" * 50)
print("PASO 3: Convolución con Stride")
print("=" * 50)

# Descomenta las siguientes líneas:

# def conv2d_stride(image, kernel, stride=1):
#     """
#     Convolución 2D con stride.
#
#     Args:
#         image: Array 2D (H, W)
#         kernel: Array 2D (Kh, Kw)
#         stride: Paso del kernel
#
#     Returns:
#         Feature map con stride aplicado
#     """
#     H, W = image.shape
#     Kh, Kw = kernel.shape
#
#     out_h = (H - Kh) // stride + 1
#     out_w = (W - Kw) // stride + 1
#
#     output = np.zeros((out_h, out_w))
#
#     for i in range(out_h):
#         for j in range(out_w):
#             h_start = i * stride
#             w_start = j * stride
#             region = image[h_start:h_start+Kh, w_start:w_start+Kw]
#             output[i, j] = np.sum(region * kernel)
#
#     return output
#
#
# # Probar con diferentes strides
# image_large = np.random.randn(8, 8)
# kernel_3x3 = np.ones((3, 3)) / 9  # Blur
#
# result_s1 = conv2d_stride(image_large, kernel_3x3, stride=1)
# result_s2 = conv2d_stride(image_large, kernel_3x3, stride=2)
#
# print(f"Input: {image_large.shape}")
# print(f"Output (stride=1): {result_s1.shape}")
# print(f"Output (stride=2): {result_s2.shape}")

print()

# ============================================
# PASO 4: Kernels de Detección de Bordes
# ============================================
print("=" * 50)
print("PASO 4: Kernels de Detección de Bordes")
print("=" * 50)

# Descomenta las siguientes líneas:

# # Definir kernels
# kernel_vertical = np.array([
#     [-1, 0, 1],
#     [-1, 0, 1],
#     [-1, 0, 1]
# ], dtype=float)
#
# kernel_horizontal = np.array([
#     [-1, -1, -1],
#     [ 0,  0,  0],
#     [ 1,  1,  1]
# ], dtype=float)
#
# sobel_x = np.array([
#     [-1, 0, 1],
#     [-2, 0, 2],
#     [-1, 0, 1]
# ], dtype=float)
#
# sobel_y = np.array([
#     [-1, -2, -1],
#     [ 0,  0,  0],
#     [ 1,  2,  1]
# ], dtype=float)
#
# kernel_sharpen = np.array([
#     [ 0, -1,  0],
#     [-1,  5, -1],
#     [ 0, -1,  0]
# ], dtype=float)
#
# kernel_blur = np.ones((3, 3)) / 9
#
# print("Kernels definidos:")
# print("- kernel_vertical: detecta bordes verticales")
# print("- kernel_horizontal: detecta bordes horizontales")
# print("- sobel_x: Sobel para gradiente X")
# print("- sobel_y: Sobel para gradiente Y")
# print("- kernel_sharpen: enfoque")
# print("- kernel_blur: desenfoque (promedio)")

print()

# ============================================
# PASO 5: Visualización de Resultados
# ============================================
print("=" * 50)
print("PASO 5: Visualización de Resultados")
print("=" * 50)

# Descomenta las siguientes líneas:

# import matplotlib.pyplot as plt
#
# def visualize_convolution(image, kernel, title="Convolución"):
#     """Visualiza imagen original y resultado de convolución."""
#     result = conv2d_padding(image, kernel, padding=1)
#
#     fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#
#     axes[0].imshow(image, cmap='gray')
#     axes[0].set_title('Imagen Original')
#     axes[0].axis('off')
#
#     im = axes[1].imshow(kernel, cmap='RdBu', vmin=-2, vmax=2)
#     axes[1].set_title('Kernel')
#     for i in range(kernel.shape[0]):
#         for j in range(kernel.shape[1]):
#             axes[1].text(j, i, f'{kernel[i,j]:.0f}',
#                         ha='center', va='center', fontsize=12)
#     axes[1].axis('off')
#
#     axes[2].imshow(result, cmap='gray')
#     axes[2].set_title(title)
#     axes[2].axis('off')
#
#     plt.tight_layout()
#     plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150)
#     plt.show()
#     print(f"✓ Guardado: {title.lower().replace(' ', '_')}.png")
#
#
# # Crear imagen de prueba con patrón
# test_image = np.zeros((32, 32))
# test_image[8:24, 12:20] = 1  # Rectángulo vertical
# test_image[12:20, 8:24] = 1  # Rectángulo horizontal (forma cruz)
#
# # Visualizar diferentes filtros
# visualize_convolution(test_image, kernel_vertical, "Bordes Verticales")
# visualize_convolution(test_image, kernel_horizontal, "Bordes Horizontales")
# visualize_convolution(test_image, sobel_x, "Sobel X")

print()

# ============================================
# PASO 6: Convolución Completa
# ============================================
print("=" * 50)
print("PASO 6: Convolución Completa (Padding + Stride)")
print("=" * 50)

# Descomenta las siguientes líneas:

# def conv2d_full(image, kernel, padding=0, stride=1):
#     """
#     Convolución 2D completa con padding y stride.
#
#     Args:
#         image: Array 2D (H, W)
#         kernel: Array 2D (Kh, Kw)
#         padding: Zero padding
#         stride: Paso del kernel
#
#     Returns:
#         Feature map resultante
#     """
#     if padding > 0:
#         image = np.pad(image, padding, mode='constant', constant_values=0)
#
#     H, W = image.shape
#     Kh, Kw = kernel.shape
#
#     out_h = (H - Kh) // stride + 1
#     out_w = (W - Kw) // stride + 1
#
#     output = np.zeros((out_h, out_w))
#
#     for i in range(out_h):
#         for j in range(out_w):
#             h_start = i * stride
#             w_start = j * stride
#             region = image[h_start:h_start+Kh, w_start:w_start+Kw]
#             output[i, j] = np.sum(region * kernel)
#
#     return output
#
#
# # Probar diferentes combinaciones
# test_img = np.random.randn(16, 16)
# test_kernel = np.random.randn(3, 3)
#
# print("Diferentes configuraciones:")
# print(f"Input: {test_img.shape}")
#
# configs = [
#     (0, 1),  # Sin padding, stride 1
#     (1, 1),  # Padding 1, stride 1 (same)
#     (0, 2),  # Sin padding, stride 2
#     (1, 2),  # Padding 1, stride 2
# ]
#
# for p, s in configs:
#     out = conv2d_full(test_img, test_kernel, padding=p, stride=s)
#     print(f"  padding={p}, stride={s} -> {out.shape}")

print()

# ============================================
# VERIFICACIÓN FINAL
# ============================================
print("=" * 50)
print("VERIFICACIÓN FINAL")
print("=" * 50)

# Descomenta para verificar tu implementación:

# # Comparar con scipy
# from scipy.signal import correlate2d
#
# test_img = np.random.randn(10, 10)
# test_kernel = np.random.randn(3, 3)
#
# # Tu implementación
# my_result = conv2d_full(test_img, test_kernel, padding=1, stride=1)
#
# # Scipy (mode='same' es equivalente a padding que mantiene tamaño)
# scipy_result = correlate2d(test_img, test_kernel, mode='same')
#
# # Comparar
# diff = np.abs(my_result - scipy_result).max()
# print(f"Diferencia máxima con scipy: {diff:.10f}")
#
# if diff < 1e-10:
#     print("✅ ¡Implementación correcta!")
# else:
#     print("❌ Hay diferencias, revisa tu implementación")

print("\n✓ Ejercicio completado")
