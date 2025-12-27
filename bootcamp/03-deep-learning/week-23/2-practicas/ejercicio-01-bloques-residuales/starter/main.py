# ============================================
# EJERCICIO 01: Implementación de Bloques Residuales
# ============================================
# Objetivo: Implementar BasicBlock y Bottleneck desde cero
# ============================================

print("=== Ejercicio 01: Bloques Residuales ===\n")

# ============================================
# PASO 1: Configuración del Entorno
# ============================================
print("--- Paso 1: Configuración ---")

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
print(f"PyTorch version: {torch.__version__}")

print()

# ============================================
# PASO 2: PlainBlock (Sin Skip Connection)
# ============================================
print("--- Paso 2: PlainBlock ---")

# Descomenta las siguientes líneas:
# class PlainBlock(nn.Module):
#     """Bloque sin conexión residual."""
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         return self.relu(out)  # Sin skip connection
#
# # Test
# plain = PlainBlock(64, 64)
# x = torch.randn(1, 64, 32, 32)
# y = plain(x)
# print(f'PlainBlock: {x.shape} -> {y.shape}')

print()

# ============================================
# PASO 3: BasicBlock (Con Skip Connection)
# ============================================
print("--- Paso 3: BasicBlock ---")

# Descomenta las siguientes líneas:
# class BasicBlock(nn.Module):
#     """Bloque residual básico (ResNet-18/34)."""
#     expansion = 1
#
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity  # ¡Skip connection!
#         return self.relu(out)
#
# # Test
# basic = BasicBlock(64, 64)
# x = torch.randn(1, 64, 32, 32)
# y = basic(x)
# print(f'BasicBlock: {x.shape} -> {y.shape}')

print()

# ============================================
# PASO 4: Bottleneck Block
# ============================================
print("--- Paso 4: Bottleneck ---")

# Descomenta las siguientes líneas:
# class Bottleneck(nn.Module):
#     """Bloque bottleneck (ResNet-50/101/152)."""
#     expansion = 4
#
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super().__init__()
#         # 1×1: Reducir canales
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#
#         # 3×3: Procesamiento espacial
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         # 1×1: Expandir canales
#         self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         return self.relu(out)
#
# # Test con downsample (64 -> 256 canales por expansion=4)
# downsample = nn.Sequential(
#     nn.Conv2d(64, 256, 1, bias=False),
#     nn.BatchNorm2d(256)
# )
# bottleneck = Bottleneck(64, 64, downsample=downsample)
# x = torch.randn(1, 64, 32, 32)
# y = bottleneck(x)
# print(f'Bottleneck: {x.shape} -> {y.shape}')

print()

# ============================================
# PASO 5: Comparar Parámetros
# ============================================
print("--- Paso 5: Comparación de Parámetros ---")

# Descomenta las siguientes líneas:
# def count_parameters(model):
#     """Cuenta parámetros entrenables."""
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# # Crear bloques con 64 canales de entrada
# plain = PlainBlock(64, 64)
# basic = BasicBlock(64, 64)
#
# # Bottleneck necesita downsample para primera capa
# downsample = nn.Sequential(
#     nn.Conv2d(64, 256, 1, bias=False),
#     nn.BatchNorm2d(256)
# )
# bottleneck = Bottleneck(64, 64, downsample=downsample)
#
# print(f'PlainBlock:  {count_parameters(plain):,} parámetros')
# print(f'BasicBlock:  {count_parameters(basic):,} parámetros')
# print(f'Bottleneck:  {count_parameters(bottleneck):,} parámetros')
#
# # Calcular FLOPs aproximados para input 64x32x32
# # BasicBlock: 2 * (64 * 64 * 3 * 3 * 32 * 32) = 75M FLOPs
# # Bottleneck: 64*64*1*1 + 64*64*3*3 + 64*256*1*1 = más eficiente por canal

print()

# ============================================
# PASO 6: Verificar Flujo de Gradientes
# ============================================
print("--- Paso 6: Flujo de Gradientes ---")

# Descomenta las siguientes líneas:
# def test_gradient_flow(block, name):
#     """Verifica que los gradientes fluyen a través del bloque."""
#     x = torch.randn(1, 64, 32, 32, requires_grad=True)
#     y = block(x)
#     loss = y.sum()
#     loss.backward()
#
#     grad_norm = x.grad.norm().item()
#     print(f'{name}: grad_norm = {grad_norm:.4f}')
#     return grad_norm
#
# # Probar cada bloque
# grad_plain = test_gradient_flow(PlainBlock(64, 64), 'PlainBlock')
# grad_basic = test_gradient_flow(BasicBlock(64, 64), 'BasicBlock')
#
# # El BasicBlock debería tener gradientes más estables
# print(f'\nRatio BasicBlock/PlainBlock: {grad_basic/grad_plain:.2f}')

print()

# ============================================
# RESUMEN
# ============================================
print("=== Resumen ===")
print("- PlainBlock: Sin skip connection")
print("- BasicBlock: Con skip connection (y = F(x) + x)")
print("- Bottleneck: 1x1 -> 3x3 -> 1x1 con expansion=4")
print("- Skip connections mejoran flujo de gradientes")
