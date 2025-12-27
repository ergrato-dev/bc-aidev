"""
Ejercicio 02: Autograd y Gradientes
Bootcamp IA: Zero to Hero | Semana 21

Descomenta cada sección según avances en el ejercicio.
"""

import torch

print("=" * 60)
print("EJERCICIO 02: AUTOGRAD Y GRADIENTES")
print("=" * 60)

# ============================================
# PASO 1: requires_grad Básico
# ============================================
print("\n--- Paso 1: requires_grad Básico ---")

# Crear tensor con seguimiento de gradientes
# x = torch.tensor([2.0], requires_grad=True)
# print(f"x = {x.item()}")
# print(f"requires_grad: {x.requires_grad}")

# Operación: y = x²
# y = x ** 2
# print(f"y = x² = {y.item()}")

# Calcular gradiente: dy/dx
# y.backward()
# print(f"dy/dx = 2x = {x.grad.item()}")  # Debería ser 4.0

# grad_fn muestra la operación que creó el tensor
# print(f"y.grad_fn: {y.grad_fn}")

print()

# ============================================
# PASO 2: Grafo Computacional
# ============================================
print("\n--- Paso 2: Grafo Computacional ---")

# Crear variables de entrada
# a = torch.tensor([2.0], requires_grad=True)
# b = torch.tensor([3.0], requires_grad=True)

# Construir grafo computacional
# c = a * b      # c = 6, MulBackward
# d = c + a      # d = 8, AddBackward
# e = d ** 2     # e = 64, PowBackward

# print(f"a = {a.item()}, b = {b.item()}")
# print(f"c = a * b = {c.item()}")
# print(f"d = c + a = {d.item()}")
# print(f"e = d² = {e.item()}")

# Backward calcula todos los gradientes
# e.backward()

# Gradientes calculados por la regla de la cadena
# de/da = de/dd * dd/dc * dc/da + de/dd * dd/da
#       = 2d * 1 * b + 2d * 1
#       = 16 * 3 + 16 = 48 + 8 = 56
# print(f"\nde/da = {a.grad.item()}")  # 56

# de/db = de/dd * dd/dc * dc/db
#       = 2d * 1 * a
#       = 16 * 2 = 32
# print(f"de/db = {b.grad.item()}")  # 32

print()

# ============================================
# PASO 3: Acumulación de Gradientes
# ============================================
print("\n--- Paso 3: Acumulación de Gradientes ---")

# x = torch.tensor([2.0], requires_grad=True)

# Primera backward
# y1 = x ** 2
# y1.backward()
# print(f"Después de y1 = x²: grad = {x.grad.item()}")  # 4

# Segunda backward (¡se acumula!)
# y2 = x ** 3
# y2.backward()
# print(f"Después de y2 = x³: grad = {x.grad.item()}")  # 4 + 12 = 16

# Limpiar gradientes
# x.grad.zero_()
# print(f"Después de zero_grad(): grad = {x.grad.item()}")  # 0

# Tercera backward (ahora desde cero)
# y3 = x ** 2
# y3.backward()
# print(f"Después de y3 = x²: grad = {x.grad.item()}")  # 4

print()

# ============================================
# PASO 4: torch.no_grad()
# ============================================
print("\n--- Paso 4: torch.no_grad() ---")

# x = torch.tensor([2.0], requires_grad=True)

# Con gradientes (modo entrenamiento)
# y = x ** 2
# print(f"y = x² con gradientes:")
# print(f"  requires_grad: {y.requires_grad}")
# print(f"  grad_fn: {y.grad_fn}")

# Sin gradientes (modo inferencia)
# with torch.no_grad():
#     z = x ** 2
#     print(f"\nz = x² sin gradientes (dentro de no_grad):")
#     print(f"  requires_grad: {z.requires_grad}")
#     print(f"  grad_fn: {z.grad_fn}")

# También se puede usar como decorador
# @torch.no_grad()
# def inference(model, data):
#     return model(data)

print()

# ============================================
# PASO 5: detach()
# ============================================
print("\n--- Paso 5: detach() ---")

# x = torch.tensor([2.0], requires_grad=True)
# y = x ** 2

# y está conectado al grafo
# print(f"y.requires_grad: {y.requires_grad}")
# print(f"y.grad_fn: {y.grad_fn}")

# detach() crea copia sin conexión al grafo
# z = y.detach()
# print(f"\nz = y.detach():")
# print(f"z.requires_grad: {z.requires_grad}")
# print(f"z.grad_fn: {z.grad_fn}")

# Útil para pasar a NumPy
# numpy_array = y.detach().numpy()
# print(f"\ny.detach().numpy() = {numpy_array}")

# Útil para "congelar" valores intermedios
# frozen = y.detach()
# result = frozen * 2  # No propaga gradientes hacia y
# print(f"\nfrozen * 2 = {result.item()}")

print()

# ============================================
# PASO 6: Gradientes de Funciones Comunes
# ============================================
print("\n--- Paso 6: Gradientes de Funciones Comunes ---")

# x = torch.tensor([2.0], requires_grad=True)

# Lineal: y = 3x + 2 → dy/dx = 3
# y = 3 * x + 2
# y.backward()
# print(f"Lineal (3x + 2): dy/dx = {x.grad.item()}")  # 3
# x.grad.zero_()

# Cuadrática: y = x² → dy/dx = 2x
# y = x ** 2
# y.backward()
# print(f"Cuadrática (x²): dy/dx = 2x = {x.grad.item()}")  # 4
# x.grad.zero_()

# Exponencial: y = e^x → dy/dx = e^x
# y = torch.exp(x)
# y.backward()
# print(f"Exponencial (e^x): dy/dx = {x.grad.item():.4f}")  # ~7.389
# x.grad.zero_()

# Logaritmo: y = ln(x) → dy/dx = 1/x
# y = torch.log(x)
# y.backward()
# print(f"Logaritmo (ln x): dy/dx = 1/x = {x.grad.item()}")  # 0.5
# x.grad.zero_()

# Sigmoid: σ(x) → σ(x)(1-σ(x))
# y = torch.sigmoid(x)
# y.backward()
# sigmoid_val = torch.sigmoid(torch.tensor(2.0)).item()
# expected = sigmoid_val * (1 - sigmoid_val)
# print(f"Sigmoid: dy/dx = {x.grad.item():.4f} (expected: {expected:.4f})")
# x.grad.zero_()

# ReLU: max(0, x) → 1 si x > 0, 0 si x ≤ 0
# y = torch.relu(x)
# y.backward()
# print(f"ReLU (x > 0): dy/dx = {x.grad.item()}")  # 1

print()

# ============================================
# PASO 7: Regresión Lineal Manual
# ============================================
print("\n--- Paso 7: Regresión Lineal Manual ---")

# Datos: y = 2x + 1 (con ruido)
# torch.manual_seed(42)
# X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
# y_true = 2 * X + 1 + torch.randn_like(X) * 0.1

# Parámetros a aprender
# w = torch.randn(1, requires_grad=True)
# b = torch.zeros(1, requires_grad=True)

# print(f"Inicial: w = {w.item():.4f}, b = {b.item():.4f}")

# Entrenamiento manual
# learning_rate = 0.01
# epochs = 500

# for epoch in range(epochs):
#     # Forward: y_pred = Xw + b
#     y_pred = X * w + b
#
#     # Loss: MSE
#     loss = ((y_pred - y_true) ** 2).mean()
#
#     # Backward
#     loss.backward()
#
#     # Actualizar parámetros (sin crear grafo)
#     with torch.no_grad():
#         w -= learning_rate * w.grad
#         b -= learning_rate * b.grad
#
#     # Limpiar gradientes
#     w.grad.zero_()
#     b.grad.zero_()
#
#     if (epoch + 1) % 100 == 0:
#         print(f"Epoch {epoch+1}: loss = {loss.item():.6f}, w = {w.item():.4f}, b = {b.item():.4f}")

# print(f"\nResultado final: y = {w.item():.4f}x + {b.item():.4f}")
# print(f"Esperado: y = 2x + 1")

print()

# ============================================
# VERIFICACIÓN FINAL
# ============================================
print("\n" + "=" * 60)
print("✅ Ejercicio completado!")
print("=" * 60)
print(
    """
Resumen de lo aprendido:
1. requires_grad activa el seguimiento de operaciones
2. backward() calcula gradientes automáticamente
3. Los gradientes se ACUMULAN - usar zero_grad()
4. no_grad() desactiva gradientes (inferencia)
5. detach() desconecta del grafo computacional
6. Autograd calcula derivadas de cualquier función
7. Podemos implementar optimización manual con gradientes
"""
)
