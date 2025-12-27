# ⚡ Ejercicio 02: Autograd y Gradientes

## 🎯 Objetivo

Comprender el sistema de diferenciación automática de PyTorch para calcular gradientes necesarios en el entrenamiento de redes neuronales.

---

## 📋 Instrucciones

Este ejercicio te guiará a través del uso de autograd. Abre `starter/main.py` y descomenta cada sección según avances.

---

## Paso 1: requires_grad Básico

El flag `requires_grad` indica que queremos calcular gradientes:

```python
import torch

# Tensor con gradientes
x = torch.tensor([2.0], requires_grad=True)

# Operación
y = x ** 2  # y = x²

# Calcular gradiente
y.backward()

print(x.grad)  # dy/dx = 2x = 4
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

## Paso 2: Grafo Computacional

PyTorch construye un grafo de operaciones automáticamente:

```python
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)

c = a * b    # c = 6
d = c + a    # d = 8
e = d ** 2   # e = 64

e.backward()

print(a.grad)  # de/da
print(b.grad)  # de/db
```

**Descomenta** la sección del Paso 2.

---

## Paso 3: Acumulación de Gradientes

Los gradientes se acumulan por defecto. Hay que limpiarlos:

```python
x = torch.tensor([2.0], requires_grad=True)

# Primera backward
y1 = x ** 2
y1.backward()
print(x.grad)  # 4

# Segunda backward (se acumula!)
y2 = x ** 3
y2.backward()
print(x.grad)  # 4 + 12 = 16

# Limpiar gradientes
x.grad.zero_()
```

**Descomenta** la sección del Paso 3.

---

## Paso 4: torch.no_grad()

Desactiva el cálculo de gradientes para inferencia:

```python
x = torch.tensor([2.0], requires_grad=True)

# Con gradientes
y = x ** 2
print(y.requires_grad)  # True

# Sin gradientes (inferencia)
with torch.no_grad():
    z = x ** 2
    print(z.requires_grad)  # False
```

**Descomenta** la sección del Paso 4.

---

## Paso 5: detach()

Desconecta un tensor del grafo computacional:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

# y está conectado al grafo
z = y.detach()  # z NO está conectado

# Útil para:
# - Pasar a NumPy: y.detach().numpy()
# - Congelar parte de la red
```

**Descomenta** la sección del Paso 5.

---

## Paso 6: Gradientes de Funciones Comunes

Verifica que autograd calcula gradientes correctamente:

```python
# Lineal: y = 3x + 2 → dy/dx = 3
# Cuadrática: y = x² → dy/dx = 2x
# Exponencial: y = e^x → dy/dx = e^x
# Sigmoid: y = σ(x) → dy/dx = σ(x)(1-σ(x))
```

**Descomenta** la sección del Paso 6.

---

## Paso 7: Regresión Lineal Manual

Implementa regresión lineal usando solo autograd:

```python
# y = wx + b
# Aprende w y b usando gradientes
```

**Descomenta** la sección del Paso 7.

---

## ✅ Verificación

Al completar todos los pasos, deberías:
- Entender cómo funciona `requires_grad`
- Ver el grafo computacional en acción
- Saber cuándo usar `no_grad()` y `detach()`
- Implementar regresión lineal con gradientes manuales

---

## 📚 Recursos

- [Autograd Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
