# 🔀 Ejercicio 03: Control de Flujo

## 🎯 Objetivos

- Dominar estructuras condicionales (if/elif/else)
- Implementar bucles for y while
- Usar break, continue y else en bucles
- Aplicar comprensiones de lista básicas

---

## 📋 Instrucciones

Abre el archivo `starter/main.py` y sigue los pasos descomentando el código indicado.

---

### Paso 1: Condicionales if/elif/else

Las estructuras condicionales ejecutan código basado en condiciones:

```python
if condicion:
    # código si True
elif otra_condicion:
    # código si anterior False y esta True
else:
    # código si todas False
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Operador Ternario

Forma compacta de escribir if/else en una línea:

```python
resultado = "Aprobado" if nota >= 70 else "Reprobado"
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Bucle for con range()

El bucle for itera sobre secuencias:

```python
for i in range(5):      # 0, 1, 2, 3, 4
    print(i)

for i in range(1, 6):   # 1, 2, 3, 4, 5
    print(i)
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: for con Listas y enumerate()

Iterar sobre listas con acceso al índice:

```python
frutas = ["manzana", "banana", "cereza"]
for i, fruta in enumerate(frutas):
    print(f"{i}: {fruta}")
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Bucle while

Ejecuta mientras la condición sea True:

```python
contador = 0
while contador < 5:
    print(contador)
    contador += 1
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: break y continue

Control del flujo dentro de bucles:

```python
# break: termina el bucle
for i in range(10):
    if i == 5:
        break

# continue: salta a siguiente iteración
for i in range(10):
    if i % 2 == 0:
        continue
    print(i)  # Solo impares
```

**Descomenta** la sección del Paso 6.

---

### Paso 7: Comprensiones de Lista

Forma compacta de crear listas:

```python
cuadrados = [x ** 2 for x in range(5)]
# [0, 1, 4, 9, 16]

pares = [x for x in range(10) if x % 2 == 0]
# [0, 2, 4, 6, 8]
```

**Descomenta** la sección del Paso 7.

---

## ✅ Verificación

Al finalizar, tu programa debe mostrar resultados para cada estructura de control.

---

## 📚 Recursos

- [Python Docs - Control Flow](https://docs.python.org/3/tutorial/controlflow.html)

---

_Anterior: [Ejercicio 02](../ejercicio-02-operadores/) | Siguiente: [Ejercicio 04](../ejercicio-04-integrador/)_
