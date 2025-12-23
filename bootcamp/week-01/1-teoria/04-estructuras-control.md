# 🔀 Estructuras de Control de Flujo

## 🎯 Objetivos

- Dominar las estructuras condicionales (if/elif/else)
- Implementar bucles for y while
- Usar break, continue y else en bucles
- Aplicar comprensiones de lista básicas

---

## 📋 Contenido

### 1. Condicionales: if / elif / else

Permiten ejecutar código basado en condiciones:

```python
age = 18

if age < 18:
    print("Menor de edad")
elif age == 18:
    print("Justo 18 años")
else:
    print("Mayor de edad")
```

#### Sintaxis

```python
if condicion:
    # Código si condición es True
    # IMPORTANTE: La indentación (4 espacios) es obligatoria
elif otra_condicion:
    # Código si otra_condicion es True
else:
    # Código si ninguna condición es True
```

#### Ejemplos Prácticos

```python
# Clasificación de modelo ML
accuracy = 0.85

if accuracy >= 0.9:
    status = "Excelente"
elif accuracy >= 0.8:
    status = "Bueno"
elif accuracy >= 0.7:
    status = "Aceptable"
else:
    status = "Necesita mejora"

print(f"Modelo: {status}")  # Modelo: Bueno
```

#### Condicional en una línea (Ternario)

```python
# Forma tradicional
if score >= 70:
    result = "Aprobado"
else:
    result = "Reprobado"

# Forma ternaria (una línea)
result = "Aprobado" if score >= 70 else "Reprobado"
```

#### Condiciones múltiples

```python
age = 25
has_id = True
has_money = True

# AND: todas las condiciones deben ser True
if age >= 18 and has_id and has_money:
    print("Puede comprar")

# OR: al menos una condición True
if has_id or has_money:
    print("Tiene al menos algo")

# Combinado
if (age >= 18 and has_id) or is_vip:
    print("Acceso permitido")
```

---

### 2. Bucle for

Itera sobre una secuencia (lista, string, range, etc.):

```python
# Iterar sobre una lista
fruits = ["manzana", "banana", "cereza"]
for fruit in fruits:
    print(fruit)

# Iterar sobre un string
for char in "Python":
    print(char)

# Iterar sobre un rango de números
for i in range(5):      # 0, 1, 2, 3, 4
    print(i)

for i in range(1, 6):   # 1, 2, 3, 4, 5
    print(i)

for i in range(0, 10, 2):  # 0, 2, 4, 6, 8 (paso de 2)
    print(i)
```

#### Patrones comunes con for

```python
# Enumerar: obtener índice y valor
names = ["Ana", "Bob", "Carlos"]
for index, name in enumerate(names):
    print(f"{index}: {name}")
# 0: Ana
# 1: Bob
# 2: Carlos

# Iterar dos listas en paralelo
names = ["Ana", "Bob"]
ages = [25, 30]
for name, age in zip(names, ages):
    print(f"{name} tiene {age} años")

# Iterar diccionarios
data = {"name": "Ana", "age": 25}
for key, value in data.items():
    print(f"{key}: {value}")
```

---

### 3. Bucle while

Ejecuta mientras la condición sea True:

```python
count = 0
while count < 5:
    print(count)
    count += 1  # ¡IMPORTANTE! Evita bucles infinitos
```

#### ⚠️ Cuidado con bucles infinitos

```python
# ❌ PELIGRO: Bucle infinito
# while True:
#     print("Infinito!")

# ✅ CORRECTO: Con condición de salida
while True:
    user_input = input("Escribe 'salir' para terminar: ")
    if user_input == "salir":
        break
```

#### Ejemplo: Entrenamiento de modelo (simulado)

```python
# Simular entrenamiento hasta convergencia
loss = 1.0
epoch = 0
max_epochs = 100
threshold = 0.01

while loss > threshold and epoch < max_epochs:
    # Simular reducción de loss
    loss = loss * 0.9
    epoch += 1
    print(f"Epoch {epoch}: loss = {loss:.4f}")

print(f"Entrenamiento finalizado en {epoch} epochs")
```

---

### 4. Control de Bucles: break, continue, else

#### break: Termina el bucle

```python
# Buscar un elemento
numbers = [1, 3, 5, 7, 9, 2, 4]
for num in numbers:
    if num % 2 == 0:  # Encontrar primer número par
        print(f"Encontrado: {num}")
        break
```

#### continue: Salta a la siguiente iteración

```python
# Procesar solo números positivos
numbers = [1, -2, 3, -4, 5]
for num in numbers:
    if num < 0:
        continue  # Salta los negativos
    print(f"Procesando: {num}")
# Output: 1, 3, 5
```

#### else en bucles: Se ejecuta si NO hubo break

```python
# Buscar elemento
target = 10
numbers = [1, 2, 3, 4, 5]

for num in numbers:
    if num == target:
        print("¡Encontrado!")
        break
else:
    print("No encontrado")  # Se ejecuta porque no hubo break
```

---

### 5. Comprensiones de Lista (List Comprehensions)

Forma compacta de crear listas:

```python
# Forma tradicional
squares = []
for x in range(5):
    squares.append(x ** 2)

# Comprensión de lista (Pythonic)
squares = [x ** 2 for x in range(5)]
# [0, 1, 4, 9, 16]

# Con condición
evens = [x for x in range(10) if x % 2 == 0]
# [0, 2, 4, 6, 8]

# Transformar elementos
names = ["ana", "bob", "carlos"]
capitalized = [name.upper() for name in names]
# ["ANA", "BOB", "CARLOS"]
```

#### Cuándo usar comprensiones

```python
# ✅ USA comprensiones para transformaciones simples
doubled = [x * 2 for x in numbers]

# ❌ NO uses comprensiones para lógica compleja
# Esto es difícil de leer:
# result = [x if x > 0 else -x if x < 0 else 0 for x in data if x is not None]

# ✅ Mejor usar bucle tradicional para lógica compleja
result = []
for x in data:
    if x is None:
        continue
    if x > 0:
        result.append(x)
    elif x < 0:
        result.append(-x)
    else:
        result.append(0)
```

---

### 6. Patrones para IA/ML

#### Iterar sobre batches

```python
data = list(range(100))  # 100 elementos
batch_size = 32

for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]
    print(f"Batch {i // batch_size}: {len(batch)} elementos")
```

#### Épocas de entrenamiento

```python
epochs = 10
for epoch in range(epochs):
    # Simular entrenamiento
    loss = 1.0 / (epoch + 1)  # Loss que decrece
    accuracy = 1 - loss

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Acc: {accuracy:.4f}")

    # Early stopping
    if accuracy > 0.95:
        print("Early stopping: objetivo alcanzado")
        break
```

#### Validación de input

```python
while True:
    try:
        value = int(input("Ingresa un número (1-10): "))
        if 1 <= value <= 10:
            break
        print("Debe estar entre 1 y 10")
    except ValueError:
        print("Debe ser un número entero")

print(f"Valor válido: {value}")
```

---

### 7. Ejercicio Mental

¿Cuál es la salida?

```python
# Ejercicio 1
for i in range(3):
    for j in range(3):
        if i == j:
            continue
        print(f"({i},{j})", end=" ")
    print()

# Ejercicio 2
numbers = [1, 2, 3, 4, 5]
result = [x * 2 for x in numbers if x > 2]
print(result)

# Ejercicio 3
count = 0
while count < 10:
    count += 1
    if count == 5:
        break
else:
    print("Bucle completado")
print(f"count = {count}")
```

<details>
<summary>Ver respuestas</summary>

```
# Ejercicio 1
(0,1) (0,2)
(1,0) (1,2)
(2,0) (2,1)

# Ejercicio 2
[6, 8, 10]

# Ejercicio 3
count = 5
# (No imprime "Bucle completado" porque hubo break)
```

</details>

---

## 📊 Resumen Visual

```
        ESTRUCTURAS DE CONTROL EN PYTHON
        =================================

    CONDICIONALES              BUCLES
    ─────────────              ──────

    if condicion:              for item in secuencia:
        # código                   # código
    elif otra:
        # código               while condicion:
    else:                          # código
        # código

    CONTROL DE BUCLES          COMPRENSIONES
    ─────────────────          ─────────────

    break    → Salir           [expr for item in seq]
    continue → Siguiente       [expr for item in seq if cond]
    else     → Si no break
```

---

## ✅ Checklist de Verificación

- [ ] Domino if/elif/else
- [ ] Entiendo el operador ternario
- [ ] Uso for con range, listas y enumerate
- [ ] Implemento while con condiciones de salida
- [ ] Aplico break, continue y else en bucles
- [ ] Creo comprensiones de lista básicas
- [ ] Evito bucles infinitos

---

## 📚 Recursos Adicionales

- [Python Docs - Control Flow](https://docs.python.org/3/tutorial/controlflow.html)
- [Real Python - Conditionals](https://realpython.com/python-conditional-statements/)
- [Real Python - For Loops](https://realpython.com/python-for-loop/)

---

_Anterior: [03 - Operadores](03-operadores.md) | Siguiente: [Prácticas](../2-practicas/)_
