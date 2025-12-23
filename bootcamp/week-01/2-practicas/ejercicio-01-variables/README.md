# 📦 Ejercicio 01: Variables y Tipos de Datos

## 🎯 Objetivos

- Crear y asignar variables
- Identificar tipos de datos básicos
- Realizar conversiones de tipo
- Aplicar naming conventions

---

## 📋 Instrucciones

Abre el archivo `starter/main.py` y sigue los pasos descomentando el código indicado.

> 💡 **Nota**: Este es un ejercicio guiado. Sigue las instrucciones paso a paso.

---

### Paso 1: Crear Variables Básicas

Las variables en Python se crean con una simple asignación:

```python
nombre = "valor"
numero = 42
```

**Abre `starter/main.py`** y descomenta la sección del Paso 1.

---

### Paso 2: Tipos de Datos Numéricos

Python tiene dos tipos numéricos principales:

- `int`: números enteros
- `float`: números decimales

```python
entero = 42
decimal = 3.14
```

**Descomenta** la sección del Paso 2.

---

### Paso 3: Strings y f-strings

Los strings pueden definirse con comillas simples o dobles. Los **f-strings** permiten insertar variables:

```python
nombre = "Python"
mensaje = f"Hola, {nombre}!"
```

**Descomenta** la sección del Paso 3.

---

### Paso 4: Booleanos y None

Los booleanos representan valores lógicos (`True`/`False`). `None` representa ausencia de valor:

```python
activo = True
resultado = None
```

**Descomenta** la sección del Paso 4.

---

### Paso 5: Verificar Tipos con type()

La función `type()` devuelve el tipo de una variable:

```python
print(type(42))  # <class 'int'>
```

**Descomenta** la sección del Paso 5.

---

### Paso 6: Conversión de Tipos (Type Casting)

Puedes convertir entre tipos usando funciones como `int()`, `str()`, `float()`:

```python
numero_str = "42"
numero_int = int(numero_str)  # 42
```

**Descomenta** la sección del Paso 6.

---

### Paso 7: Naming Conventions

Python usa **snake_case** para variables y funciones:

```python
# ✅ Correcto
user_name = "Ana"
total_count = 100

# ❌ Incorrecto
userName = "Ana"  # camelCase es para JavaScript
```

**Descomenta** la sección del Paso 7.

---

## ✅ Verificación

Al finalizar, tu programa debe mostrar:

```
--- Paso 1: Variables Básicas ---
Nombre: Ana
Edad: 25
Altura: 1.65

--- Paso 2: Tipos Numéricos ---
Entero: 42
Decimal: 3.14159
Notación científica: 0.0025

--- Paso 3: Strings y f-strings ---
Mensaje: ¡Hola, Python!
Multilinea:
Este es un texto
en múltiples líneas

--- Paso 4: Booleanos y None ---
Activo: True
Tiene error: False
Resultado: None

--- Paso 5: Verificar Tipos ---
Tipo de 42: <class 'int'>
Tipo de 3.14: <class 'float'>
Tipo de 'hola': <class 'str'>
Tipo de True: <class 'bool'>
Tipo de None: <class 'NoneType'>

--- Paso 6: Conversiones ---
String '42' a int: 42
Int 42 a string: '42'
Float 3.99 a int: 3
String '3.14' a float: 3.14

--- Paso 7: Naming Conventions ---
user_name: Ana (snake_case ✓)
MAX_RETRIES: 3 (UPPER_SNAKE_CASE para constantes ✓)
```

---

## 📚 Recursos

- [Python Docs - Built-in Types](https://docs.python.org/3/library/stdtypes.html)
- [PEP 8 - Style Guide](https://pep8.org/)

---

_Siguiente: [Ejercicio 02 - Operadores](../ejercicio-02-operadores/)_
