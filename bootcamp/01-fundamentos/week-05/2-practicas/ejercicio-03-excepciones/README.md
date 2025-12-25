# ⚠️ Ejercicio 03: Excepciones

## 🎯 Objetivos

- Usar try/except/else/finally
- Capturar excepciones específicas
- Crear excepciones personalizadas
- Re-lanzar excepciones

---

## 📋 Instrucciones

1. Abre `starter/main.py`
2. Descomenta cada paso y ejecútalo
3. Observa el flujo de ejecución

---

## Paso 1: try/except Básico

Capturar errores para evitar crashes.

```python
# Sin manejo - crashea
value = int("no es número")  # ValueError!

# Con manejo
try:
    value = int("no es número")
except ValueError:
    print("No es un número válido")
    value = 0
```

**Descomenta** el Paso 1 en `starter/main.py`.

---

## Paso 2: Acceder al Objeto Excepción

Obtener información del error.

```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
    print(f"Tipo: {type(e).__name__}")
```

**Descomenta** el Paso 2 y observa la información.

---

## Paso 3: Múltiples Excepciones

Manejar diferentes tipos de errores.

```python
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        print("No se puede dividir por cero")
        return None
    except TypeError:
        print("Los valores deben ser números")
        return None
```

**Descomenta** el Paso 3 en `starter/main.py`.

---

## Paso 4: else y finally

Control de flujo completo.

```python
try:
    file = open('data.txt', 'r')
except FileNotFoundError:
    print("Archivo no encontrado")
else:
    # Solo si NO hubo error
    content = file.read()
    file.close()
    print("Archivo leído correctamente")
finally:
    # SIEMPRE se ejecuta
    print("Operación completada")
```

**Descomenta** el Paso 4 y observa el flujo.

---

## Paso 5: Lanzar Excepciones (raise)

Generar errores cuando sea necesario.

```python
def validate_age(age):
    if not isinstance(age, int):
        raise TypeError("La edad debe ser un entero")
    if age < 0:
        raise ValueError("La edad no puede ser negativa")
    if age > 150:
        raise ValueError("Edad inválida")
    return True
```

**Descomenta** el Paso 5 en `starter/main.py`.

---

## Paso 6: Excepciones Personalizadas

Crear tus propias excepciones.

```python
class ValidationError(Exception):
    """Error de validación de datos."""
    pass

class EmailError(ValidationError):
    """Error específico de email."""
    def __init__(self, email, message="Email inválido"):
        self.email = email
        super().__init__(f"{message}: {email}")
```

**Descomenta** el Paso 6 y prueba las excepciones custom.

---

## Paso 7: Re-lanzar Excepciones

Propagar errores después de procesarlos.

```python
import logging

def process_file(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"Archivo no encontrado: {path}")
        raise  # Re-lanza la misma excepción
```

**Descomenta** el Paso 7 en `starter/main.py`.

---

## ✅ Verificación

Al completar, deberías entender:

- [ ] Diferencia entre except específico y genérico
- [ ] Cuándo usar else vs finally
- [ ] Cómo crear excepciones personalizadas
- [ ] Cuándo re-lanzar excepciones

---

## 🔗 Siguiente

[Ejercicio 04: Logging](../ejercicio-04-logging/)
