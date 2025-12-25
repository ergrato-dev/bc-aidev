# ⚠️ Manejo de Excepciones

## 🎯 Objetivos

- Entender el sistema de excepciones de Python
- Usar `try/except/else/finally` correctamente
- Crear excepciones personalizadas
- Aplicar mejores prácticas de manejo de errores

---

## 1. Fundamentos

### ¿Qué es una excepción?

Una excepción es un evento que interrumpe el flujo normal del programa cuando ocurre un error.

```python
# Esto genera una excepción
result = 10 / 0  # ZeroDivisionError

# Esto también
numbers = [1, 2, 3]
print(numbers[10])  # IndexError
```

### Sintaxis básica

```python
try:
    # Código que puede fallar
    result = risky_operation()
except SomeException:
    # Manejo del error
    handle_error()
```

---

## 2. try/except/else/finally

### Estructura completa

```python
try:
    # Código que puede lanzar excepciones
    file = open('data.txt', 'r')
    data = file.read()
except FileNotFoundError:
    # Se ejecuta si ocurre FileNotFoundError
    print("Archivo no encontrado")
except PermissionError:
    # Se ejecuta si ocurre PermissionError
    print("Sin permisos")
else:
    # Se ejecuta SOLO si NO hubo excepciones
    print(f"Leídos {len(data)} caracteres")
finally:
    # Se ejecuta SIEMPRE (haya o no excepción)
    print("Operación completada")
```

### Flujo de ejecución

```
try → (error) → except → finally
try → (ok) → else → finally
```

### Acceder al objeto excepción

```python
try:
    result = int("no es número")
except ValueError as e:
    print(f"Error: {e}")           # Error: invalid literal...
    print(f"Tipo: {type(e)}")      # Tipo: <class 'ValueError'>
    print(f"Args: {e.args}")       # Args: ("invalid literal...",)
```

---

## 3. Excepciones Comunes

| Excepción           | Causa                         |
| ------------------- | ----------------------------- |
| `ValueError`        | Valor incorrecto para el tipo |
| `TypeError`         | Operación con tipo incorrecto |
| `KeyError`          | Clave no existe en dict       |
| `IndexError`        | Índice fuera de rango         |
| `FileNotFoundError` | Archivo no existe             |
| `PermissionError`   | Sin permisos                  |
| `ZeroDivisionError` | División por cero             |
| `AttributeError`    | Atributo no existe            |
| `ImportError`       | Error al importar módulo      |
| `RuntimeError`      | Error genérico de ejecución   |

### Jerarquía de excepciones

```
BaseException
├── SystemExit
├── KeyboardInterrupt
├── GeneratorExit
└── Exception
    ├── StopIteration
    ├── ArithmeticError
    │   ├── ZeroDivisionError
    │   └── OverflowError
    ├── LookupError
    │   ├── IndexError
    │   └── KeyError
    ├── OSError
    │   ├── FileNotFoundError
    │   ├── PermissionError
    │   └── FileExistsError
    ├── ValueError
    ├── TypeError
    └── ...
```

---

## 4. Múltiples Excepciones

### Capturar varias en un bloque

```python
try:
    process_data(data)
except (ValueError, TypeError) as e:
    print(f"Error de datos: {e}")
```

### Capturar en bloques separados

```python
try:
    value = int(input("Número: "))
    result = 100 / value
except ValueError:
    print("Debe ser un número")
except ZeroDivisionError:
    print("No puede ser cero")
```

### Capturar excepción base

```python
try:
    risky_operation()
except FileNotFoundError:
    # Específico primero
    print("Archivo no encontrado")
except OSError:
    # Más general después
    print("Error de sistema de archivos")
except Exception as e:
    # Catch-all (último recurso)
    print(f"Error inesperado: {e}")
```

---

## 5. Lanzar Excepciones

### raise

```python
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("El divisor no puede ser cero")
    return a / b

# Uso
try:
    result = divide(10, 0)
except ValueError as e:
    print(e)  # El divisor no puede ser cero
```

### Re-lanzar excepciones

```python
def process_file(path: str) -> str:
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Archivo no encontrado: {path}")
        raise  # Re-lanza la misma excepción

# O con excepción diferente
def process_data(data: str) -> dict:
    try:
        return parse(data)
    except ParseError as e:
        raise ValueError(f"Datos inválidos: {e}") from e
```

### Cadena de excepciones

```python
try:
    process()
except ValueError as e:
    # `from e` preserva la excepción original
    raise RuntimeError("Falló el proceso") from e

# Al imprimir el traceback verás:
# ValueError: ...
# The above exception was the direct cause of:
# RuntimeError: Falló el proceso
```

---

## 6. Excepciones Personalizadas

### Crear excepción simple

```python
class ValidationError(Exception):
    """Error de validación de datos."""
    pass


def validate_age(age: int) -> None:
    if age < 0:
        raise ValidationError("La edad no puede ser negativa")
    if age > 150:
        raise ValidationError("Edad inválida")
```

### Con atributos adicionales

```python
class APIError(Exception):
    """Error de la API."""

    def __init__(self, message: str, status_code: int, endpoint: str):
        super().__init__(message)
        self.status_code = status_code
        self.endpoint = endpoint

    def __str__(self) -> str:
        return f"[{self.status_code}] {self.endpoint}: {super().__str__()}"


# Uso
try:
    raise APIError("Not Found", 404, "/api/users/123")
except APIError as e:
    print(e)                  # [404] /api/users/123: Not Found
    print(e.status_code)      # 404
```

### Jerarquía de excepciones

```python
class AppError(Exception):
    """Base para errores de la aplicación."""
    pass


class ValidationError(AppError):
    """Error de validación."""
    pass


class DatabaseError(AppError):
    """Error de base de datos."""
    pass


class ConnectionError(DatabaseError):
    """Error de conexión a BD."""
    pass


# Uso: capturar por jerarquía
try:
    operation()
except ConnectionError:
    print("Problema de conexión")
except DatabaseError:
    print("Error de base de datos")
except AppError:
    print("Error de aplicación")
```

---

## 7. Mejores Prácticas

### ✅ Capturar excepciones específicas

```python
# ✅ BIEN - Específico
try:
    value = data['key']
except KeyError:
    value = 'default'

# ❌ MAL - Captura todo
try:
    value = data['key']
except:
    value = 'default'
```

### ✅ No silenciar excepciones

```python
# ❌ MAL - Silencia el error
try:
    process()
except Exception:
    pass  # ¿Qué pasó?

# ✅ BIEN - Al menos loguear
import logging
logger = logging.getLogger(__name__)

try:
    process()
except Exception as e:
    logger.exception("Error en process")  # Loguea con traceback
```

### ✅ Usar else para código sin error

```python
# ✅ BIEN - Separación clara
try:
    file = open('data.txt')
except FileNotFoundError:
    data = []
else:
    data = file.read()
    file.close()
```

### ✅ finally para limpieza

```python
# ✅ BIEN - Limpieza garantizada
connection = None
try:
    connection = create_connection()
    process(connection)
except ConnectionError:
    print("Error de conexión")
finally:
    if connection:
        connection.close()  # Siempre se ejecuta
```

### ✅ Context managers cuando sea posible

```python
# ✅ MEJOR - Context manager maneja todo
with open('data.txt') as f:
    data = f.read()
# No necesitas try/finally para cerrar
```

---

## 8. Patrones Comunes

### Valor por defecto con EAFP

```python
# EAFP: Easier to Ask Forgiveness than Permission
def get_value(data: dict, key: str, default=None):
    try:
        return data[key]
    except KeyError:
        return default


# vs LBYL: Look Before You Leap
def get_value_lbyl(data: dict, key: str, default=None):
    if key in data:
        return data[key]
    return default
```

### Retry con excepciones

```python
import time
from typing import TypeVar, Callable

T = TypeVar('T')

def retry(
    func: Callable[[], T],
    max_attempts: int = 3,
    delay: float = 1.0
) -> T:
    """Reintenta una función si falla."""
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            print(f"Intento {attempt + 1} falló: {e}")
            if attempt < max_attempts - 1:
                time.sleep(delay)

    raise last_exception


# Uso
result = retry(lambda: fetch_data(url), max_attempts=3)
```

### Validación con excepciones

```python
from dataclasses import dataclass

class ValidationError(Exception):
    pass


@dataclass
class User:
    name: str
    email: str
    age: int

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        errors = []

        if not self.name:
            errors.append("Nombre requerido")
        if '@' not in self.email:
            errors.append("Email inválido")
        if self.age < 0:
            errors.append("Edad inválida")

        if errors:
            raise ValidationError(", ".join(errors))


# Uso
try:
    user = User("", "invalid", -5)
except ValidationError as e:
    print(f"Errores: {e}")
```

---

## 9. assert (para desarrollo)

```python
def calculate_average(numbers: list[float]) -> float:
    assert len(numbers) > 0, "Lista vacía"
    return sum(numbers) / len(numbers)

# assert se puede deshabilitar con python -O
# NO usar para validación en producción
```

---

## 📚 Resumen

| Bloque    | Cuándo se ejecuta          |
| --------- | -------------------------- |
| `try`     | Siempre (código principal) |
| `except`  | Si hay excepción           |
| `else`    | Si NO hay excepción        |
| `finally` | SIEMPRE                    |

| Acción    | Código                    |
| --------- | ------------------------- |
| Lanzar    | `raise ValueError("msg")` |
| Re-lanzar | `raise`                   |
| Encadenar | `raise NewError() from e` |
| Capturar  | `except Error as e:`      |

---

## ✅ Checklist

- [ ] Capturar excepciones específicas
- [ ] No usar `except:` sin tipo
- [ ] No silenciar excepciones con `pass`
- [ ] Usar `else` para código post-try exitoso
- [ ] Usar `finally` para limpieza
- [ ] Preferir context managers cuando sea posible
- [ ] Crear excepciones custom para tu dominio

---

_Siguiente: [Logging](04-logging.md)_
