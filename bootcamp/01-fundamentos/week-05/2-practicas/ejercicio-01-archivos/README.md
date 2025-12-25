# 📄 Ejercicio 01: Operaciones con Archivos

## 🎯 Objetivos

- Leer y escribir archivos de texto
- Usar context managers (`with`)
- Trabajar con `pathlib`
- Manejar encoding UTF-8

---

## 📋 Instrucciones

1. Abre `starter/main.py`
2. Descomenta cada paso y ejecútalo
3. Observa los resultados en consola y archivos creados

---

## Paso 1: Crear y Escribir Archivos

Usamos `open()` con mode `'w'` para crear/sobrescribir archivos.

```python
from pathlib import Path

# Crear archivo de texto
output_path = Path('output/mensaje.txt')
output_path.parent.mkdir(exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write('¡Hola Mundo!\n')
    f.write('Segunda línea\n')
```

**Descomenta** el Paso 1 en `starter/main.py` y ejecútalo.

---

## Paso 2: Leer Archivos

Diferentes métodos para leer contenido.

```python
# read() - Todo el contenido
with open(output_path, 'r', encoding='utf-8') as f:
    content = f.read()
    print(content)

# readlines() - Lista de líneas
with open(output_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print(lines)  # ['¡Hola Mundo!\n', 'Segunda línea\n']
```

**Descomenta** el Paso 2 y observa las diferencias.

---

## Paso 3: Iterar Líneas (Archivos Grandes)

Para archivos grandes, itera directamente sobre el objeto file.

```python
# Eficiente en memoria
with open(output_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        print(f"Línea {i}: {line.strip()}")
```

**Descomenta** el Paso 3 en `starter/main.py`.

---

## Paso 4: Append (Añadir al Final)

Mode `'a'` añade sin sobrescribir.

```python
# Añadir líneas
with open(output_path, 'a', encoding='utf-8') as f:
    f.write('Tercera línea (append)\n')
    f.write('Cuarta línea (append)\n')
```

**Descomenta** el Paso 4 y verifica que se añadieron las líneas.

---

## Paso 5: Pathlib - Lectura/Escritura Directa

`pathlib` ofrece métodos convenientes.

```python
from pathlib import Path

path = Path('output/quick.txt')

# Escribir
path.write_text('Contenido rápido\n', encoding='utf-8')

# Leer
content = path.read_text(encoding='utf-8')
print(content)
```

**Descomenta** el Paso 5 en `starter/main.py`.

---

## Paso 6: Información de Archivos

Usar `pathlib` para obtener metadatos.

```python
from pathlib import Path

path = Path('output/mensaje.txt')

print(f"Nombre: {path.name}")
print(f"Extensión: {path.suffix}")
print(f"Padre: {path.parent}")
print(f"Existe: {path.exists()}")
print(f"Es archivo: {path.is_file()}")
print(f"Tamaño: {path.stat().st_size} bytes")
```

**Descomenta** el Paso 6 y explora los atributos.

---

## Paso 7: Listar Archivos

Usar `glob` para encontrar archivos.

```python
from pathlib import Path

output_dir = Path('output')

# Todos los archivos
print("Todos los archivos:")
for file in output_dir.iterdir():
    print(f"  {file}")

# Solo .txt
print("\nArchivos .txt:")
for txt_file in output_dir.glob('*.txt'):
    print(f"  {txt_file}")
```

**Descomenta** el Paso 7 en `starter/main.py`.

---

## ✅ Verificación

Al completar, deberías tener:

- [ ] Carpeta `output/` creada
- [ ] `mensaje.txt` con 4 líneas
- [ ] `quick.txt` creado con pathlib
- [ ] Entendimiento de modos r/w/a

---

## 🔗 Siguiente

[Ejercicio 02: Formatos de Datos](../ejercicio-02-formatos/)
