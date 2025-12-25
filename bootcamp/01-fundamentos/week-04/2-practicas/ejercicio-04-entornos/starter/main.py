# ============================================
# EJERCICIO 04: Entornos Virtuales y pip
# ============================================
# Aprenderás a crear entornos virtuales,
# instalar paquetes y gestionar dependencias.
# ============================================

print('=== Ejercicio 04: Entornos Virtuales ===')
print()

# ============================================
# PASO 1: Verificar Python
# ============================================
print('--- Paso 1: Verificar Python ---')
print()
print('Ejecuta en terminal:')
print('  python --version')
print('  python -m venv --help')
print()

import sys
print(f'Python actual: {sys.version}')
print(f'Ejecutable: {sys.executable}')
print()

# ============================================
# PASO 2: Crear Entorno Virtual
# ============================================
print('--- Paso 2: Crear Entorno Virtual ---')
print()
print('Navega a este directorio y ejecuta:')
print()
print('  cd bootcamp/week-04/2-practicas/ejercicio-04-entornos')
print('  python -m venv .venv')
print()
print('Esto crea la carpeta .venv/ con:')
print('  .venv/')
print('  ├── bin/          # Linux/Mac: activate, python, pip')
print('  ├── Scripts/      # Windows: activate.bat, python.exe')
print('  ├── lib/')
print('  └── pyvenv.cfg')
print()

# ============================================
# PASO 3: Activar Entorno
# ============================================
print('--- Paso 3: Activar Entorno ---')
print()
print('Linux/Mac:')
print('  source .venv/bin/activate')
print()
print('Windows PowerShell:')
print('  .venv\\Scripts\\Activate.ps1')
print()
print('Windows CMD:')
print('  .venv\\Scripts\\activate.bat')
print()
print('Verificar activacion:')
print('  which python   # Linux/Mac')
print('  where python   # Windows')
print()
print('El prompt cambia a: (.venv) user@machine:~$')
print()

# ============================================
# PASO 4: Verificar Entorno Limpio
# ============================================
print('--- Paso 4: Verificar Entorno Limpio ---')
print()
print('Ejecuta (con entorno activado):')
print('  pip list')
print()
print('Deberia mostrar solo:')
print('  pip')
print('  setuptools (quiza)')
print()

# ============================================
# PASO 5: Instalar Paquetes
# ============================================
print('--- Paso 5: Instalar Paquetes ---')
print()
print('Instala algunos paquetes de ejemplo:')
print()
print('  pip install requests')
print('  pip install python-dotenv')
print()
print('Instalar version especifica:')
print('  pip install requests==2.31.0')
print()
print('Instalar multiples:')
print('  pip install numpy pandas')
print()
print('Ver informacion de un paquete:')
print('  pip show requests')
print()

# ============================================
# PASO 6: Listar Paquetes Instalados
# ============================================
print('--- Paso 6: Listar Paquetes ---')
print()
print('Ver todos los paquetes:')
print('  pip list')
print()
print('Ver paquetes desactualizados:')
print('  pip list --outdated')
print()
print('Formato para requirements:')
print('  pip freeze')
print()

# ============================================
# PASO 7: Crear requirements.txt
# ============================================
print('--- Paso 7: Crear requirements.txt ---')
print()
print('Exportar dependencias:')
print('  pip freeze > requirements.txt')
print()
print('El archivo tendra algo como:')
print('  certifi==2023.7.22')
print('  charset-normalizer==3.2.0')
print('  requests==2.31.0')
print('  ...')
print()

# ============================================
# PASO 8: Probar Paquetes Instalados
# ============================================
print('--- Paso 8: Probar Paquetes ---')
print()
print('Ejecuta este script con el entorno activado')
print('y descomenta el siguiente codigo:')
print()

# Descomenta despues de instalar requests:
# import requests
#
# response = requests.get('https://api.github.com')
# print(f'Status: {response.status_code}')
# print(f'Headers: {dict(response.headers)["Content-Type"]}')
#
# # Ver de donde viene el modulo
# print(f'requests location: {requests.__file__}')

print()

# ============================================
# PASO 9: Desactivar Entorno
# ============================================
print('--- Paso 9: Desactivar Entorno ---')
print()
print('Para salir del entorno virtual:')
print('  deactivate')
print()
print('El prompt vuelve a la normalidad.')
print('which python mostrara el Python global.')
print()

# ============================================
# PASO 10: Recrear desde requirements.txt
# ============================================
print('--- Paso 10: Recrear Entorno ---')
print()
print('Simula clonar el proyecto:')
print()
print('  # Eliminar entorno actual')
print('  rm -rf .venv')
print()
print('  # Crear nuevo entorno')
print('  python -m venv .venv')
print('  source .venv/bin/activate')
print()
print('  # Instalar dependencias')
print('  pip install -r requirements.txt')
print()
print('  # Verificar')
print('  pip list')
print()

# ============================================
# PASO 11: Agregar .venv a .gitignore
# ============================================
print('--- Paso 11: Configurar .gitignore ---')
print()
print('Nunca commitees el entorno virtual!')
print('Agrega a .gitignore:')
print()
print('  # Entornos virtuales')
print('  .venv/')
print('  venv/')
print('  ENV/')
print()
print('  # Cache Python')
print('  __pycache__/')
print('  *.pyc')
print()

# ============================================
# RESUMEN DE COMANDOS
# ============================================
print('=== Resumen de Comandos ===')
print()
print('CREAR Y ACTIVAR:')
print('  python -m venv .venv')
print('  source .venv/bin/activate    # Linux/Mac')
print('  .venv\\Scripts\\activate       # Windows')
print()
print('GESTIONAR PAQUETES:')
print('  pip install <package>')
print('  pip install <package>==1.0.0')
print('  pip uninstall <package>')
print('  pip list')
print('  pip show <package>')
print()
print('DEPENDENCIAS:')
print('  pip freeze > requirements.txt')
print('  pip install -r requirements.txt')
print()
print('DESACTIVAR:')
print('  deactivate')
print()

# ============================================
# EJERCICIO PRACTICO
# ============================================
print('=== Ejercicio Practico ===')
print()
print('1. Crea un entorno virtual en este directorio')
print('2. Activa el entorno')
print('3. Instala: requests, python-dotenv')
print('4. Genera requirements.txt')
print('5. Ejecuta este script y descomenta Paso 8')
print('6. Elimina .venv y recrea desde requirements.txt')
print('7. Verifica que todo funciona')
print()
print('Felicidades si completaste todos los pasos!')
