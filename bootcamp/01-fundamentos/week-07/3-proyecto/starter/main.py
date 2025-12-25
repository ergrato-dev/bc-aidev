"""
Proyecto: Analizador de Imágenes con NumPy
==========================================
Implementa funciones de procesamiento de imágenes usando NumPy.

Las imágenes se representan como arrays 3D:
- Shape: (height, width, channels)
- dtype: uint8 (valores 0-255)
- Canales: [R, G, B]
"""

import numpy as np


# ============================================
# FUNCIONES DE UTILIDAD (ya implementadas)
# ============================================

def create_sample_image(height: int = 100, width: int = 100) -> np.ndarray:
    """
    Crea una imagen de ejemplo con un patrón de colores.

    Args:
        height: Alto de la imagen
        width: Ancho de la imagen

    Returns:
        Array numpy de shape (height, width, 3) con dtype uint8
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Cuadrante superior izquierdo: rojo
    image[:height//2, :width//2] = [255, 0, 0]

    # Cuadrante superior derecho: verde
    image[:height//2, width//2:] = [0, 255, 0]

    # Cuadrante inferior izquierdo: azul
    image[height//2:, :width//2] = [0, 0, 255]

    # Cuadrante inferior derecho: amarillo
    image[height//2:, width//2:] = [255, 255, 0]

    return image


def display_image_info(image: np.ndarray, name: str = "Image") -> None:
    """Muestra información básica de una imagen."""
    print(f"\n{name}:")
    print(f"  Shape: {image.shape}")
    print(f"  dtype: {image.dtype}")
    print(f"  Rango valores: [{image.min()}, {image.max()}]")
    print(f"  Tamaño en memoria: {image.nbytes / 1024:.2f} KB")


# ============================================
# NIVEL BÁSICO
# ============================================

def create_gradient_image(height: int, width: int) -> np.ndarray:
    """
    Crea una imagen con un gradiente horizontal de negro a blanco.

    El gradiente va de izquierdo (negro) a derecha (blanco).
    Cada columna tiene el mismo valor en todos los píxeles.

    Args:
        height: Alto de la imagen
        width: Ancho de la imagen

    Returns:
        Array numpy de shape (height, width, 3) con dtype uint8

    Example:
        >>> img = create_gradient_image(10, 256)
        >>> img[0, 0]   # Esquina izquierda
        array([0, 0, 0], dtype=uint8)
        >>> img[0, 255] # Esquina derecha
        array([255, 255, 255], dtype=uint8)
    """
    # TODO: Implementar
    # Pista: Usa np.linspace para crear valores de 0 a 255
    # Pista: Usa broadcasting para expandir a 3 canales
    pass


def get_image_stats(image: np.ndarray) -> dict:
    """
    Calcula estadísticas de la imagen por canal de color.

    Args:
        image: Array de shape (height, width, 3)

    Returns:
        Diccionario con estadísticas por canal:
        {
            'R': {'mean': float, 'std': float, 'min': int, 'max': int},
            'G': {'mean': float, 'std': float, 'min': int, 'max': int},
            'B': {'mean': float, 'std': float, 'min': int, 'max': int}
        }

    Example:
        >>> img = np.array([[[255, 0, 128]]], dtype=np.uint8)
        >>> stats = get_image_stats(img)
        >>> stats['R']['mean']
        255.0
    """
    # TODO: Implementar
    # Pista: Accede a cada canal con image[:, :, 0], image[:, :, 1], etc.
    # Pista: Usa np.mean(), np.std(), np.min(), np.max()
    pass


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen RGB a escala de grises.

    Usa la fórmula de luminosidad:
    Gray = 0.299 * R + 0.587 * G + 0.114 * B

    Args:
        image: Array de shape (height, width, 3)

    Returns:
        Array de shape (height, width) con dtype uint8

    Example:
        >>> rgb = np.array([[[255, 255, 255]]], dtype=np.uint8)
        >>> gray = to_grayscale(rgb)
        >>> gray[0, 0]
        255
    """
    # TODO: Implementar
    # Pista: Usa los pesos [0.299, 0.587, 0.114]
    # Pista: Puedes usar np.dot o multiplicación + suma
    # Pista: Convierte a uint8 al final
    pass


# ============================================
# NIVEL INTERMEDIO
# ============================================

def adjust_brightness(image: np.ndarray, value: int) -> np.ndarray:
    """
    Ajusta el brillo de la imagen.

    Args:
        image: Array de shape (height, width, 3)
        value: Valor a sumar (-255 a 255). Positivo = más brillo.

    Returns:
        Nueva imagen con brillo ajustado (dtype uint8)

    Example:
        >>> img = np.array([[[100, 100, 100]]], dtype=np.uint8)
        >>> bright = adjust_brightness(img, 50)
        >>> bright[0, 0]
        array([150, 150, 150], dtype=uint8)
    """
    # TODO: Implementar
    # Pista: Suma el valor a todos los píxeles
    # Pista: Usa np.clip para mantener valores en [0, 255]
    # Pista: Convierte a float para evitar overflow, luego a uint8
    pass


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Ajusta el contraste de la imagen.

    La fórmula es: output = (input - mean) * factor + mean

    Args:
        image: Array de shape (height, width, 3)
        factor: Factor de contraste (>1 aumenta, <1 reduce)

    Returns:
        Nueva imagen con contraste ajustado (dtype uint8)

    Example:
        >>> img = np.array([[[100, 100, 100]]], dtype=np.uint8)
        >>> high_contrast = adjust_contrast(img, 1.5)
    """
    # TODO: Implementar
    # Pista: Calcula la media de la imagen
    # Pista: Aplica la fórmula (image - mean) * factor + mean
    # Pista: Usa np.clip y convierte a uint8
    pass


def crop_image(image: np.ndarray, x: int, y: int,
               width: int, height: int) -> np.ndarray:
    """
    Recorta una región rectangular de la imagen.

    Args:
        image: Array de shape (H, W, 3)
        x: Coordenada X inicial (columna)
        y: Coordenada Y inicial (fila)
        width: Ancho del recorte
        height: Alto del recorte

    Returns:
        Imagen recortada de shape (height, width, 3)

    Example:
        >>> img = create_sample_image(100, 100)
        >>> crop = crop_image(img, 10, 10, 50, 50)
        >>> crop.shape
        (50, 50, 3)
    """
    # TODO: Implementar
    # Pista: Usa slicing [y:y+height, x:x+width, :]
    # Pista: Asegúrate de no exceder los límites de la imagen
    pass


# ============================================
# NIVEL AVANZADO
# ============================================

def flip_image(image: np.ndarray, horizontal: bool = True) -> np.ndarray:
    """
    Voltea la imagen horizontal o verticalmente.

    Args:
        image: Array de shape (height, width, 3)
        horizontal: Si True, voltea horizontalmente. Si False, verticalmente.

    Returns:
        Imagen volteada (dtype uint8)

    Example:
        >>> img = np.array([[[1, 1, 1], [2, 2, 2]]], dtype=np.uint8)
        >>> flipped = flip_image(img, horizontal=True)
        >>> flipped[0, 0]
        array([2, 2, 2], dtype=uint8)
    """
    # TODO: Implementar
    # Pista: Para horizontal usa [:, ::-1, :]
    # Pista: Para vertical usa [::-1, :, :]
    pass


def rotate_90(image: np.ndarray, clockwise: bool = True) -> np.ndarray:
    """
    Rota la imagen 90 grados.

    Args:
        image: Array de shape (height, width, 3)
        clockwise: Si True, rota en sentido horario. Si False, antihorario.

    Returns:
        Imagen rotada (dtype uint8)

    Example:
        >>> img = create_sample_image(100, 200)  # 100 alto, 200 ancho
        >>> rotated = rotate_90(img, clockwise=True)
        >>> rotated.shape
        (200, 100, 3)
    """
    # TODO: Implementar
    # Pista: Usa np.rot90()
    # Pista: k=1 para antihorario, k=-1 o k=3 para horario
    pass


def apply_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Aplica un filtro de convolución simple a la imagen.

    Implementación simplificada que aplica el kernel a cada canal.
    No maneja bordes (reduce tamaño de imagen).

    Args:
        image: Array de shape (height, width, 3)
        kernel: Array 2D de shape (k, k) donde k es impar

    Returns:
        Imagen filtrada (puede tener shape reducido)

    Example:
        # Kernel de blur simple 3x3
        >>> blur_kernel = np.ones((3, 3)) / 9
        >>> blurred = apply_filter(img, blur_kernel)
    """
    # TODO: Implementar (opcional - más difícil)
    # Pista: Itera sobre cada posición válida
    # Pista: Extrae la región del tamaño del kernel
    # Pista: Multiplica por el kernel y suma
    # Nota: Esta es una implementación básica, no optimizada
    pass


# ============================================
# PROGRAMA PRINCIPAL
# ============================================

def main():
    """Función principal para probar las implementaciones."""
    print("=" * 60)
    print("PROYECTO: Analizador de Imágenes con NumPy")
    print("=" * 60)

    # Crear imagen de ejemplo
    sample = create_sample_image(100, 100)
    display_image_info(sample, "Imagen de ejemplo")

    # --- Probar funciones básicas ---
    print("\n" + "=" * 60)
    print("NIVEL BÁSICO")
    print("=" * 60)

    # Test: create_gradient_image
    print("\n[Test] create_gradient_image")
    gradient = create_gradient_image(50, 256)
    if gradient is not None:
        display_image_info(gradient, "Gradiente")
        print(f"  Esquina izquierda: {gradient[0, 0]}")
        print(f"  Esquina derecha: {gradient[0, -1]}")
    else:
        print("  ❌ No implementada")

    # Test: get_image_stats
    print("\n[Test] get_image_stats")
    stats = get_image_stats(sample)
    if stats is not None:
        for channel, data in stats.items():
            print(f"  {channel}: mean={data['mean']:.1f}, "
                  f"std={data['std']:.1f}, "
                  f"min={data['min']}, max={data['max']}")
    else:
        print("  ❌ No implementada")

    # Test: to_grayscale
    print("\n[Test] to_grayscale")
    gray = to_grayscale(sample)
    if gray is not None:
        print(f"  Shape original: {sample.shape}")
        print(f"  Shape grayscale: {gray.shape}")
        print(f"  Rango valores: [{gray.min()}, {gray.max()}]")
    else:
        print("  ❌ No implementada")

    # --- Probar funciones intermedias ---
    print("\n" + "=" * 60)
    print("NIVEL INTERMEDIO")
    print("=" * 60)

    # Test: adjust_brightness
    print("\n[Test] adjust_brightness")
    bright = adjust_brightness(sample, 50)
    if bright is not None:
        print(f"  Original mean: {sample.mean():.1f}")
        print(f"  Bright (+50) mean: {bright.mean():.1f}")
    else:
        print("  ❌ No implementada")

    # Test: adjust_contrast
    print("\n[Test] adjust_contrast")
    contrast = adjust_contrast(sample, 1.5)
    if contrast is not None:
        print(f"  Original std: {sample.std():.1f}")
        print(f"  High contrast (1.5) std: {contrast.std():.1f}")
    else:
        print("  ❌ No implementada")

    # Test: crop_image
    print("\n[Test] crop_image")
    cropped = crop_image(sample, 10, 10, 30, 30)
    if cropped is not None:
        print(f"  Original shape: {sample.shape}")
        print(f"  Cropped shape: {cropped.shape}")
    else:
        print("  ❌ No implementada")

    # --- Probar funciones avanzadas ---
    print("\n" + "=" * 60)
    print("NIVEL AVANZADO")
    print("=" * 60)

    # Test: flip_image
    print("\n[Test] flip_image")
    flipped_h = flip_image(sample, horizontal=True)
    flipped_v = flip_image(sample, horizontal=False)
    if flipped_h is not None:
        print(f"  Flip horizontal: shape {flipped_h.shape}")
        print(f"  Flip vertical: shape {flipped_v.shape}")
    else:
        print("  ❌ No implementada")

    # Test: rotate_90
    print("\n[Test] rotate_90")
    rect_img = create_sample_image(50, 100)  # Imagen rectangular
    rotated = rotate_90(rect_img, clockwise=True)
    if rotated is not None:
        print(f"  Original shape: {rect_img.shape}")
        print(f"  Rotated 90° CW: {rotated.shape}")
    else:
        print("  ❌ No implementada")

    # Test: apply_filter
    print("\n[Test] apply_filter")
    blur_kernel = np.ones((3, 3)) / 9
    filtered = apply_filter(sample, blur_kernel)
    if filtered is not None:
        print(f"  Original shape: {sample.shape}")
        print(f"  Filtered shape: {filtered.shape}")
    else:
        print("  ❌ No implementada (opcional)")

    print("\n" + "=" * 60)
    print("✅ Tests completados")
    print("=" * 60)


if __name__ == "__main__":
    main()
