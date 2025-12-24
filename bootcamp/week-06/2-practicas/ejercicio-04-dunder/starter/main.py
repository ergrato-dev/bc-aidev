"""
Ejercicio 04: Dunder Methods
============================
Aprende a implementar métodos especiales para personalizar tus clases.

Instrucciones:
1. Lee cada sección
2. Descomenta el código
3. Ejecuta y observa los resultados
"""

from typing import Iterator

# ============================================
# PASO 1: __str__ y __repr__
# ============================================
print('--- Paso 1: __str__ y __repr__ ---')

# __str__: representación legible para usuarios (print)
# __repr__: representación técnica para desarrolladores (debug)

# Descomenta las siguientes líneas:
# class Point:
#     """A point in 2D space."""
#
#     def __init__(self, x: float, y: float):
#         self.x = x
#         self.y = y
#
#     def __str__(self) -> str:
#         """User-friendly representation."""
#         return f"Point at ({self.x}, {self.y})"
#
#     def __repr__(self) -> str:
#         """Developer representation (should be unambiguous)."""
#         return f"Point(x={self.x}, y={self.y})"
#
# p = Point(3, 4)
# print(f"str: {str(p)}")    # Uses __str__
# print(f"repr: {repr(p)}")  # Uses __repr__

print()

# ============================================
# PASO 2: __eq__ para Igualdad
# ============================================
print('--- Paso 2: __eq__ ---')

# __eq__ define cuándo dos objetos son "iguales".
# Sin __eq__, Python compara por identidad (is), no por valor.

# Descomenta las siguientes líneas:
# class Point:
#     """Point with equality comparison."""
#
#     def __init__(self, x: float, y: float):
#         self.x = x
#         self.y = y
#
#     def __repr__(self) -> str:
#         return f"Point({self.x}, {self.y})"
#
#     def __eq__(self, other: object) -> bool:
#         """Two points are equal if they have same coordinates."""
#         if not isinstance(other, Point):
#             return NotImplemented
#         return self.x == other.x and self.y == other.y
#
# p1 = Point(3, 4)
# p2 = Point(3, 4)
# p3 = Point(1, 2)
#
# print(f"p1 == p2: {p1 == p2}")  # True (same coordinates)
# print(f"p1 == p3: {p1 == p3}")  # False (different coordinates)
# print(f"p1 is p2: {p1 is p2}")  # False (different objects)

print()

# ============================================
# PASO 3: Operadores de Comparación
# ============================================
print('--- Paso 3: Ordenamiento ---')

# __lt__ (less than) permite usar sorted() y min/max.
# Python puede derivar otros operadores de __eq__ y __lt__.

# Descomenta las siguientes líneas:
# import math
#
# class Point:
#     """Point with comparison operators."""
#
#     def __init__(self, x: float, y: float):
#         self.x = x
#         self.y = y
#
#     def __repr__(self) -> str:
#         return f"Point({self.x}, {self.y})"
#
#     @property
#     def distance(self) -> float:
#         """Distance from origin."""
#         return math.sqrt(self.x ** 2 + self.y ** 2)
#
#     def __eq__(self, other: object) -> bool:
#         if not isinstance(other, Point):
#             return NotImplemented
#         return self.x == other.x and self.y == other.y
#
#     def __lt__(self, other: "Point") -> bool:
#         """Compare by distance from origin."""
#         if not isinstance(other, Point):
#             return NotImplemented
#         return self.distance < other.distance
#
# points = [Point(3, 4), Point(1, 1), Point(0, 5)]
# print(f"Original: {points}")
# print(f"Sorted: {sorted(points)}")  # Sorts by distance

print()

# ============================================
# PASO 4: Operadores Aritméticos
# ============================================
print('--- Paso 4: Operadores Aritméticos ---')

# __add__, __sub__, __mul__ permiten usar +, -, * con objetos.
# Deben retornar una nueva instancia, no modificar la actual.

# Descomenta las siguientes líneas:
# class Vector:
#     """2D Vector with arithmetic operators."""
#
#     def __init__(self, x: float, y: float):
#         self.x = x
#         self.y = y
#
#     def __repr__(self) -> str:
#         return f"Vector({self.x}, {self.y})"
#
#     def __add__(self, other: "Vector") -> "Vector":
#         """Add two vectors."""
#         if not isinstance(other, Vector):
#             return NotImplemented
#         return Vector(self.x + other.x, self.y + other.y)
#
#     def __sub__(self, other: "Vector") -> "Vector":
#         """Subtract two vectors."""
#         if not isinstance(other, Vector):
#             return NotImplemented
#         return Vector(self.x - other.x, self.y - other.y)
#
#     def __mul__(self, scalar: float) -> "Vector":
#         """Multiply vector by scalar."""
#         return Vector(self.x * scalar, self.y * scalar)
#
#     def __rmul__(self, scalar: float) -> "Vector":
#         """Support scalar * vector (reverse multiplication)."""
#         return self.__mul__(scalar)
#
# v1 = Vector(1, 2)
# v2 = Vector(3, 4)
#
# print(f"v1 + v2 = {v1 + v2}")
# print(f"v1 * 3 = {v1 * 3}")

print()

# ============================================
# PASO 5: Colección Personalizada
# ============================================
print('--- Paso 5: Colección Personalizada ---')

# __len__, __getitem__, __iter__ hacen que tu clase se comporte como lista.

# Descomenta las siguientes líneas:
# class Playlist:
#     """A playlist that behaves like a list."""
#
#     def __init__(self, name: str):
#         self.name = name
#         self._songs: list[str] = []
#
#     def add(self, song: str) -> None:
#         """Add a song to the playlist."""
#         self._songs.append(song)
#
#     def __len__(self) -> int:
#         """Return number of songs."""
#         return len(self._songs)
#
#     def __getitem__(self, index: int) -> str:
#         """Get song by index."""
#         return self._songs[index]
#
#     def __iter__(self) -> Iterator[str]:
#         """Iterate over songs."""
#         return iter(self._songs)
#
#     def __contains__(self, song: str) -> bool:
#         """Check if song is in playlist."""
#         return song in self._songs
#
#     def __repr__(self) -> str:
#         return f"Playlist({self.name!r}, {len(self)} songs)"
#
# # Create playlist
# rock = Playlist("Classic Rock")
# rock.add("Bohemian Rhapsody")
# rock.add("Stairway to Heaven")
# rock.add("Hotel California")
#
# # Use like a list
# print(f"Playlist length: {len(rock)}")
# print(f"First song: {rock[0]}")
# print(f"Songs: {', '.join(rock)}")

print()

# ============================================
# 🎯 DESAFÍO EXTRA (Opcional)
# ============================================
# Crea una clase Money con:
# - amount (float), currency (str)
# - __str__: "$100.00 USD"
# - __eq__: compara amount y currency
# - __add__: suma Money del mismo currency (raise error si diferente)
# - __lt__: compara por amount (solo mismo currency)
#
# class Money:
#     def __init__(self, amount: float, currency: str = "USD"):
#         pass  # Tu código aquí
#
#     def __str__(self) -> str:
#         pass  # Tu código aquí
#
#     def __eq__(self, other: object) -> bool:
#         pass  # Tu código aquí
#
#     def __add__(self, other: "Money") -> "Money":
#         pass  # Tu código aquí
