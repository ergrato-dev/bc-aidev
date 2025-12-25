"""
Proyecto: Sistema de Gestión de Biblioteca
==========================================
Implementa un sistema de biblioteca usando OOP.

Instrucciones:
1. Completa cada clase siguiendo los TODOs
2. Ejecuta para probar tu implementación
3. Consulta .solution/main.py si necesitas ayuda
"""

from __future__ import annotations


# ============================================
# CLASE: Book
# Representa un libro en la biblioteca
# ============================================

class Book:
    """A book in the library."""

    def __init__(self, title: str, author: str, isbn: str):
        """
        Initialize a book.

        Args:
            title: Book title
            author: Book author
            isbn: Unique ISBN code
        """
        # TODO: Implementar inicialización
        # - Guardar title, author, isbn como atributos
        # - _available debe ser True inicialmente
        pass

    @property
    def available(self) -> bool:
        """Check if book is available for loan."""
        # TODO: Retornar _available
        pass

    @available.setter
    def available(self, value: bool) -> None:
        """Set availability status."""
        # TODO: Asignar valor a _available
        pass

    def __str__(self) -> str:
        """Human-readable representation."""
        # TODO: Retornar "Title by Author - Available/Not Available"
        pass

    def __repr__(self) -> str:
        """Developer representation."""
        # TODO: Retornar "Book(title='...', author='...', isbn='...')"
        pass


# ============================================
# CLASE: User
# Representa un usuario de la biblioteca
# ============================================

class User:
    """A library user."""

    MAX_BOOKS = 3  # Maximum books a user can borrow

    def __init__(self, name: str, user_id: str):
        """
        Initialize a user.

        Args:
            name: User name
            user_id: Unique user ID
        """
        # TODO: Implementar inicialización
        # - Guardar name, user_id
        # - _borrowed_books: lista vacía de ISBNs
        pass

    @property
    def borrowed_books(self) -> list[str]:
        """Get list of borrowed book ISBNs."""
        # TODO: Retornar copia de _borrowed_books
        pass

    @property
    def borrow_count(self) -> int:
        """Get number of borrowed books."""
        # TODO: Retornar cantidad de libros prestados
        pass

    @property
    def can_borrow(self) -> bool:
        """Check if user can borrow more books."""
        # TODO: True si tiene menos de MAX_BOOKS
        pass

    def add_book(self, isbn: str) -> None:
        """Add book ISBN to borrowed list."""
        # TODO: Agregar ISBN a _borrowed_books
        pass

    def remove_book(self, isbn: str) -> bool:
        """Remove book ISBN from borrowed list."""
        # TODO: Remover ISBN si existe, retornar True/False
        pass

    def __str__(self) -> str:
        """Human-readable representation."""
        # TODO: Retornar "Name (ID) - X books borrowed"
        pass


# ============================================
# CLASE: Library
# Gestiona libros, usuarios y préstamos
# ============================================

class Library:
    """A library that manages books and users."""

    def __init__(self, name: str):
        """
        Initialize the library.

        Args:
            name: Library name
        """
        # TODO: Implementar inicialización
        # - Guardar name
        # - _books: lista vacía de Book
        # - _users: lista vacía de User
        pass

    def add_book(self, book: Book) -> bool:
        """
        Add a book to the library.

        Args:
            book: Book to add

        Returns:
            True if added, False if ISBN already exists
        """
        # TODO: Agregar libro si ISBN no existe
        pass

    def register_user(self, user: User) -> bool:
        """
        Register a new user.

        Args:
            user: User to register

        Returns:
            True if registered, False if ID already exists
        """
        # TODO: Registrar usuario si ID no existe
        pass

    def find_book(self, isbn: str) -> Book | None:
        """Find book by ISBN."""
        # TODO: Buscar y retornar libro o None
        pass

    def find_user(self, user_id: str) -> User | None:
        """Find user by ID."""
        # TODO: Buscar y retornar usuario o None
        pass

    def loan_book(self, isbn: str, user_id: str) -> bool:
        """
        Loan a book to a user.

        Args:
            isbn: Book ISBN
            user_id: User ID

        Returns:
            True if loan successful, False otherwise
        """
        # TODO: Implementar préstamo
        # 1. Buscar libro y usuario
        # 2. Verificar que libro existe y está disponible
        # 3. Verificar que usuario existe y puede tomar más libros
        # 4. Marcar libro como no disponible
        # 5. Agregar ISBN a libros del usuario
        # 6. Retornar True si exitoso
        pass

    def return_book(self, isbn: str, user_id: str) -> bool:
        """
        Return a book from a user.

        Args:
            isbn: Book ISBN
            user_id: User ID

        Returns:
            True if return successful, False otherwise
        """
        # TODO: Implementar devolución
        # 1. Buscar libro y usuario
        # 2. Verificar que usuario tiene el libro
        # 3. Remover ISBN de libros del usuario
        # 4. Marcar libro como disponible
        # 5. Retornar True si exitoso
        pass

    def get_available_books(self) -> list[Book]:
        """Get list of available books."""
        # TODO: Retornar lista de libros disponibles
        pass

    def get_user_loans(self, user_id: str) -> list[Book]:
        """Get books currently loaned to a user."""
        # TODO: Retornar libros prestados al usuario
        pass

    def __str__(self) -> str:
        """Library summary."""
        # TODO: Retornar "Name: X books, Y users"
        pass


# ============================================
# DEMO / TESTS
# ============================================

def main():
    """Demo the library system."""
    print("=== Library System Demo ===\n")

    # Create library
    library = Library("City Library")

    # Create books
    books = [
        Book("1984", "George Orwell", "978-0-452-28423-4"),
        Book("To Kill a Mockingbird", "Harper Lee", "978-0-06-112008-4"),
        Book("The Great Gatsby", "F. Scott Fitzgerald", "978-0-7432-7356-5"),
    ]

    # Add books
    print("Added books:")
    for book in books:
        library.add_book(book)
        print(f"  - {book}")

    # Create and register users
    print("\nRegistered users:")
    users = [
        User("Alice", "U001"),
        User("Bob", "U002"),
    ]
    for user in users:
        library.register_user(user)
        print(f"  - {user}")

    # Test loan
    print("\nLoan operations:")
    isbn = "978-0-452-28423-4"
    user_id = "U001"

    result = library.loan_book(isbn, user_id)
    print(f"  Alice borrowed '1984': {result}")

    book = library.find_book(isbn)
    if book:
        status = "Available" if book.available else "Not Available"
        print(f"  {book.title} by {book.author} - {status}")

    user = library.find_user(user_id)
    if user:
        print(f"  Alice's books: {user.borrowed_books}")

    # Test return
    print("\nReturn operations:")
    result = library.return_book(isbn, user_id)
    print(f"  Alice returned '1984': {result}")

    book = library.find_book(isbn)
    if book:
        status = "Available" if book.available else "Not Available"
        print(f"  {book.title} by {book.author} - {status}")

    user = library.find_user(user_id)
    if user:
        print(f"  Alice's books: {user.borrowed_books}")

    # Summary
    print(f"\nAvailable books: {len(library.get_available_books())}")


if __name__ == "__main__":
    main()
