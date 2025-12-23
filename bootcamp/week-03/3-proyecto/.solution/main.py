"""
Proyecto: Sistema de Gestión de Biblioteca - SOLUCIÓN
=====================================================
Implementación completa del sistema de biblioteca.
"""

from __future__ import annotations


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
        self.title = title
        self.author = author
        self.isbn = isbn
        self._available = True

    @property
    def available(self) -> bool:
        """Check if book is available for loan."""
        return self._available

    @available.setter
    def available(self, value: bool) -> None:
        """Set availability status."""
        self._available = value

    def __str__(self) -> str:
        """Human-readable representation."""
        status = "Available" if self._available else "Not Available"
        return f"{self.title} by {self.author} ({self.isbn}) - {status}"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"Book(title='{self.title}', author='{self.author}', isbn='{self.isbn}')"


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
        self.name = name
        self.user_id = user_id
        self._borrowed_books: list[str] = []

    @property
    def borrowed_books(self) -> list[str]:
        """Get list of borrowed book ISBNs."""
        return self._borrowed_books.copy()

    @property
    def borrow_count(self) -> int:
        """Get number of borrowed books."""
        return len(self._borrowed_books)

    @property
    def can_borrow(self) -> bool:
        """Check if user can borrow more books."""
        return self.borrow_count < self.MAX_BOOKS

    def add_book(self, isbn: str) -> None:
        """Add book ISBN to borrowed list."""
        if isbn not in self._borrowed_books:
            self._borrowed_books.append(isbn)

    def remove_book(self, isbn: str) -> bool:
        """Remove book ISBN from borrowed list."""
        if isbn in self._borrowed_books:
            self._borrowed_books.remove(isbn)
            return True
        return False

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"{self.name} ({self.user_id}) - {self.borrow_count} books borrowed"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"User(name='{self.name}', user_id='{self.user_id}')"


class Library:
    """A library that manages books and users."""

    def __init__(self, name: str):
        """
        Initialize the library.

        Args:
            name: Library name
        """
        self.name = name
        self._books: list[Book] = []
        self._users: list[User] = []

    @property
    def book_count(self) -> int:
        """Total number of books."""
        return len(self._books)

    @property
    def user_count(self) -> int:
        """Total number of users."""
        return len(self._users)

    def add_book(self, book: Book) -> bool:
        """
        Add a book to the library.

        Args:
            book: Book to add

        Returns:
            True if added, False if ISBN already exists
        """
        if self.find_book(book.isbn) is not None:
            return False
        self._books.append(book)
        return True

    def register_user(self, user: User) -> bool:
        """
        Register a new user.

        Args:
            user: User to register

        Returns:
            True if registered, False if ID already exists
        """
        if self.find_user(user.user_id) is not None:
            return False
        self._users.append(user)
        return True

    def find_book(self, isbn: str) -> Book | None:
        """Find book by ISBN."""
        for book in self._books:
            if book.isbn == isbn:
                return book
        return None

    def find_user(self, user_id: str) -> User | None:
        """Find user by ID."""
        for user in self._users:
            if user.user_id == user_id:
                return user
        return None

    def loan_book(self, isbn: str, user_id: str) -> bool:
        """
        Loan a book to a user.

        Args:
            isbn: Book ISBN
            user_id: User ID

        Returns:
            True if loan successful, False otherwise
        """
        # Find book and user
        book = self.find_book(isbn)
        user = self.find_user(user_id)

        # Validate
        if book is None or user is None:
            return False
        if not book.available:
            return False
        if not user.can_borrow:
            return False

        # Process loan
        book.available = False
        user.add_book(isbn)
        return True

    def return_book(self, isbn: str, user_id: str) -> bool:
        """
        Return a book from a user.

        Args:
            isbn: Book ISBN
            user_id: User ID

        Returns:
            True if return successful, False otherwise
        """
        # Find book and user
        book = self.find_book(isbn)
        user = self.find_user(user_id)

        # Validate
        if book is None or user is None:
            return False
        if isbn not in user.borrowed_books:
            return False

        # Process return
        user.remove_book(isbn)
        book.available = True
        return True

    def get_available_books(self) -> list[Book]:
        """Get list of available books."""
        return [book for book in self._books if book.available]

    def get_user_loans(self, user_id: str) -> list[Book]:
        """Get books currently loaned to a user."""
        user = self.find_user(user_id)
        if user is None:
            return []
        return [
            book for book in self._books
            if book.isbn in user.borrowed_books
        ]

    def __str__(self) -> str:
        """Library summary."""
        return f"{self.name}: {self.book_count} books, {self.user_count} users"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"Library(name='{self.name}')"


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
    print(f"\nLibrary summary: {library}")


if __name__ == "__main__":
    main()
