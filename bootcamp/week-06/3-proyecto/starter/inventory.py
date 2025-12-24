"""
Inventory Module
================
Item classes and inventory system with collection behavior.

Classes:
- Item: Base class for all items
- Weapon: Items that deal damage
- Armor: Items that provide defense
- Potion: Consumable items for healing
- Inventory: Collection of items with list-like behavior
"""

from __future__ import annotations
from typing import Iterator


class Item:
    """
    Base class for all items.

    Attributes:
        name: Item name
        value: Gold value of the item
    """

    def __init__(self, name: str, value: int = 0) -> None:
        """
        Initialize an Item.

        Args:
            name: Name of the item
            value: Gold value (default: 0)
        """
        # TODO: Set name and value
        pass

    def __str__(self) -> str:
        """Return item name and value."""
        # TODO: Return "Name (Value: X gold)"
        pass

    def __repr__(self) -> str:
        """Return developer representation."""
        # TODO: Return "Item(name='X', value=Y)"
        pass

    def __eq__(self, other: object) -> bool:
        """Items are equal if same name and type."""
        # TODO: Check type and compare names
        pass


class Weapon(Item):
    """
    Weapon item that increases attack damage.

    Additional Attributes:
        damage: Bonus damage when equipped
    """

    def __init__(self, name: str, damage: int, value: int = 0) -> None:
        """
        Initialize a Weapon.

        Args:
            name: Weapon name
            damage: Bonus damage
            value: Gold value
        """
        # TODO: Call parent __init__
        # TODO: Set damage
        pass

    def __str__(self) -> str:
        """Include damage in representation."""
        # TODO: Return "Name (Damage: +X, Value: Y gold)"
        pass

    def __repr__(self) -> str:
        """Developer representation."""
        # TODO: Return "Weapon(name='X', damage=Y, value=Z)"
        pass


class Armor(Item):
    """
    Armor item that increases defense.

    Additional Attributes:
        defense: Bonus defense when equipped
    """

    def __init__(self, name: str, defense: int, value: int = 0) -> None:
        """
        Initialize Armor.

        Args:
            name: Armor name
            defense: Bonus defense
            value: Gold value
        """
        # TODO: Call parent __init__
        # TODO: Set defense
        pass

    def __str__(self) -> str:
        """Include defense in representation."""
        # TODO: Return "Name (Defense: +X, Value: Y gold)"
        pass

    def __repr__(self) -> str:
        """Developer representation."""
        # TODO: Return "Armor(name='X', defense=Y, value=Z)"
        pass


class Potion(Item):
    """
    Consumable potion for healing.

    Additional Attributes:
        healing: Health restored when used
    """

    def __init__(self, name: str, healing: int, value: int = 0) -> None:
        """
        Initialize a Potion.

        Args:
            name: Potion name
            healing: Health restored on use
            value: Gold value
        """
        # TODO: Call parent __init__
        # TODO: Set healing
        pass

    def __str__(self) -> str:
        """Include healing in representation."""
        # TODO: Return "Name (Healing: +X HP, Value: Y gold)"
        pass

    def __repr__(self) -> str:
        """Developer representation."""
        # TODO: Return "Potion(name='X', healing=Y, value=Z)"
        pass


class Inventory:
    """
    Character inventory with collection behavior.

    Implements __len__, __getitem__, __iter__, __contains__
    to behave like a list.

    Attributes:
        capacity: Maximum number of items
        items: List of items
    """

    def __init__(self, capacity: int = 20) -> None:
        """
        Initialize an empty inventory.

        Args:
            capacity: Maximum items (default: 20)
        """
        # TODO: Set capacity
        # TODO: Initialize empty items list
        pass

    def add(self, item: Item) -> bool:
        """
        Add an item to inventory.

        Args:
            item: Item to add

        Returns:
            True if added, False if inventory full
        """
        # TODO: Check if inventory is full
        # TODO: Add item if space available
        # TODO: Return success status
        pass

    def remove(self, item: Item) -> bool:
        """
        Remove an item from inventory.

        Args:
            item: Item to remove

        Returns:
            True if removed, False if not found
        """
        # TODO: Try to remove item
        # TODO: Return success status
        pass

    def get_by_name(self, name: str) -> Item | None:
        """
        Find item by name.

        Args:
            name: Item name to search

        Returns:
            Item if found, None otherwise
        """
        # TODO: Search for item by name
        # TODO: Return item or None
        pass

    @property
    def total_value(self) -> int:
        """Calculate total gold value of all items."""
        # TODO: Sum all item values
        pass

    # ==========================================
    # DUNDER METHODS FOR COLLECTION BEHAVIOR
    # ==========================================

    def __len__(self) -> int:
        """Return number of items in inventory."""
        # TODO: Return length of items list
        pass

    def __getitem__(self, index: int) -> Item:
        """Get item by index."""
        # TODO: Return item at index
        pass

    def __iter__(self) -> Iterator[Item]:
        """Iterate over items."""
        # TODO: Return iterator over items
        pass

    def __contains__(self, item: Item) -> bool:
        """Check if item is in inventory."""
        # TODO: Return if item in items
        pass

    def __str__(self) -> str:
        """String representation of inventory."""
        # TODO: Return "Inventory (X/Y items)"
        pass

    def __repr__(self) -> str:
        """Developer representation."""
        # TODO: Return "Inventory(capacity=X, items=[...])"
        pass
