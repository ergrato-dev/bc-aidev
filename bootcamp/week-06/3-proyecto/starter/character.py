"""
Character Module
================
Base class for all RPG characters.

Implements:
- Encapsulation with @property
- Dunder methods (__str__, __repr__, __eq__)
- Base methods for combat
"""

from __future__ import annotations
from inventory import Inventory


class Character:
    """
    Base class for all RPG characters.

    Attributes:
        name: Character's name (read-only after creation)
        health: Current health points (0 to max_health)
        max_health: Maximum health points
        level: Character level (1-100)
        defense: Base defense value
        inventory: Character's inventory

    Example:
        >>> char = Character("Hero", health=100, level=1)
        >>> print(char)
        Character: Hero (HP: 100/100, Level: 1)
    """

    def __init__(
        self,
        name: str,
        health: int = 100,
        level: int = 1,
    ) -> None:
        """
        Initialize a new Character.

        Args:
            name: Character's name (cannot be empty)
            health: Starting health points (default: 100)
            level: Starting level (default: 1)

        Raises:
            ValueError: If name is empty or health/level invalid
        """
        # TODO: Validate and set name (use _name for storage)

        # TODO: Set max_health and health using property

        # TODO: Set level using property

        # TODO: Initialize defense

        # TODO: Create inventory
        pass

    # ==========================================
    # PROPERTIES
    # ==========================================

    @property
    def name(self) -> str:
        """Get character's name (read-only)."""
        # TODO: Return _name
        pass

    @property
    def health(self) -> int:
        """Get current health points."""
        # TODO: Return _health
        pass

    @health.setter
    def health(self, value: int) -> None:
        """
        Set health with validation.

        Health is clamped between 0 and max_health.
        """
        # TODO: Clamp value between 0 and max_health
        # TODO: Set _health
        pass

    @property
    def level(self) -> int:
        """Get character level."""
        # TODO: Return _level
        pass

    @level.setter
    def level(self, value: int) -> None:
        """
        Set level with validation.

        Level must be between 1 and 100.
        """
        # TODO: Validate level is between 1 and 100
        # TODO: Set _level
        pass

    @property
    def is_alive(self) -> bool:
        """Check if character is still alive."""
        # TODO: Return True if health > 0
        pass

    # ==========================================
    # COMBAT METHODS
    # ==========================================

    def attack(self, target: Character) -> int:
        """
        Perform a basic attack on target.

        Args:
            target: The character to attack

        Returns:
            Amount of damage dealt
        """
        # TODO: Calculate base damage (e.g., 10 + level * 2)
        # TODO: Apply damage to target (target.take_damage)
        # TODO: Return damage dealt
        pass

    def take_damage(self, damage: int) -> int:
        """
        Receive damage, reduced by defense.

        Args:
            damage: Raw damage amount

        Returns:
            Actual damage taken after defense
        """
        # TODO: Calculate actual damage (damage - defense, minimum 1)
        # TODO: Reduce health
        # TODO: Return actual damage taken
        pass

    def heal(self, amount: int) -> int:
        """
        Restore health points.

        Args:
            amount: Amount to heal

        Returns:
            Actual amount healed
        """
        # TODO: Calculate actual healing (don't exceed max_health)
        # TODO: Increase health
        # TODO: Return actual healing
        pass

    def level_up(self) -> None:
        """
        Increase character level by 1.

        Also increases max_health and restores health.
        """
        # TODO: Increase level by 1
        # TODO: Increase max_health by 10
        # TODO: Restore health to max
        pass

    # ==========================================
    # DUNDER METHODS
    # ==========================================

    def __str__(self) -> str:
        """
        User-friendly string representation.

        Returns:
            String like "Character: Name (HP: 80/100, Level: 5)"
        """
        # TODO: Return formatted string
        pass

    def __repr__(self) -> str:
        """
        Developer string representation.

        Returns:
            String like "Character(name='Hero', health=80, level=5)"
        """
        # TODO: Return repr string
        pass

    def __eq__(self, other: object) -> bool:
        """
        Check equality based on name and type.

        Two characters are equal if they have the same name
        and are the same type.
        """
        # TODO: Check if other is a Character
        # TODO: Compare name and type
        pass
