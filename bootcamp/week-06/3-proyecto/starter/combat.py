"""
Combat Module
=============
Turn-based combat system between characters.

Classes:
- Combat: Manages combat between two characters
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from character import Character


class Combat:
    """
    Turn-based combat system.

    Manages combat between two characters, tracking turns,
    damage dealt, and determining the winner.

    Attributes:
        char1: First combatant
        char2: Second combatant
        turn: Current turn number
        log: List of combat events
    """

    def __init__(self, char1: Character, char2: Character) -> None:
        """
        Initialize combat between two characters.

        Args:
            char1: First combatant
            char2: Second combatant
        """
        # TODO: Store characters
        # TODO: Initialize turn counter to 0
        # TODO: Initialize empty log list
        pass

    @property
    def is_over(self) -> bool:
        """Check if combat is finished (one character defeated)."""
        # TODO: Return True if either character is not alive
        pass

    @property
    def winner(self) -> Character | None:
        """
        Get the winner of combat.

        Returns:
            Winning character, or None if combat ongoing
        """
        # TODO: Return char1 if char2 not alive
        # TODO: Return char2 if char1 not alive
        # TODO: Return None if both alive
        pass

    def execute_turn(
        self,
        attacker: Character,
        defender: Character,
    ) -> str:
        """
        Execute a single combat turn.

        Args:
            attacker: Character performing the attack
            defender: Character receiving the attack

        Returns:
            Description of what happened
        """
        # TODO: Increment turn counter
        # TODO: Attacker attacks defender
        # TODO: Create log message
        # TODO: Add to log
        # TODO: Return message
        pass

    def run(self, max_turns: int = 100) -> Character | None:
        """
        Run combat until one character is defeated.

        Characters alternate attacking.

        Args:
            max_turns: Maximum turns before draw (default: 100)

        Returns:
            Winning character, or None if draw
        """
        # TODO: Loop while combat not over and turns < max
        # TODO: char1 attacks char2
        # TODO: If combat over, break
        # TODO: char2 attacks char1
        # TODO: Return winner (or None)
        pass

    def get_log(self) -> list[str]:
        """Return combat log."""
        # TODO: Return log list
        pass

    def print_log(self) -> None:
        """Print all combat events."""
        # TODO: Print each log entry
        pass

    def __str__(self) -> str:
        """Combat status representation."""
        # TODO: Return "Combat: Char1 vs Char2 (Turn X)"
        pass
