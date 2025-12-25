"""
Character Classes Module
========================
Specialized character classes that inherit from Character.

Classes:
- Warrior: High health, rage-based attacks
- Mage: Magic attacks, mana-based spells
- Archer: Ranged attacks, arrow management
"""

from __future__ import annotations
from character import Character


class Warrior(Character):
    """
    Warrior class - melee fighter with rage mechanic.

    Additional Attributes:
        rage: Rage points (0-100), builds during combat
        base_damage: Higher base damage than other classes

    Special Ability:
        berserk(): Consume rage for massive damage boost
    """

    def __init__(
        self,
        name: str,
        health: int = 120,
        level: int = 1,
        rage: int = 0,
    ) -> None:
        """
        Initialize a Warrior.

        Args:
            name: Warrior's name
            health: Starting health (default: 120, higher than base)
            level: Starting level
            rage: Starting rage points (default: 0)
        """
        # TODO: Call parent __init__ with super()
        # TODO: Set rage using property
        # TODO: Increase base defense (warriors are tougher)
        pass

    @property
    def rage(self) -> int:
        """Get current rage points."""
        # TODO: Return _rage
        pass

    @rage.setter
    def rage(self, value: int) -> None:
        """Set rage, clamped between 0 and 100."""
        # TODO: Clamp value between 0 and 100
        # TODO: Set _rage
        pass

    def attack(self, target: Character) -> int:
        """
        Warrior attack - gains rage on hit.

        Damage = base_damage + level * 3 + rage_bonus
        Gains 10 rage per attack.
        """
        # TODO: Calculate damage with warrior bonuses
        # TODO: Apply damage to target
        # TODO: Gain rage (10 points)
        # TODO: Return damage dealt
        pass

    def berserk(self) -> int:
        """
        Consume all rage for a damage multiplier.

        Returns the bonus damage for next attack.
        Requires at least 50 rage.
        """
        # TODO: Check if rage >= 50
        # TODO: Calculate bonus (rage * 0.5)
        # TODO: Reset rage to 0
        # TODO: Return bonus damage
        pass

    def __str__(self) -> str:
        """Include rage in string representation."""
        # TODO: Return "Warrior: Name (HP: x/y, Level: z, Rage: r)"
        pass


class Mage(Character):
    """
    Mage class - magic user with mana-based spells.

    Additional Attributes:
        mana: Mana points for casting spells (0-100)
        spells: Dictionary of available spells

    Special Ability:
        cast_spell(spell_name, target): Cast a spell consuming mana
    """

    # Class attribute: available spells with (damage, mana_cost)
    SPELLS: dict[str, tuple[int, int]] = {
        "Fireball": (30, 20),
        "Ice Shard": (20, 15),
        "Lightning": (40, 30),
        "Heal": (-25, 25),  # Negative damage = healing
    }

    def __init__(
        self,
        name: str,
        health: int = 80,
        level: int = 1,
        mana: int = 100,
    ) -> None:
        """
        Initialize a Mage.

        Args:
            name: Mage's name
            health: Starting health (default: 80, lower than base)
            level: Starting level
            mana: Starting mana points (default: 100)
        """
        # TODO: Call parent __init__
        # TODO: Set mana using property
        # TODO: Set max_mana
        pass

    @property
    def mana(self) -> int:
        """Get current mana points."""
        # TODO: Return _mana
        pass

    @mana.setter
    def mana(self, value: int) -> None:
        """Set mana, clamped between 0 and max_mana."""
        # TODO: Clamp value
        # TODO: Set _mana
        pass

    def attack(self, target: Character) -> int:
        """
        Mage basic attack - weaker than physical classes.

        Damage = 5 + level * 2 (mages rely on spells)
        """
        # TODO: Calculate magic damage
        # TODO: Apply to target
        # TODO: Return damage
        pass

    def cast_spell(self, spell_name: str, target: Character) -> int:
        """
        Cast a spell on target.

        Args:
            spell_name: Name of spell to cast
            target: Character to target (can be self for Heal)

        Returns:
            Damage dealt (or healing done if negative)

        Raises:
            ValueError: If spell doesn't exist or not enough mana
        """
        # TODO: Check if spell exists
        # TODO: Get spell damage and cost
        # TODO: Check if enough mana
        # TODO: Consume mana
        # TODO: Apply effect (damage or heal)
        # TODO: Return effect amount
        pass

    def meditate(self) -> int:
        """
        Restore mana by meditating.

        Restores 20 mana points.
        """
        # TODO: Calculate restoration
        # TODO: Increase mana
        # TODO: Return amount restored
        pass

    def __str__(self) -> str:
        """Include mana in string representation."""
        # TODO: Return "Mage: Name (HP: x/y, Level: z, Mana: m)"
        pass


class Archer(Character):
    """
    Archer class - ranged fighter with arrow management.

    Additional Attributes:
        arrows: Current arrow count
        max_arrows: Maximum arrows that can be carried

    Special Ability:
        shoot(target): Ranged attack with critical chance
    """

    def __init__(
        self,
        name: str,
        health: int = 90,
        level: int = 1,
        arrows: int = 20,
    ) -> None:
        """
        Initialize an Archer.

        Args:
            name: Archer's name
            health: Starting health (default: 90)
            level: Starting level
            arrows: Starting arrow count (default: 20)
        """
        # TODO: Call parent __init__
        # TODO: Set max_arrows
        # TODO: Set arrows using property
        pass

    @property
    def arrows(self) -> int:
        """Get current arrow count."""
        # TODO: Return _arrows
        pass

    @arrows.setter
    def arrows(self, value: int) -> None:
        """Set arrows, clamped between 0 and max_arrows."""
        # TODO: Clamp value
        # TODO: Set _arrows
        pass

    def attack(self, target: Character) -> int:
        """
        Archer basic attack - uses arrows.

        Damage = 12 + level * 2
        Consumes 1 arrow per attack.
        Returns 0 if no arrows.
        """
        # TODO: Check if arrows > 0
        # TODO: Consume arrow
        # TODO: Calculate and apply damage
        # TODO: Return damage (or 0 if no arrows)
        pass

    def shoot(self, target: Character) -> int:
        """
        Special ranged attack with critical chance.

        25% chance to deal double damage.
        Consumes 2 arrows.
        """
        # TODO: Check if arrows >= 2
        # TODO: Consume 2 arrows
        # TODO: Calculate damage with crit chance
        # TODO: Apply damage
        # TODO: Return damage
        pass

    def resupply(self, amount: int = 10) -> int:
        """
        Restore arrows.

        Args:
            amount: Arrows to add (default: 10)

        Returns:
            Actual arrows added
        """
        # TODO: Calculate actual addition
        # TODO: Add arrows
        # TODO: Return amount added
        pass

    def __str__(self) -> str:
        """Include arrows in string representation."""
        # TODO: Return "Archer: Name (HP: x/y, Level: z, Arrows: a)"
        pass
