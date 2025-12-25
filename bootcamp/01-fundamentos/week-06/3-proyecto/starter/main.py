"""
Sistema RPG - Main Entry Point
==============================
Programa principal para probar el sistema de personajes RPG.

Ejecuta este archivo después de completar los módulos.
"""

from character import Character
from classes import Warrior, Mage, Archer
from inventory import Inventory, Weapon, Potion, Armor
from combat import Combat


def main():
    """Main function to demonstrate the RPG system."""
    print("=" * 50)
    print("🎮 RPG CHARACTER SYSTEM")
    print("=" * 50)
    print()

    # ==========================================
    # 1. CREATE CHARACTERS
    # ==========================================
    print("--- Creating Characters ---")

    # TODO: Create a Warrior named "Conan" with 120 health
    warrior = None

    # TODO: Create a Mage named "Gandalf" with 80 health and 100 mana
    mage = None

    # TODO: Create an Archer named "Legolas" with 90 health and 20 arrows
    archer = None

    # Print characters
    print(warrior)
    print(mage)
    print(archer)
    print()

    # ==========================================
    # 2. TEST PROPERTIES
    # ==========================================
    print("--- Testing Properties ---")

    # TODO: Try to set warrior's health to -10 (should raise error or clamp)
    # TODO: Try to set mage's mana to 150 (should clamp to 100)

    print()

    # ==========================================
    # 3. TEST ATTACKS
    # ==========================================
    print("--- Combat Demonstration ---")

    # TODO: Warrior attacks Mage
    # damage = warrior.attack(mage)
    # print(f"Warrior dealt {damage} damage to Mage!")
    # print(f"Mage health: {mage.health}")

    print()

    # ==========================================
    # 4. TEST SPECIAL ABILITIES
    # ==========================================
    print("--- Special Abilities ---")

    # TODO: Test Warrior's berserk ability
    # warrior.berserk()

    # TODO: Test Mage's cast_spell ability
    # mage.cast_spell("Fireball", warrior)

    # TODO: Test Archer's shoot ability
    # archer.shoot(mage)

    print()

    # ==========================================
    # 5. TEST INVENTORY
    # ==========================================
    print("--- Inventory System ---")

    # TODO: Add items to warrior's inventory
    # warrior.inventory.add(Weapon("Steel Sword", damage=15, value=100))
    # warrior.inventory.add(Potion("Health Potion", healing=30, value=25))
    # warrior.inventory.add(Armor("Iron Shield", defense=10, value=75))

    # TODO: Print inventory size and contents
    # print(f"Inventory size: {len(warrior.inventory)}")
    # for item in warrior.inventory:
    #     print(f"  - {item}")

    print()

    # ==========================================
    # 6. TEST COMBAT SYSTEM
    # ==========================================
    print("--- Full Combat ---")

    # TODO: Create and run a combat between warrior and mage
    # combat = Combat(warrior, mage)
    # winner = combat.run()
    # print(f"Winner: {winner.name}")

    print()
    print("=" * 50)
    print("🎮 DEMO COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
