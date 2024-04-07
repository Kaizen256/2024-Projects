from tabulate import tabulate
import os
import json
import uuid
import sys
import random
from enemies import generate_enemy_data
import time

#ANSI escape codes for color
UNDERLINE = "\033[4m"
BOLD = "\033[1m"
ORANGE = "\033[38;5;208m"
RED = "\033[31m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
BLACK = "\033[30m"
RESET = "\033[0m"
BLINK = "\033[5m"
BGRED = "\033[41m"

class Player:
    def __init__(self, name, max_health=100):
        self.name = name
        self.health = max_health
        self.max_health = max_health
        self.attack_power = 10
        self.gold = 0
        self.level = 1
        self.story_index = 1
        self.weapon = {"name": "Bronze Sword", "rarity": "Bronze", "attack_power_bonus": 5, "upgrade_cost": 10}
        self.xp = 0
        self.xp_needed = 50
        self.critical_blessing = 0
        self.misc = 0

    def attack(self, enemy):
        if self.critical_blessing == 1:
            critical = random.randint(1, 10)
        elif self.critical_blessing == 2:
            critical = random.randint(1, 4)
        if self.critical_blessing != 0 and critical == 1:
            damage = random.randint((self.attack_power + self.weapon["attack_power_bonus"]) * 2, (self.attack_power + self.weapon["attack_power_bonus"]) * 2 + 15)
            time.sleep(0.1)
            print_slow(BOLD + ORANGE + f"{self.name} attacked {enemy.name} and dealt {damage} damage with a critical strike!" + RESET)
        else:
            damage = random.randint(self.attack_power + self.weapon["attack_power_bonus"], self.attack_power + self.weapon["attack_power_bonus"] + 15)
            time.sleep(0.1)
            print_slow(BOLD + PURPLE + f"{self.name} attacked {enemy.name} and dealt {damage} damage!" + RESET)
            print()
        enemy.take_damage(damage)

    def strong_attack(self, enemy):
        if self.critical_blessing == 1:
            critical = random.randint(1, 10)
        elif self.critical_blessing == 2:
            critical = random.randint(1, 4)
        if self.critical_blessing != 0 and critical == 1:
            damage = random.randint((self.attack_power + self.weapon["attack_power_bonus"]) * 4, (self.attack_power + self.weapon["attack_power_bonus"]) * 4 + 15)
            time.sleep(0.1)
            print_slow(BOLD + BGRED + BLACK + f"{self.name} attacked {enemy.name} with a strong attack and dealt {damage} damage with a critical strike!" + RESET)
        else:
            damage = random.randint((self.attack_power + self.weapon["attack_power_bonus"]) * 2, (self.attack_power + self.weapon["attack_power_bonus"]) * 2 + 15)
            time.sleep(0.1)
            print_slow(BOLD + ORANGE + f"{self.name} attacked {enemy.name} with a strong attack and dealt {damage} damage!" + RESET)
            print()
        enemy.take_damage(damage)

    def heal(self):
        max_healing = self.max_health // 3
        min_healing = self.max_health // 5
        healing_amount = random.randint(min_healing, max_healing)

        if self.health + healing_amount > self.max_health:
            healing_amount = self.max_health - self.health
        self.health += healing_amount
        time.sleep(0.1)
        print_slow(BOLD + GREEN + f"{self.name} heals for {healing_amount} health." + RESET)

    def update_max_health(self, amount):
        self.max_health += amount
        self.health = self.max_health

    def upgrade_weapon(self):
        if self.gold >= self.weapon["upgrade_cost"]:
            if self.weapon["rarity"] == "Bronze":
                self.weapon["attack_power_bonus"] = 10
                self.weapon["rarity"] = "Silver"
                self.weapon["name"] = "Silver Sword"
                self.gold -= self.weapon["upgrade_cost"]
                self.weapon["upgrade_cost"] = 25
            elif self.weapon["rarity"] == "Silver":
                self.weapon["attack_power_bonus"] = 30
                self.weapon["rarity"] = "Gold"
                self.weapon["name"] = "Golden Sword"
                self.gold -= self.weapon["upgrade_cost"]
                self.weapon["upgrade_cost"] = 100
            elif self.weapon["rarity"] == "Gold":
                self.weapon["attack_power_bonus"] = 75
                self.weapon["rarity"] = "Platinum"
                self.weapon["name"] = "Platinum Sword"
                self.gold -= self.weapon["upgrade_cost"]
                self.weapon["upgrade_cost"] = 500
            elif self.weapon["rarity"] == "Platinum":
                self.weapon["attack_power_bonus"] = 100
                self.weapon["rarity"] = "Diamond"
                self.weapon["name"] = "Diamond Sword"
                self.gold -= self.weapon["upgrade_cost"]
                self.weapon["upgrade_cost"] = 1000
            elif self.weapon["rarity"] == "Diamond":
                self.weapon["attack_power_bonus"] = 250
                self.weapon["rarity"] = "Adamantium"
                self.weapon["name"] = "Adamantium Sword"
                self.gold -= self.weapon["upgrade_cost"]
                self.weapon["upgrade_cost"] = 2500
            elif self.weapon["rarity"] == "Adamantium":
                self.weapon["attack_power_bonus"] = 500
                self.weapon["rarity"] = "Vibranium"
                self.weapon["name"] = "Vibranium Sword"
                self.gold -= self.weapon["upgrade_cost"]
                self.weapon["upgrade_cost"] = 5000
            elif self.weapon["rarity"] == "Vibranium":
                self.weapon["attack_power_bonus"] = 1500
                self.weapon["rarity"] = "Ethernium"
                self.weapon["name"] = "Ethernium Sword"
                self.gold -= self.weapon["upgrade_cost"]
                self.weapon["upgrade_cost"] = 10000
            elif self.weapon["rarity"] == "Ethernium":
                self.weapon["attack_power_bonus"] = 10000
                self.weapon["rarity"] = "Aurorium"
                self.weapon["name"] = "Aurorium Sword"
                self.gold -= self.weapon["upgrade_cost"]
                self.weapon["upgrade_cost"] = 100000

            print_slow(GREEN + "Weapon upgraded successfully!" + RESET)
        else:
            print_slow("Insufficient gold to upgrade weapon.")

    def take_damage(self, damage):
        self.health -= damage

    def level_up(self):
        if self.xp >= self.xp_needed:
            self.xp -= self.xp_needed
            self.xp_needed = self.xp_needed * 3
            self.attack_power += 10 * self.level
            self.max_health += 25 * self.level
            self.level += 1
            print()
            print_slow(BOLD + YELLOW + f"Congratulations! You have leveled up to level {self.level}!" + RESET)

class Enemy:
    def __init__(self, name, health, attack_power, attacks):
        self.name = name
        self.health = health
        self.max_health = health
        self.attack_power = attack_power
        self.attacks = attacks

    def attack_player(self, player):
        attack = random.choice(self.attacks)
        if "damage" in attack:
            damage = attack["damage"]
            if "telegraphed" in attack:
                print_slow(BOLD + RED + f"{self.name} is charging up for a powerful attack!" + RESET)
                dodge_direction = input(BLUE + "Dodge 'left' or 'right' " + RESET).strip().lower()
                if dodge_direction != 'left' and dodge_direction != 'right':
                    print_slow("The Attack lands!")
                else:
                    roll_two = random.randint(0,1)
                    if roll_two == 0:
                        print_slow(ORANGE + f"You attempt to evade {dodge_direction}" + RESET)
                        time.sleep(2)
                        print_slow(RED + BOLD + f"You fail and the attack lands!\n" + RESET)
                    elif roll_two == 1:
                        print_slow(ORANGE + f"You attempt to evade {dodge_direction}" + RESET)
                        time.sleep(2)
                        print_slow(GREEN + BOLD + f"You successfully evade the attack!\b" + RESET)
                        return
            player.take_damage(damage)
            print_slow(BOLD + RED + f"{self.name} used {attack['name']} on {player.name} and dealt {damage} damage!" + RESET)
        elif "healing" in attack:
            healing = attack["healing"]
            if self.health + healing > self.max_health:
                healing = self.max_health - self.health
            self.health += healing
            print_slow(BOLD + YELLOW + f"{self.name} used {attack['name']} and healed for {healing} health!" + RESET)
    def take_damage(self, damage):
        self.health -= damage


def main():
    filename = "save.json"
    is_saved_game = ""
    player_name = input(BLUE + "Enter your name: " + RESET)
    if os.path.exists(filename):
        try:
            saved_data = load_game(filename)
            for player_id, player_data in saved_data.items():
                if player_data['name'] == player_name:
                    save_choice = input("Do you wish to continue where you last left off? Y/N ").lower()
                    if save_choice == 'y':
                        player = Player(player_data['name'])
                        player.__dict__.update(player_data)
                        print_slow("Loaded saved game.")
                        is_saved_game = "y"
                        break

                    else:
                        sys.exit("Then pick a different name please!")
        except FileNotFoundError:
            pass
    print_slow(ORANGE + "Welcome to RPG Game!" + RESET)
    if is_saved_game != 'y':
        player = Player(player_name)
    while player.story_index < 50:
        stats(player)
        story(player.story_index, player)
        option(player.story_index, player)
        if player.story_index == 1:
            encounter(player, "Orc")
        if player.story_index == 3 and player.misc == 1:
            player.story_index += 1
            encounter(player, "Bandit")

def encounter(player, enemy_to_fight):
    tactic = ""
    stun = 0
    player.health = player.max_health

    enemy_data = generate_enemy_data(enemy_to_fight)
    enemy = Enemy(enemy_data["name"], enemy_data["health"], enemy_data["attack_power"], enemy_data["attacks"])
    print_slow(BOLD + RED + f"You encountered a {enemy.name}!\n" + RESET)
    time.sleep(0.2)
    print_slow(BOLD + RED + f"{enemy.name}'s health: {enemy.health}" + RESET)
    while enemy.health > 0 and player.health > 0:
        print_slow(PURPLE + "Moves List:")
        time.sleep(0.1)
        print_slow("Attack(A)")
        time.sleep(0.1)
        print_slow("Strong Attack(SA) (50% Chance)")
        time.sleep(0.1)
        print_slow("Heal(H)")
        time.sleep(0.1)
        print_slow("Stun(S) (40% Chance)" + RESET)
        time.sleep(0.1)
        print()
        tactic = input("Your Move: ").strip().lower()
        while tactic != "a" and tactic != "h" and tactic != "s" and tactic != "sa":
            tactic = input("Your Move: ").strip().lower()
        print()
        if tactic == "a":
            player.attack(enemy)
            if enemy.health <= 0:
                print_slow(BOLD + RED + f"{enemy.name}'s health: 0\n" + RESET)
                print_slow(UNDERLINE + GREEN + f"You defeated the {enemy.name}!" + RESET)
                gold_gained = enemy_data.get("gold_dropped", 0)
                player.gold += gold_gained
                print_slow(BOLD + YELLOW + f"You gained {gold_gained} gold!" + RESET)
                xp_gained = enemy_data.get("xp_recieved", 0)
                player.xp += xp_gained
                print_slow(BOLD + YELLOW + f"You gained {xp_gained} XP!" + RESET)
                if player.xp >= player.xp_needed:
                    player.level_up()
                continue_one = ""
                while continue_one != "y":
                    continue_one = input(UNDERLINE + BLUE+"Continue Y/N: "+RESET).lower().strip()
                    print()
                if player.story_index == 1 or player.story_index == 3 or player.story_index == 5:
                    player.story_index += 1
                return
        elif tactic == "sa":
            print_slow(ORANGE + "You charge up your attack" + RESET)
            print_slow(BLUE + "You must roll higher than a ten in order to land your attack" + RESET)
            print()
            print_slow("Rolling...")
            roll = roll_dice()
            time.sleep(1.5)
            print_slow(PURPLE + BOLD + f"Your roll: {roll}" + RESET)
            print()
            time.sleep(0.5)
            if roll < 11:
                print_slow(BOLD + RED + "You missed!" + RESET)
            else:
                print_slow(BOLD + GREEN + "You successfully landed your attack!" + RESET)
                print()
                time.sleep(0.5)
                player.strong_attack(enemy)
                if enemy.health <= 0:
                    print_slow(BOLD + RED + f"{enemy.name}'s health: 0\n" + RESET)
                    print_slow(UNDERLINE + GREEN + f"You defeated the {enemy.name}!" + RESET)
                    gold_gained = enemy_data.get("gold_dropped", 0)
                    player.gold += gold_gained
                    print_slow(BOLD + YELLOW + f"You gained {gold_gained} gold!" + RESET)
                    xp_gained = enemy_data.get("xp_recieved", 0)
                    player.xp += xp_gained
                    print_slow(BOLD + YELLOW + f"You gained {xp_gained} XP!" + RESET)
                    if player.xp >= player.xp_needed:
                        player.level_up()
                    continue_one = ""
                    while continue_one != "y":
                        continue_one = input(UNDERLINE + BLUE+"Continue Y/N: "+RESET).lower().strip()
                        print()
                    if player.story_index == 1 or player.story_index == 3 or player.story_index == 5:
                        player.story_index += 1
                    return

        elif tactic == "h":
            player.heal()
        elif tactic == "s":
            print_slow(ORANGE + f"You attempt to stun the {enemy.name}" + RESET)
            print_slow(BLUE + f"You must roll a 12 or higher in order to stun the opponent" + RESET)
            print()
            print_slow("Rolling...")
            roll = roll_dice()
            time.sleep(1.5)
            print()
            print_slow(PURPLE + BOLD + f"Your roll: {roll}" + RESET)
            print()
            time.sleep(0.5)
            if roll < 12:
                print_slow(BOLD + RED + f"You failed to stun the {enemy.name}" + RESET)
            else:
                print_slow(BOLD + GREEN + f"You successfully stunned the {enemy.name}!" + RESET)
                stun += 2

        print_slow(RED + BOLD + f"{enemy.name}'s health: {enemy.health}" + RESET)
        print_slow(ORANGE + BOLD + f"Your health: {player.health}" + RESET)
        print("<------------------------------------------------------------------------>")
        print()
        time.sleep(0.5)
        if stun == 0:
            enemy.attack_player(player)
            print()
            time.sleep(0.5)
            if player.health <= 0:
                sys.exit(RED + "You Died" + RESET)
            time.sleep(0.25)
            print_slow(RED + BOLD + f"{enemy.name}'s health: {enemy.health}" + RESET)
            print_slow(ORANGE + BOLD + f"Your health: {player.health}" + RESET)
            print()
        else:
            print_slow(BOLD + ORANGE + f"{enemy.name} is stunned and unable to attack" + RESET)
            print()
            stun -= 1
def option(n, player):
    if n == 1:
        option_text = ""
        print_slow(PURPLE + "Do you want to search your surroundings or find shelter?\n" + RESET)
        while option_text != 'search' or option_text != 'find':
            option_text = input(UNDERLINE + BLUE + "'search' or 'find': " + RESET).strip().lower()
            if option_text == 'search':
                print()
                print_slow(BOLD + YELLOW + "You search your surroundings and find 5 gold on the ground!\n" + RESET)
                player.gold += 5
                continue_one = ""
                while continue_one != "y":
                    continue_one = input(UNDERLINE + BLUE+"Continue Y/N: "+RESET).lower().strip()
                    print()
                print_slow("You hear a sound in the bushes behind you. Summoning your courage you push through the bushes, heart pounding in your chest. There, looming before you in the darkness, is a colossal statue, its imposing form half-shrouded in shadow. Carved from stone, it stands as a silent sentinel, its features weathered by time and neglect.\n")
                continue_one = ""
                while continue_one != "y":
                    continue_one = input(UNDERLINE + BLUE+"Continue Y/N: "+RESET).lower().strip()
                    print()
                print_slow(f"As {player.name} approaches the colossal statue, a deep growl resonates from the shadows, sending shivers down his spine. Suddenly, with a thunderous roar, a monstrous creature emerges, its eyes gleaming with malice in the dim moonlight. Towering over {player.name}, the creature's menacing form fills him with dread. With no way to retreat, {player.name} braces himself for battle, his heart pounding with adrenaline.")
                continue_one = ""
                while continue_one != "y":
                    continue_one = input(UNDERLINE + BOLD + BLUE+"Continue Y/N: "+RESET).lower().strip()
                print()
                return
            if option_text == 'find':
                print()
                print_slow(f"As {player.name}’s heart races with apprehension, he sweeps his gaze across the oppressive darkness surrounding him. With trembling hands, {player.name} reaches out, trying to find his way. He pushes through some bushes, heart pounding in his chest. There, looming before him in the darkness, is a colossal statue, its imposing form half-shrouded in shadow. Carved from stone, it stands as a silent sentinel, its features weathered by time and neglect.\n")
                continue_one = ""
                while continue_one != "y":
                    continue_one = input(UNDERLINE + BLUE+"Continue Y/N: "+RESET).lower().strip()
                    print()
                print_slow(f"As {player.name} approaches the colossal statue, a deep growl resonates from the shadows, sending shivers down his spine. Suddenly, with a thunderous roar, a monstrous creature emerges, its eyes gleaming with malice in the dim moonlight. Towering over {player.name}, the creature's menacing form fills him with dread. With no way to retreat, {player.name} braces himself for battle, his heart pounding with adrenaline.")
                continue_one = ""
                while continue_one != "y":
                    continue_one = input(UNDERLINE + BOLD + BLUE+"Continue Y/N: "+RESET).lower().strip()
                print()
                return
    elif n == 2:
        option_text = ""
        while option_text != 'run' and option_text != 'fight':
            option_text = input(BOLD + RED + "'Run' or 'Fight': " + RESET).strip().lower()
            if option_text == 'fight':
                print()
                print_slow(BOLD + "As the monstrous statue lurches forward, its eyes blazing with an otherworldly light, you steel yourself for the battle ahead. With gritted teeth and trembling hands, you raise your weapon, ready to face this ancient evil head-on.\n" + RESET)
                continue_one = ""
                while continue_one != "y":
                    continue_one = input(UNDERLINE + BLUE+"Continue Y/N: "+RESET).lower().strip()
                    print()
                encounter(player, "Titanic Colossus")
                return
            if option_text == 'run':
                print()
                print_slow(f"Your instincts scream at you to flee, to escape the looming threat before it's too late. With a surge of adrenaline, you turn on your heels and dash through the dense undergrowth, heart pounding in your chest. The forest blurs past you in a dizzying whirl of shadows and moonlight, branches clawing at your skin as you push yourself to the brink of exhaustion. Behind you, the ground trembles with each thunderous footfall of the advancing statue, its monstrous form casting a long, dark shadow over the forest floor. You can feel its malevolent presence looming ever closer, a relentless force of destruction hot on your heels. But you refuse to give in to despair. With every ounce of strength left in your weary limbs, you push yourself to run faster, to outrun the encroaching darkness and find sanctuary in the safety of the unknown.\n")
                continue_one = ""
                while continue_one != "y":
                    continue_one = input(UNDERLINE + BLUE+"Continue Y/N: "+RESET).lower().strip()
                    print()
                print_slow(f"Just when you fear that your legs might give out beneath you, you burst into a small clearing bathed in the soft glow of moonlight. Before you stands a towering waterfall, its cascading waters shimmering like liquid silver in the pale light. The roar of rushing water fills the air, drowning out the thunderous footsteps of your pursuer. Without hesitation, you plunge into the icy embrace of the waterfall, the cool spray washing away the sweat and grime of battle. As you emerge on the other side, you find yourself standing in a hidden grotto illuminated by the soft glow of bioluminescent fungi clinging to the walls. The air is thick with magic and mystery, the ancient stones whispering tales of forgotten civilizations and untold secrets. In the heart of this hidden sanctuary, you catch a glimpse of something truly awe-inspiring—a towering tree, its branches reaching towards the heavens with an otherworldly grace. Drawn by an irresistible curiosity, you approach the tree, its gnarled roots weaving intricate patterns in the soft earth below. As you reach out to touch its weathered bark, you feel a surge of energy coursing through your veins")
                continue_one = ""
                while continue_one != "y":
                    continue_one = input(UNDERLINE + BOLD + BLUE+"Continue Y/N: "+RESET).lower().strip()
                print()
                player.update_max_health(25)
                player.critical_blessing += 1
                print_slow(BOLD + YELLOW + "Your max HP has increased by 25!")
                print_slow("You have recieved the blessing of the Eldertree")
                print_slow("Blessing of the Eldertree grants a 10% chance for a critical strike" + RESET)
                player.story_index += 1
                return
    elif n == 4:
        option_text = ""
        print_slow("As you traverse the ruins, you arrive at a fork in the ancient stone pathway. You can either go left towards a narrow passage, or you can go right towards a library.")
        while option_text != 'right' and option_text != 'left':
            option_text = input(BOLD + RED + "'left' or 'right': " + RESET).strip().lower()
        if option_text == 'right':
            print()
            print_slow("You walk into the library filled with ancient tomes and scrolls, their pages yellowed with age. Dust hangs thick in the air, lending the chamber an air of solemnity and reverence. With a sense of awe, you peruse the shelves, marveling at the wealth of knowledge contained within these ancient texts. Each volume holds the promise of untold wisdom, waiting to be unlocked by those able enough to seek it. Among the myriad books, you come across a tome bound in cracked leather, its title embossed in faded gold lettering: \"The Grimoire of Eternal Vigor.\" As you leaf through its pages, you feel a surge of power coursing through you, as if the very knowledge contained within the tome is seeking to impart its secrets to you. With each revelation, you sense your vitality increasing, your body infused with newfound strength and resilience.\n")
            continue_one = ""
            while continue_one != "y":
                continue_one = input(UNDERLINE + BLUE+"Continue Y/N: "+RESET).lower().strip()
                print()
            player.update_max_health(50)
            print_slow(BOLD + YELLOW + "This ancient tome, imbued with potent magic, grants you the boon of longevity and endurance, increasing your maximum hit points by 50.")
            print_slow("Your max HP has increased by 50!" + RESET)
            player.misc = 2
            continue_one = ""
            while continue_one != "y":
                continue_one = input(UNDERLINE + BLUE+"Continue Y/N: "+RESET).lower().strip()
                print()
            print_slow("As you immerse yourself in the ancient tome, deciphering its cryptic symbols and absorbing its arcane knowledge, a low, guttural growl rumbles through the chamber, shattering the silence like thunder. You look up just in time to see the massive form of a troll emerging from the shadows, its towering frame casting a sinister shadow across the room. Standing at least ten feet tall, the troll is a monstrous sight to behold, its muscular body bulging with sinewy strength. Thick, matted fur covers its hulking form, while jagged tusks protrude from its gaping maw, dripping with saliva. Its eyes, glowing with a feral intensity, fixate on you with a hunger that sends a chill down your spine. With a roar that rattles the very foundations of the chamber, the troll charges forward, its massive fists raised in a menacing display of brute force. You can feel the ground shake beneath its thunderous footsteps as it closes the distance between you, its primal fury unleashed upon you like a relentless storm.")
            print()
            time.sleep(0.5)
            encounter(player, "Troll")
        elif option_text == 'left':
            print()
            print_slow(f"You choose to veer left, drawn towards the unknown dangers that lie hidden within the narrow passage. With cautious steps, you navigate the winding corridors, your senses on high alert for any sign of danger lurking in the shadows. After what feels like an eternity of treacherous exploration, you stumble upon a hidden alcove nestled within the depths of the ruins. Within this secluded chamber lies a glittering treasure, a gleaming amulet pulsating with mystical energy.\n")
            continue_one = ""
            while continue_one != "y":
                continue_one = input(UNDERLINE + BLUE+"Continue Y/N: "+RESET).lower().strip()
                print()
            player.critical_blessing = 2
            print_slow(BOLD + YELLOW + "The amulet enhances your combat prowess, increasing your chance of landing critical strikes in battle.")
            print_slow("Your critical strike chance has increased by 15%!" + RESET)
            player.misc = 3
            continue_one = ""
            while continue_one != "y":
                continue_one = input(UNDERLINE + BLUE+"Continue Y/N: "+RESET).lower().strip()
                print()
            print_slow("As you claim the glittering amulet, a low, guttural growl rumbles through the chamber, shattering the silence like thunder. You look up just in time to see the massive form of a werewolf emerging from the shadows, its towering frame casting a sinister shadow across the room. Standing at least eight feet tall, the werewolf is a monstrous sight to behold, its muscular body rippling with primal power. Thick, dark fur covers its hulking form, while razor-sharp claws glint in the dim light, ready to rend flesh from bone. Its eyes, glowing with a feral intensity, fixate on you with a hunger that sends a chill down your spine. With a snarl that echoes through the chamber, the werewolf lunges forward, its massive jaws snapping shut with enough force to crush bone. You barely have time to react as you dodge its vicious attack, the stench of its hot breath filling your nostrils.")
            print()
            time.sleep(0.5)
            encounter(player, "Werewolf")
        player.story_index = 5
        return

def story(n, player):
    if n == 1:
        print("--------------------------------------------------------------------------------------------------Part One - Lost---------------------------------------------------------------------------------------------------")
        print(f"As the sun dips below the horizon, casting long shadows upon the forest floor, {player.name} glimpses something out of the corner of his eye. He chases after the fleeing figure hoping to figure out what it is. As he pursues the elusive figure deeper into the woods, the forest grows denser, and the shadows more ominous. Soon, {player.name} finds himself hopelessly lost amidst the towering trees, his only companions are the haunting whispers that echo through the night.")
        continue_one = ""
        while continue_one != "y":
            continue_one = input(UNDERLINE + BLUE+"Continue Y/N: "+RESET).lower().strip()
        print("<---------------------------------------------------->")
    elif n == 2:
        save_game(player, "save.json")
        print("-----------------------------------------------------------------------------------------------Part Two - The Statue------------------------------------------------------------------------------------------------")
        print(f"As you celebrate your hard-earned victory over the vicious Orc, the eerie silence of the forest is shattered by a deep, rumbling sound. At first, you dismiss it as mere coincidence, perhaps the groans of the trees swaying in the wind or the distant roar of a predator. But then, you notice something unsettling. The colossal statue, once looming silently in the darkness, begins to stir. Its stone limbs creak and crack as if awakening from a long slumber. You watch in horror as its massive form shifts, sending tremors through the ground beneath your feet. A sense of dread washes over you, cold and palpable. Your heart races as you realize the truth - this is no ordinary statue. It's something ancient, something powerful, something that should not be disturbed. With each ominous movement, the statue draws closer, its towering figure casting long shadows that dance eerily in the moonlight. You stand frozen, unable to tear your gaze away from the unfolding spectacle, your mind reeling with fear and uncertainty. As the statue looms ever closer, you can feel its malevolent presence enveloping you like a suffocating shroud. Your instincts scream at you to flee, to escape this waking nightmare before it's too late. But something holds you rooted to the spot, a morbid curiosity mingled with sheer terror. And then, with a deafening roar that echoes through the forest, the statue lurches forward, its eyes blazing with an otherworldly light. In that moment, you realize the true extent of the danger you're facing.")
        continue_one = ""
        while continue_one != "y":
            continue_one = input(UNDERLINE + BOLD + BLUE+"Continue Y/N: "+RESET).lower().strip()
        print()
    elif n == 3:
        save_game(player, "save.json")
        print("----------------------------------------------------------------------------------------Part Three - Whispers in the Grotto-----------------------------------------------------------------------------------------")
        print(f"As you stand in the tranquil grotto, a sense of calm washes over you, momentarily easing the tension that had gripped your heart. The bioluminescent fungi cast eerie shadows on the walls, creating a surreal ambiance that seems to belong to another realm entirely. But just as you begin to relax, a faint whisper floats through the air, barely audible yet unmistakably sinister. A chill runs down your spine as you turn around, searching for the source of the unsettling sound. Your hand instinctively goes to the hilt of your sword, ready for whatever danger may lurk in the shadows. Then, from the darkness, emerges a figure cloaked in tattered robes, its face obscured by a hood pulled low over its features. With each step it takes, the air grows colder, and the very essence of the grotto seems to recoil in fear.\"Who... who are you?\" You demand, your voice trembling slightly despite your efforts to appear brave. The figure responds with a low, gravelly chuckle that sends shivers down your spine. \"Gimme all your money,\" it intones, its voice echoing ominously through the grotto. You grip your sword tighter, your heart pounding with a mixture of fear and determination. Whatever this creature is, you know you cannot let it stand in the way of uncovering the truths hidden within this mysterious place. With a sudden burst of speed, the creature lunges forward, his own blade flashing in the dim light. ")
        continue_one = ""
        while continue_one != "y":
            continue_one = input(UNDERLINE + BOLD + BLUE+"Continue Y/N: "+RESET).lower().strip()
        print()
        player.misc += 1
    elif n == 4:
        save_game(player, "save.json")
        print("---------------------------------------------------------------------------------------------Part Four - Ancient Ruins----------------------------------------------------------------------------------------------")
        print(f"As you stand amidst the aftermath of the battle, the echoes of clashing swords still ringing in your ears, you take a moment to catch your breath. The adrenaline slowly ebbs away, replaced by a sense of grim satisfaction at emerging victorious against the bandit. But there is no time for celebration. The forest around you seems to hold its breath, as if waiting for the next threat to emerge from the shadows. In the distance, a faint glimmer catches your attention. Squinting against the dim light, you make out the outline of a crumbling ruin, its ancient stones cloaked in the shadows of the forest. It beckons to you like a silent sentinel, promising both danger and opportunity. With a resolute nod, you sheathe your sword and set off towards the ruins, each step a cautious reminder of the perils that await. As you draw nearer, the air grows heavy with the weight of forgotten history, the whispers of bygone civilizations echoing through the trees. At the entrance to the ruins, you pause, steeling yourself for whatever mysteries lie within. With a deep breath, you step across the threshold, the darkness swallowing you whole as you delve deeper into the heart of the ancient structure. The air is thick with dust and decay, the remnants of a forgotten era clinging to the walls like ghosts of the past. As you press on, a sense of foreboding settles over you like a suffocating shroud. The ruins seem to come alive around you, their ancient stones whispering tales of long-forgotten tragedies and untold horrors.")
        continue_one = ""
        while continue_one != "y":
            continue_one = input(UNDERLINE + BOLD + BLUE+"Continue Y/N: "+RESET).lower().strip()
        print()
    elif n == 5:
        save_game(player, "save.json")
        print("------------------------------------------------------------------------------------------------Part Five - Dragon--------------------------------------------------------------------------------------------------")
        print(f"As you catch your breath, your senses still heightened from the adrenaline-fueled battle, you steel yourself for what lies ahead. With cautious steps, you navigate the labyrinthine corridors of the ancient ruins. After what feels like an eternity of treacherous exploration, you finally arrive at a massive chamber, its walls adorned with intricate carvings depicting scenes of ancient battles and forgotten legends. The air crackles with a palpable sense of magic, sending shivers down your spine as you step into the heart of the room. But before you can fully take in your surroundings, a deafening roar echoes through the chamber, shaking the very foundations of the ruins. You whirl around to see a sight that freezes you in your tracks - a dragon, its massive form coiled around a massive pillar, its eyes blazing with an otherworldly fire. With each beat of its powerful wings, the dragon fills the room with a searing heat, its scales gleaming like molten gold in the dim light. You can feel the raw power emanating from the creature, a primal force of nature that commands respect and fear in equal measure. As the dragon fixes its gaze upon you, you can sense the ancient intelligence behind its eyes, a wisdom that spans centuries of existence. In that moment, you realize the true magnitude of the challenge you face - to confront a creature of such majesty and might is to defy the very laws of nature itself.")
        continue_one = ""
        while continue_one != "y":
            continue_one = input(UNDERLINE + BOLD + BLUE+"Continue Y/N: "+RESET).lower().strip()
        print()
        encounter(player, "Dragon")
        if player.misc == 2:
            player.critical_blessing = 2
            print_slow(BOLD + YELLOW + "The dragon drops a shiny amulet!\nThe amulet enhances your combat prowess, increasing your chance of landing critical strikes in battle.")
            print_slow("Your critical strike chance has increased by 15%!" + RESET)
        elif player.misc == 3:
            player.update_max_health(50)
            print_slow(BOLD + YELLOW + "The dragon drops an ancient tome!\nThis ancient tome, imbued with potent magic, grants you the boon of longevity and endurance, increasing your maximum hit points by 50.")
            print_slow("Your max HP has increased by 50!" + RESET)

    elif n == 6:
        save_game(player, "save.json")
        print("Next Boss!")
        encounter(player, "Celestial Seraph")

def stats(self):
    while True:
        headers = [PURPLE + "Player", self.name + RESET]
        weapon_info = [
            ["Weapon", self.weapon["name"]],
            ["Attack Power Bonus", self.weapon["attack_power_bonus"]],
            ["Upgrade Cost", self.weapon["upgrade_cost"]]
        ]
        data = [
            ["Level", self.level],
            ["Gold", self.gold],
            ["XP", self.xp],
            ["XP needed to lvl up", self.xp_needed],
            ["Health", self.max_health],
            ["Atk Pow", self.attack_power]
        ]
        print("<---------------------------------------------------->")
        print_slow(GREEN + "Your current stats:" + RESET)
        print(tabulate(data, headers=headers, tablefmt="grid"))
        blacksmith_choice = input("Would you like to take a look at your weapon? Y/N ").strip().lower()
        if blacksmith_choice == 'y':
            print_slow(PURPLE + "Weapon Info:" + RESET)
            print(tabulate(weapon_info, tablefmt="grid"))
            while True:
                if self.gold >= self.weapon["upgrade_cost"]:
                    level_choice = input(UNDERLINE + BLUE + "Do you wish to upgrade your weapon? Y/N " + RESET).strip().lower()
                    if level_choice == 'y':
                        self.upgrade_weapon()
                        print()
                        print_slow(YELLOW + "Updated stats:" + RESET)
                        weapon_info = [
                            ["Weapon", self.weapon["name"]],
                            ["Attack Power Bonus", self.weapon["attack_power_bonus"]],
                            ["Upgrade Cost", self.weapon["upgrade_cost"]]
                        ]
                        print(YELLOW + tabulate(weapon_info, headers=["Player", self.name], tablefmt="grid") + RESET)
                    elif level_choice == 'n':
                        return
                else:
                    print_slow(BOLD + RED + "You don't have enough gold to upgrade your weapon." + RESET)
                    continue_one = ""
                    while continue_one != "y":
                        continue_one = input(UNDERLINE + BLUE+"Continue Y/N: "+RESET).lower().strip()
                        if continue_one == 'y':
                            return
                    return
        else:
            return

def save_game(player, filename):
    try:
        with open(filename, 'r') as file:
            players_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        players_data = {}


    players_data[player.name] = {
        'name': player.name,
        'health': player.max_health,
        'max_health': player.max_health,
        'attack_power': player.attack_power,
        'gold': player.gold,
        'level': player.level,
        'story_index': player.story_index,
        'weapon': player.weapon,
        'xp': player.xp,
        'xp_needed': player.xp_needed,
        'critical_blessing': player.critical_blessing
    }

    with open(filename, 'w') as file:
        json.dump(players_data, file)

def load_game(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def roll_dice():
    return random.randint(1, 20)

def print_slow(text):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.01)
    print()

def i_cannot_test():
    return "With my functions"

def each_of_my_functions():
    return "Requires a player class and data accumulated over time"

def please_email_me():
    return "If this is unacceptable"

if __name__ == "__main__":
    main()
