def generate_enemy_data(enemy_name):
    if enemy_name == "Goblin":
        return {
            "name": "Goblin",
            "health": 40,
            "attack_power": 12,
            "attacks": [
                {"name": "Slash", "damage": 18},
                {"name": "Stab", "damage": 15},
                {"name": "Throw Dirt", "damage": 10}
            ],
            "gold_dropped": 12,
            "xp_recieved": 20
        }
    elif enemy_name == "Orc":
        return {
            "name": "Orc",
            "health": 75,
            "attack_power": 25,
            "attacks": [
                {"name": "Smash", "damage": 60, "telegraphed": True},
                {"name": "Bite", "damage": 28},
                {"name": "Furious Roar", "damage": 20}
            ],
            "gold_dropped": 25,
            "xp_recieved": 50
        }
    elif enemy_name == "Troll":
        return {
            "name": "Troll",
            "health": 300,
            "attack_power": 80,
            "attacks": [
                {"name": "Club Swing", "damage": 140, "telegraphed": True},
                {"name": "Boulder Throw", "damage": 70},
                {"name": "Regenerate", "healing": 80}
            ],
            "gold_dropped": 75,
            "xp_recieved": 200
        }
    elif enemy_name == "Skeleton":
        return {
            "name": "Skeleton",
            "health": 110,
            "attack_power": 55,
            "attacks": [
                {"name": "Bone Slash", "damage": 60},
                {"name": "Spectral Bolt", "damage": 55},
                {"name": "Summon Minions", "healing": 40}
            ],
            "gold_dropped": 55,
            "xp_recieved": 110
        }
    elif enemy_name == "Bandit":
        return {
            "name": "Bandit",
            "health": 150,
            "attack_power": 70,
            "attacks": [
                {"name": "Backstab", "damage": 90, "telegraphed": True},
                {"name": "Steal Gold", "damage": 45},
                {"name": "Smoke Bomb", "damage": 40}
            ],
            "gold_dropped": 50,
            "xp_recieved": 150
        }
    elif enemy_name == "Werewolf":
        return {
            "name": "Werewolf",
            "health": 275,
            "attack_power": 85,
            "attacks": [
                {"name": "Claw Swipe", "damage": 150, "telegraphed": True},
                {"name": "Bite", "damage": 70},
                {"name": "Howl", "damage": 80}
            ],
            "gold_dropped": 85,
            "xp_recieved": 200
        }
    elif enemy_name == "Giant":
        return {
            "name": "Giant",
            "health": 300,
            "attack_power": 100,
            "attacks": [
                {"name": "Ground Pound", "damage": 110},
                {"name": "Rock Throw", "damage": 105},
                {"name": "Stomp", "damage": 100}
            ],
            "gold_dropped": 100
        }
    elif enemy_name == "Dragon":
        return {
            "name": "Dragon",
            "health": 750,
            "attack_power": 120,
            "attacks": [
                {"name": "Fire Breath", "damage": 150, "telegraphed": True},
                {"name": "Claw Swipe", "damage": 100},
                {"name": "Wing Buffet", "damage": 100}
            ],
            "gold_dropped": 400
        }
    elif enemy_name == "Wendigo":
        return {
            "name": "Wendigo",
            "health": 500,
            "attack_power": 140,
            "attacks": [
                {"name": "Hellfire", "damage": 150},
                {"name": "Dark Slash", "damage": 145},
                {"name": "Soul Drain", "damage": 140}
            ],
            "gold_dropped": 140
        }
    elif enemy_name == "Vampire":
        return {
            "name": "Vampire",
            "health": 600,
            "attack_power": 160,
            "attacks": [
                {"name": "Blood Drain", "damage": 170},
                {"name": "Shadow Step", "damage": 165},
                {"name": "Vampiric Touch", "damage": 160}
            ],
            "gold_dropped": 160
        }
    elif enemy_name == "Lich":
        return {
            "name": "Lich",
            "health": 700,
            "attack_power": 180,
            "attacks": [
                {"name": "Necrotic Blast", "damage": 190},
                {"name": "Summon Undead", "damage": 185},
                {"name": "Soul Siphon", "damage": 180}
            ],
            "gold_dropped": 180
        }
    elif enemy_name == "Hydra":
        return {
            "name": "Hydra",
            "health": 800,
            "attack_power": 200,
            "attacks": [
                {"name": "Bite", "damage": 210},
                {"name": "Acid Spit", "damage": 205},
                {"name": "Regenerate", "healing": 200}
            ],
            "gold_dropped": 200
        }
    elif enemy_name == "Kraken":
        return {
            "name": "Kraken",
            "health": 900,
            "attack_power": 220,
            "attacks": [
                {"name": "Tentacle Slam", "damage": 230},
                {"name": "Ink Cloud", "damage": 225},
                {"name": "Tidal Wave", "damage": 220}
            ],
            "gold_dropped": 220
        }
    elif enemy_name == "Behemoth":
        return {
            "name": "Behemoth",
            "health": 1000,
            "attack_power": 240,
            "attacks": [
                {"name": "Stampede", "damage": 250},
                {"name": "Earthquake", "damage": 245},
                {"name": "Thunderous Roar", "damage": 240}
            ],
            "gold_dropped": 240
        }
    elif enemy_name == "Balrog":
        return {
            "name": "Balrog",
            "health": 1100,
            "attack_power": 260,
            "attacks": [
                {"name": "Fire Whip", "damage": 270},
                {"name": "Inferno Burst", "damage": 265},
                {"name": "Shadow Dive", "damage": 260}
            ],
            "gold_dropped": 260
        }
    elif enemy_name == "Chimera":
        return {
            "name": "Chimera",
            "health": 1200,
            "attack_power": 280,
            "attacks": [
                {"name": "Claw Swipe", "damage": 290},
                {"name": "Flame Breath", "damage": 285},
                {"name": "Venomous Sting", "damage": 280}
            ],
            "gold_dropped": 280
        }
    elif enemy_name == "Basilisk":
        return {
            "name": "Basilisk",
            "health": 1300,
            "attack_power": 300,
            "attacks": [
                {"name": "Petrifying Gaze", "damage": 310},
                {"name": "Venom Spit", "damage": 305},
                {"name": "Tail Whip", "damage": 300}
            ],
            "gold_dropped": 300
        }
    elif enemy_name == "Sphinx":
        return {
            "name": "Sphinx",
            "health": 1400,
            "attack_power": 320,
            "attacks": [
                {"name": "Riddle", "damage": 0},  # No damage
                {"name": "Pounce", "damage": 330},
                {"name": "Guardian's Wrath", "damage": 325}
            ],
            "gold_dropped": 320
        }
    elif enemy_name == "Leviathan":
        return {
            "name": "Leviathan",
            "health": 1500,
            "attack_power": 340,
            "attacks": [
                {"name": "Tidal Wave", "damage": 350},
                {"name": "Whirlpool", "damage": 345},
                {"name": "Abyssal Crush", "damage": 340}
            ],
            "gold_dropped": 340
        }
    elif enemy_name == "Phoenix":
        return {
            "name": "Phoenix",
            "health": 1600,
            "attack_power": 360,
            "attacks": [
                {"name": "Flame Burst", "damage": 370},
                {"name": "Rebirth", "healing": 400},
                {"name": "Solar Flare", "damage": 365}
            ],
            "gold_dropped": 360
        }
    elif enemy_name == "Siren":
        return {
            "name": "Siren",
            "health": 1700,
            "attack_power": 380,
            "attacks": [
                {"name": "Enthralling Song", "damage": 0},  # No damage
                {"name": "Drown", "damage": 390},
                {"name": "Lure", "damage": 385}
            ],
            "gold_dropped": 380
        }
    elif enemy_name == "Minotaur":
        return {
            "name": "Minotaur",
            "health": 1800,
            "attack_power": 400,
            "attacks": [
                {"name": "Gore", "damage": 410},
                {"name": "Charge", "damage": 405},
                {"name": "Rage", "damage": 400}
            ],
            "gold_dropped": 400
        }
    elif enemy_name == "Cerberus":
        return {
            "name": "Cerberus",
            "health": 1900,
            "attack_power": 420,
            "attacks": [
                {"name": "Bite", "damage": 430},
                {"name": "Hellfire Breath", "damage": 425},
                {"name": "Soul Devour", "damage": 420}
            ],
            "gold_dropped": 420
        }
    elif enemy_name == "Gorgon":
        return {
            "name": "Gorgon",
            "health": 2000,
            "attack_power": 440,
            "attacks": [
                {"name": "Petrifying Gaze", "damage": 450},
                {"name": "Venomous Bite", "damage": 445},
                {"name": "Stone Throw", "damage": 440}
            ],
            "gold_dropped": 440
        }
    elif enemy_name == "Juggernaut":
        return {
            "name": "Juggernaut",
            "health": 2100,
            "attack_power": 460,
            "attacks": [
                {"name": "Smash", "damage": 470},
                {"name": "Steamroll", "damage": 465},
                {"name": "Iron Defense", "damage": 0}  # No damage
            ],
            "gold_dropped": 460
        }
    elif enemy_name == "Banshee":
        return {
            "name": "Banshee",
            "health": 2200,
            "attack_power": 480,
            "attacks": [
                {"name": "Wail", "damage": 490},
                {"name": "Spectral Touch", "damage": 485},
                {"name": "Ethereal Form", "damage": 0}  # No damage
            ],
            "gold_dropped": 480
        }
    elif enemy_name == "Grim Reaper":
        return {
            "name": "Grim Reaper",
            "health": 3200,
            "attack_power": 640,
            "attacks": [
                {"name": "Scythe Slash", "damage": 650},
                {"name": "Death's Grasp", "damage": 645},
                {"name": "Soul Harvest", "damage": 640}
            ],
            "gold_dropped": 640
        }
    elif enemy_name == "Dark Knight":
        return {
            "name": "Dark Knight",
            "health": 3400,
            "attack_power": 660,
            "attacks": [
                {"name": "Shadow Strike", "damage": 670},
                {"name": "Cursed Blade", "damage": 665},
                {"name": "Dread Charge", "damage": 660}
            ],
            "gold_dropped": 660
        }
    elif enemy_name == "Elder Dragon":
        return {
            "name": "Elder Dragon",
            "health": 4000,
            "attack_power": 700,
            "attacks": [
                {"name": "Supernova Breath", "damage": 710},
                {"name": "Wing Buffet", "damage": 705},
                {"name": "Celestial Roar", "damage": 700}
            ],
            "gold_dropped": 700
        }
    elif enemy_name == "Ancient Lich":
        return {
            "name": "Ancient Lich",
            "health": 4500,
            "attack_power": 750,
            "attacks": [
                {"name": "Eternal Curse", "damage": 760},
                {"name": "Necrotic Wave", "damage": 755},
                {"name": "Dark Obliteration", "damage": 750}
            ],
            "gold_dropped": 750
        }
    elif enemy_name == "Titan":
        return {
            "name": "Titan",
            "health": 5000,
            "attack_power": 800,
            "attacks": [
                {"name": "Crushing Blow", "damage": 810},
                {"name": "Earthquake", "damage": 805},
                {"name": "Colossal Slam", "damage": 800}
            ],
            "gold_dropped": 800,
            "xp_recieved": 10000
        }
    elif enemy_name == "Dreadlord":
        return {
            "name": "Dreadlord",
            "health": 5500,
            "attack_power": 850,
            "attacks": [
                {"name": "Demonic Slash", "damage": 860},
                {"name": "Shadow Nova", "damage": 855},
                {"name": "Dreadful Curse", "damage": 850}
            ],
            "gold_dropped": 850
        }
    elif enemy_name == "Celestial Seraph":
        return {
            "name": "Celestial Seraph",
            "health": 6000,
            "attack_power": 900,
            "attacks": [
                {"name": "Heavenly Strike", "damage": 910},
                {"name": "Divine Wrath", "damage": 905},
                {"name": "Sacred Nova", "damage": 900}
            ],
            "gold_dropped": 900
        }
    elif enemy_name == "Abyssal Leviathan":
        return {
            "name": "Abyssal Leviathan",
            "health": 6500,
            "attack_power": 950,
            "attacks": [
                {"name": "Abyssal Crush", "damage": 960},
                {"name": "Tidal Barrage", "damage": 955},
                {"name": "Dark Whirlpool", "damage": 950}
            ],
            "gold_dropped": 950
        }
    elif enemy_name == "Ethereal Reaper":
        return {
            "name": "Ethereal Reaper",
            "health": 7000,
            "attack_power": 1000,
            "attacks": [
                {"name": "Soul Harvest", "damage": 1010},
                {"name": "Ghostly Slash", "damage": 1005},
                {"name": "Shadow Surge", "damage": 1000}
            ],
            "gold_dropped": 1000
        }
    elif enemy_name == "Ancient Wyrm":
        return {
            "name": "Ancient Wyrm",
            "health": 7500,
            "attack_power": 1050,
            "attacks": [
                {"name": "Dragonfire Breath", "damage": 1060},
                {"name": "Tail Whip", "damage": 1055},
                {"name": "Earthshaking Roar", "damage": 1050}
            ],
            "gold_dropped": 1050
        }
    elif enemy_name == "Titanic Colossus":
        return {
            "name": "Titanic Colossus",
            "health": 8000,
            "attack_power": 1100,
            "attacks": [
                {"name": "Mountainous Slam", "damage": 1110},
                {"name": "Earthquake Stomp", "damage": 1105},
                {"name": "Thunderous Roar", "damage": 1100}
            ],
            "gold_dropped": 1100
        }
    elif enemy_name == "Infernal Overlord":
        return {
            "name": "Infernal Overlord",
            "health": 25000,
            "attack_power": 3000,
            "attacks": [
                {"name": "Hellfire Nova", "damage": 3010},
                {"name": "Demonic Fury", "damage": 3005},
                {"name": "Infernal Conflagration", "damage": 3000}
            ],
            "gold_dropped": 3000
        }
    elif enemy_name == "Abyssal Sovereign":
        return {
            "name": "Abyssal Sovereign",
            "health": 30000,
            "attack_power": 3500,
            "attacks": [
                {"name": "Void Eruption", "damage": 3510},
                {"name": "Cataclysmic Annihilation", "damage": 3505},
                {"name": "Eldritch Obliteration", "damage": 3500}
            ],
            "gold_dropped": 3500
        }
