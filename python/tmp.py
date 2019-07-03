class Character():
    def __init__(self, name, health = '', defense = '', aa = 'bye'):
        
        self.name = name
        #self.aa = aa
        self.aa = 'hi'

class Player(Character):
    def __init__(self, name, health, defense, str, int):
        Character.__init__(self, name)


hero = Player("Billy", 200, 10, 10, 2)
print (hero.name)
