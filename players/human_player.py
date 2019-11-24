import sys
import numpy as np
sys.path.append("..")
from player import Player

class HumanPlayer(Player):

    def __init__(self, game):
        self.game = game

    def update_state(self, s):
        while True:
            try:
                a = input()
                a = tuple([int(x) for x in a.split(" ")])
                available = self.game.get_available_actions(s)
                template = np.zeros_like(available)
                template[a] = 1
                if available[template] == 1:
                    break
            except (ValueError, IndexError):
                pass
            print("Invalid move! Try again.")
        s_prime = self.game.take_action(s, template)
        return s_prime

    def reset(self):
        pass
