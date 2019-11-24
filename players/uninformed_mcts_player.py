import sys
import numpy as np
sys.path.append("..")
from mcts import MCTS
from player import Player
from models.dumbnet import DumbNet
from neural_network import NeuralNetwork

class UninformedMCTSPlayer(Player):

    def __init__(self, game, simulations):
        self.game = game
        self.simulations = simulations
        self.tree = MCTS(game, NeuralNetwork(game, DumbNet))

    def update_state(self, s):
        for _ in range(self.simulations):
            self.tree.simulate(s)

        dist = self.tree.get_distribution(s, 0)
        a = tuple(dist[np.argmax(dist[:,1]),0])
        available = self.game.get_available_actions(s)
        template = np.zeros_like(available)
        template[a] = 1
        s_prime = self.game.take_action(s, template)
        return s_prime

    def reset(self):
        self.tree = MCTS(self.game, NeuralNetwork(self.game, DumbNet))
