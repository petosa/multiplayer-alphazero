import numpy as np
import sys
import copy
from scipy.signal import correlate2d
sys.path.append("..")
from game import Game

# Implementation for three-player Tic-Tac-Toe.
class TicTacMo(Game):

    # Returns a blank Tic-Tac-Mo board.
    # There are extra layers to represent X and O pieces, as well as a turn indicator layer.
    def get_initial_state(self):
        env = 0
        init = np.zeros((3, 5, self.get_num_players() + self.get_num_players()), dtype=np.float32) # Final planes are turn indicators
        init[:,:,self.get_num_players()] = 1
        return {"env":env, "state":init}

    # Returns a 3x5 boolean ndarray indicating open squares. 
    def get_available_actions(self, s):
        return s["state"][:, :, :self.get_num_players()].sum(axis=-1) == 0
        
    # Place an X, O, or Y in state.
    def take_action(self, s, a):
        p = self.get_player(s)
        s = copy.deepcopy(s)
        s["state"][:,:,p] += a.astype(np.float32) # Next move
        num_p = self.get_num_players()
        s["state"][:,:,num_p+p] = 0
        s["state"][:,:,num_p + (p+1) % num_p] = 1
        s["env"] += 1
        return s

   # Check all possible 3-in-a-rows for a win.
    def check_game_over(self, s):
        for p in range(self.get_num_players()):
            reward = -np.ones(self.get_num_players())
            reward[p] = 1
            board = s["state"][:,:,p]
            if np.isin(3, correlate2d(board, np.ones((1, 3)), mode="valid")): return reward # Horizontal
            if np.isin(3, correlate2d(board, np.ones((3, 1)), mode="valid")): return reward # Vertical
            i = np.eye(3)
            if np.isin(3, correlate2d(board, i, mode="valid")): return reward # Downward diagonol
            if np.isin(3, correlate2d(board, np.fliplr(i), mode="valid")): return reward # Upward diagonol
        if self.get_available_actions(s).sum() == 0: # Full board, draw
            return np.zeros(self.get_num_players())

    # Return 0 for X's turn or 1 for O's turn.
    def get_player(self, s):
        indicator = s["state"][0,0,self.get_num_players():]
        return np.argwhere(indicator == 1).item()

    # Fixed constant for Tic-Tac-Toe
    def get_num_players(self):
        return 3

    # Print a human-friendly visualization of the board.
    def visualize(self, s):
        board = np.ones((3,5)).astype(np.object)
        board[:,:] = " "
        board[s["state"][:,:,0] == 1] = 'x'
        board[s["state"][:,:,1] == 1] = 'o'
        board[s["state"][:,:,2] == 1] = 'y'
        print(board)

