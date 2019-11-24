import numpy as np
import sys
from scipy.signal import correlate2d
sys.path.append("..")
from game import Game

# Implementation for the classic Tic-Tac-Toe game.
class TicTacToe(Game):

    # Returns a blank Tic-Tac-Toe board.
    # There are extra layers to represent X and O pieces, as well as a turn indicator layer.
    def get_initial_state(self):
        return np.zeros((3, 3, 2 + 1), dtype=np.float32) # Final plane is a turn indicator

    # Returns a 3x3 boolean ndarray indicating open squares. 
    def get_available_actions(self, s):
        return s[:, :, :2].sum(axis=-1) == 0
        
    # Place an X or O in state.
    def take_action(self, s, a):
        p = self.get_player(s)
        s = s.copy()
        s[:,:,p] += a.astype(np.float32) # Next move
        s[:,:,2] = (s[:,:,2] + 1) % 2 # Toggle player
        return s

   # Check all possible 3-in-a-rows for a win.
    def check_game_over(self, s):
        for p in [0, 1]:
            reward = -np.ones(self.get_num_players())
            reward[p] = 1
            board = s[:,:,p]
            if np.isin(3, correlate2d(board, np.ones((1, 3)), mode="valid")): return reward # Horizontal
            if np.isin(3, correlate2d(board, np.ones((3, 1)), mode="valid")): return reward # Vertical
            i = np.eye(3)
            if np.isin(3, correlate2d(board, i, mode="valid")): return reward # Downward diagonol
            if np.isin(3, correlate2d(board, np.fliplr(i), mode="valid")): return reward # Upward diagonol
        if self.get_available_actions(s).sum() == 0: # Full board, draw
            return np.zeros(self.get_num_players())

    # Return 0 for X's turn or 1 for O's turn.
    def get_player(self, s):
        return int(s[0,0,2])

    # Fixed constant for Tic-Tac-Toe
    def get_num_players(self):
        return 2

    # Print a human-friendly visualization of the board.
    def visualize(self, s):
        board = np.ones((3,3)).astype(np.object)
        board[:,:] = " "
        board[s[:,:,0] == 1] = 'x'
        board[s[:,:,1] == 1] = 'o'
        print(board)

