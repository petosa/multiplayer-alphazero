import numpy as np
import sys
import copy
from scipy.signal import correlate2d
sys.path.append("..")
from game import Game

# Implementation for Connect 3x3.
class Connect3x3(Game):

    # Returns a blank Connect 3x3 board.
    # There are extra layers to represent the pieces, as well as a turn indicator layer.
    def get_initial_state(self):
        env = 0
        board = np.zeros((6, 7, self.get_num_players()*2), dtype=np.float32) # Final plane is a turn indicator
        board[:,:,self.get_num_players()] = 1
        return {"env":env, "obs":board}

    # Returns a 7-item boolean array indicating open slots. 
    def get_available_actions(self, s):
        pieces = s["obs"][:, :, :self.get_num_players()].sum(axis=-1)
        counts = pieces.sum(axis=0)
        return counts != 6
        
    # Drop a piece in a slot.
    def take_action(self, s, a):
        p = self.get_player(s)
        s = copy.deepcopy(s)
        pieces = s["obs"][:, :, :self.get_num_players()].sum(axis=-1)
        col = np.argwhere(a.astype(np.float32) == 1.).item()
        row = np.argwhere(pieces[:,col] == 0).max()
        s["obs"][row, col, p] = 1.
        num_p = self.get_num_players()
        s["obs"][:,:,num_p+p] = 0 # No longer p's turn
        s["obs"][:,:,num_p + (p+1) % num_p] = 1
        s["env"] += 1
        return s

    # Check all possible 3 in-a-rows for a win.
    def check_game_over(self, s):
        in_a_row = 3
        for p in range(self.get_num_players()):
            reward = -np.ones(self.get_num_players())
            reward[p] = 1
            board = s["obs"][:,:,p]
            if np.isin(in_a_row, correlate2d(board, np.ones((1, in_a_row)), mode="valid")): return reward # Horizontal
            if np.isin(in_a_row, correlate2d(board, np.ones((in_a_row, 1)), mode="valid")): return reward # Vertical
            i = np.eye(in_a_row)
            if np.isin(in_a_row, correlate2d(board, i, mode="valid")): return reward # Downward diagonol
            if np.isin(in_a_row, correlate2d(board, np.fliplr(i), mode="valid")): return reward # Upward diagonol
        if self.get_available_actions(s).sum() == 0: # Full board, draw
            return np.zeros(self.get_num_players())

    def get_player(self, s):
        indicator = s["obs"][0,0,self.get_num_players():]
        return np.argwhere(indicator == 1).item()

    # Fixed constant
    def get_num_players(self):
        return 3

    def get_hash(self, s):
        return s["obs"].tostring()

    # Print a human-friendly visualization of the board.
    def visualize(self, s):
        board = np.ones((6,7)).astype(np.object)
        board[:,:] = "_"
        board[s["obs"][:,:,0] == 1] = '🔴'
        board[s["obs"][:,:,1] == 1] = '🉐'
        board[s["obs"][:,:,2] == 1] = '🔵'
        last_line = np.array([str(x) for x in np.arange(7)])
        print(np.concatenate([board, last_line.reshape(1,-1)], axis=0))

