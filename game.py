# Interface for defining a new game.
# See the games folder for examples.
class Game:

    # Returns a dict with hidden states "env" (e.g. gym env) and "obs" as ndarray representing the initial game state.
    # Note that array values should be between 0 and 1.
    def get_initial_state(self):
        raise NotImplementedError

    # Returns a boolean ndarray of actions, where True indicates an available action
    # and False indicates an unavailable action at the current state s.
    # The shape of this action ndarray does not have to match the shape of the state.
    def get_available_actions(self, s):
        raise NotImplementedError

    # Given the current state, evaluate if the game has ended.
    # Convention:
    # Return None if there is no winner yet.
    # Otherwise, return a numpy series of scores, where index corresponds to player number.
    def check_game_over(self, s):
        raise NotImplementedError()
    
    # Given a state s and action a, produces a dict with new hidden states "env" (e.g. gym env) 
    # and "obs" as new ndarray s' which is the
    # resulting state from taking action a in state s.
    # Note that array values should be between 0 and 1.
    # Make sure this does NOT modify s in-place; return a new ndarray instead.
    def take_action(self, s, a):
        raise NotImplementedError()

    # Given the current state s, return an integer indicating which player's turn it is.
    # The first player is 0, second player is 1, and so on.
    def get_player(self, s):
        raise NotImplementedError()

    # Return the number of max players in this game.
    def get_num_players(self):
        raise NotImplementedError()

    # Visualizes the given state.
    def visualize(self, s):
        raise NotImplementedError()
