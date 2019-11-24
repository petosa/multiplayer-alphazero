# An interface for an agent that interfaces with states.
class Player:

    # Player is given a state s and returns a mutated state.
    def update_state(self, s):
        raise NotImplementedError

    # Resets the internal data members of the player.
    # For example, resets the accumulated Monte Carlo tree.
    # Needed to reset player's experience between games without having to reinstantiate the player.
    def reset(self):
        raise NotImplementedError
