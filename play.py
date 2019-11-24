from itertools import permutations
import numpy as np

# Runs a match with the given game and list of players.
# Returns an array of points. Player number is the index into this array.
# For each match, a player gains a point if it wins, loses a point if it loses,
# and gains no points if it ties.
def play_match(game, players, verbose=False, permute=False):

    # You can use permutations to break the dependence on player order in measuring strength.
    matches = list(permutations(np.arange(len(players)))) if permute else [np.arange(len(players))]
    
    # Initialize scoreboard
    scores = np.zeros(game.get_num_players())

    # Run the matches (there will be multiple if permute=True)
    for order in matches:

        for p in players:
            p.reset() # Clear player trees to make the next match fair

        s = game.get_initial_state()
        if verbose: game.visualize(s)
        game_over = game.check_game_over(s)

        while game_over is None:
            p = order[game.get_player(s)]
            if verbose: print("Player #{}'s turn.".format(p))
            s = players[p].update_state(s)
            if verbose: game.visualize(s)
            game_over = game.check_game_over(s)

        scores[list(order)] += game_over
        if verbose: print("Δ" + str(game_over[list(order)]) + ", Current scoreboard: " + str(scores))


    if verbose: print("Final scores:", scores)
    return scores



if __name__ == "__main__":
    from players.human_player import HumanPlayer
    from neural_network import NeuralNetwork
    from models.senet import SENet
    from players.uninformed_mcts_player import UninformedMCTSPlayer
    from players.deep_mcts_player import DeepMCTSPlayer
    from games.tictactoe import TicTacToe
    from games.tictacmo import TicTacMo


    # Change these variable 
    game = TicTacMo()
    #ckpt = 15
    #nn = NeuralNetwork(game, SENet, cuda=True)
    #nn.load(ckpt)
    
    # HumanPlayer(game),
    # UninformedMCTSPlayer(game, simulations=1000)
    # DeepMCTSPlayer(game, nn, simulations=50)
    
    players = [HumanPlayer(game), HumanPlayer(game),UninformedMCTSPlayer(game, simulations=3000)]
    for _ in range(1):
        play_match(game, players, verbose=True, permute=True)
    
