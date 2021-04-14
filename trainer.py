import time
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from mcts import MCTS
from play import play_match
from players.uninformed_mcts_player import UninformedMCTSPlayer
from players.deep_mcts_player import DeepMCTSPlayer

# Object that coordinates AlphaZero training.
class Trainer:

    def __init__(self, game, nn, num_simulations, num_games, num_updates, buffer_size_limit, cpuct, num_threads):
        self.game = game
        self.nn = nn
        self.num_simulations = num_simulations
        self.num_games = num_games
        self.num_updates = num_updates
        self.buffer_size_limit = buffer_size_limit
        self.training_data = np.zeros((0,4))
        self.cpuct = cpuct
        self.num_threads = num_threads
        self.error_log = []


    # Does one game of self play and generates training samples.
    def self_play(self, temperature):
        s = self.game.get_initial_state()
        tree = MCTS(self.game, self.nn)

        data = []
        scores = self.game.check_game_over(s)
        root = True
        alpha = 1
        weight = .25
        while scores is None:
            
            # Think
            for _ in range(self.num_simulations):
                tree.simulate(s, cpuct=self.cpuct)

            # Fetch action distribution and append training example template.
            dist = tree.get_distribution(s, temperature=temperature)

            # Add dirichlet noise to root
            if root:
                noise = np.random.dirichlet(np.array(alpha*np.ones_like(dist[:,1].astype(np.float32))))
                dist[:,1] = dist[:,1]*(1-weight) + noise*weight
                root = False

            available = self.game.get_available_actions(s)

            data.append([s["obs"], dist[:,1], None, available]) # state, prob, action_mask, outcome

            # Sample an action
            idx = np.random.choice(len(dist), p=dist[:,1].astype(np.float))
            a = tuple(dist[idx, 0])

            # Apply action
            template = np.zeros_like(available)
            template[a] = 1
            s = self.game.take_action(s, template)

            # Check scores
            scores = self.game.check_game_over(s)

        # Update training examples with outcome
        for i, _ in enumerate(data):
            data[i][2] = scores

        return np.array(data, dtype=np.object)


    # Performs one iteration of policy improvement.
    # Creates some number of games, then updates network parameters some number of times from that training data.
    def policy_iteration(self, verbose=False):
        temperature = 1   

        if verbose:
            print("SIMULATING " + str(self.num_games) + " games")
            start = time.time()
        if self.num_threads > 1:
            jobs = [temperature]*self.num_games
            pool = ThreadPool(self.num_threads)
            new_data = pool.map(self.self_play, jobs)
            pool.close()
            pool.join()
            self.training_data = np.concatenate([self.training_data] + new_data, axis=0)
        else:
            for _ in range(self.num_games): # Self-play games
                new_data = self.self_play(temperature)
                self.training_data = np.concatenate([self.training_data, new_data], axis=0)
        if verbose:
            print("Simulating took " + str(int(time.time()-start)) + " seconds")

        # Prune oldest training samples if a buffer size limit is set.
        if self.buffer_size_limit is not None:
            self.training_data = self.training_data[-self.buffer_size_limit:,:]

        if verbose:
            print("TRAINING")
            start = time.time()
        mean_loss = None
        count = 0
        for _ in range(self.num_updates):
            self.nn.train(self.training_data)
            new_loss = self.nn.latest_loss.item()
            if mean_loss is None:
                mean_loss = new_loss
            else:
                (mean_loss*count + new_loss)/(count+1)
            count += 1
        self.error_log.append(mean_loss)

        if verbose:
            print("Training took " + str(int(time.time()-start)) + " seconds")
            print("Average train error:", mean_loss)

