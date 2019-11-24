import json
import sys
import numpy as np

from models.senet import SENet
from models.senetbig import SENetBig
from games.tictactoe import TicTacToe
from games.tictacmo import TicTacMo
from games.connect3x3 import Connect3x3
from neural_network import NeuralNetwork
from trainer import Trainer
from experiments import evaluate_against_uninformed


# Load in a run configuration
with open(sys.argv[1], "r") as f:
    config = json.loads(f.read())

# Instantiate
game = globals()[config["game"]]()
model_class = globals()[config["model"]]
sims = config["num_simulations"]
cuda = config["cuda"]
nn = NeuralNetwork(game=game, model_class=model_class, lr=config["lr"],
    weight_decay=config["weight_decay"], batch_size=config["batch_size"], cuda=cuda)
trainer = Trainer(game=game, nn=nn, num_simulations=sims,
num_games=config["num_games"], num_updates=config["num_updates"], 
buffer_size_limit=config["buffer_size_limit"], cpuct=config["cpuct"],
num_threads=config["num_threads"])

# Logic for resuming training
checkpoints = nn.list_checkpoints()
if config["resume"]:
    if len(checkpoints) == 0:
        print("No existing checkpoints to resume.")
        quit()
    iteration = int(checkpoints[-1])
    trainer.training_data, trainer.error_log = nn.load(iteration, load_supplementary_data=True)
else:
    if len(checkpoints) != 0:
        print("Please delete the existing checkpoints for this game+model combination, or change resume flag to True.")
        quit()
    iteration = 0

# Training loop
while True:

    # Run multiple policy iterations to develop a checkpoint.
    for _ in range(config["ckpt_frequency"]):
        if config["verbose"]: print("Iteration:",iteration)
        trainer.policy_iteration(verbose=config["verbose"]) # One iteration of PI
        iteration += 1
        if config["verbose"]: print("Training examples:", len(trainer.training_data))
    
    # Save checkpoint
    nn.save(name=iteration, training_data=trainer.training_data, error_log=trainer.error_log)

    # Evaluate how the current checkpoint performs against MCTS agents of increasing strength
    # that do no use a heursitic.
    for opponent_strength in [10, 20, 40, 80, 160, 320, 640, 1280]:
        evaluate_against_uninformed(checkpoint=iteration, game=game, model_class=model_class,
            my_sims=sims, opponent_sims=opponent_strength, cuda=cuda)
