# Journal Article
https://arxiv.org/abs/1910.13012

# What is this
An independent implementation of DeepMind's AlphaZero algorithm with support for multiplayer games.
AlphaZero is a deep reinforcement learning algorithm which can learn to master a certain class of adversarial games through self-play.
Strategies are learned *tabula rasa* and, with enough time and computation, achieve super-human performance.
The canonical AlphaZero algorithm is intended for 2-player games like Chess and Go, though this project supports multiplayer games as well.

# Benefits

### Clean & Simple
Clear and concise - a no-frills AlphaZero implementation written with Python 3 and PyTorch.
Extensively commented and easy to extend.
Support for CPU and GPU training, as well as pausing and resuming training.

### Modular & Extensible
Easily plug in your own games or neural networks by implementing the Game and Model interface. Only PyTorch models are supported. Example games and networks included with this repo are listed below.

#### Example Games
- Tic-Tac-Toe
- Tic-Tac-Mo
- Connect 3x3

#### Example Networks
- SENet

### Novel Multiplayer Support
The first of its kind; support for games with more than 2 players.


# Tutorial

## Training an agent
You can start training your own AlphaZero agents very easily.

`python main.py configs/my_run_configuration.json`

You must run main.py with the location of a run configuration json file as the first argument.
