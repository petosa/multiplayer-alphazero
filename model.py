import torch

# Interface for defining a PyTorch model.
# See the models folder for examples.
class Model(torch.nn.Module):

  def __init__(self, input_shape, p_shape, v_shape):
    super(Model, self).__init__()
    self.input_shape = input_shape
    self.p_shape = p_shape
    self.v_shape = v_shape

  # Simply define the forward pass.
  # Your input will be a batch of ndarrays representing board state.
  # Your output must have both a policy and value head, so you must return 2 tensors p, v in that order.
  # The policy head must be logits, and the value head must be passed through a tanh non-linearity.
  # Policy softmaxing is handled for you in neural_network.py.
  def forward(self, x):
    raise NotImplementedError
