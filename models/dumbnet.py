import sys
import torch
import numpy as np
sys.path.append("..")
from model import Model


class DumbNet(Model):


  def forward(self, x):
    batch_size = len(x)
    this_p_shape = tuple([batch_size] + list(self.p_shape))
    this_v_shape = tuple([batch_size] + list(self.v_shape))

    p_logits = torch.ones(this_p_shape)
    v = torch.zeros(this_v_shape)

    return p_logits, v
