import torch
import torch.nn
from torch.nn import functional as F
import numpy as np
import os
from options import args_parser

"""
The different uncertainty methods loss implementation.
Including:
    Ignore, Zeros, Ones, SelfTrained, MultiClass
"""

METHODS = ['U-Ignore', 'U-Zeros', 'U-Ones', 'U-SelfTrained', 'U-MultiClass']
CLASS_NUM = [141, 927, 679, 1125, 2136]
args = args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
CLASS_WEIGHT = torch.Tensor([5000 / i for i in CLASS_NUM]).cuda()


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    # input_softmax = F.softmax(input_logits, dim=1)
    # target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_logits - target_logits) ** 2
    return mse_loss
