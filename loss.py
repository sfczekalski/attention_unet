import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def normalize(x):
    return x / 255.0


def dice_loss(input, target):
    # target.size() -> [n, h, w] \ in {0, 1}
    inter = (input * target).sum(-1).sum(-1)
    union = input.sum(-1).sum(-1) + target.sum(-1).sum(-1)
    result = (2 * inter / union).mean()
    return result
