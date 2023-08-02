import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, x: Tensor, target: Tensor):
        '''

        :param x: batch_size, num_candidates
        :param target: batch_size, num_candidates
        :return: scalar
        '''

        assert x.shape == target.shape

        positive = x[:, 0]      # batch_size
        positive = positive.unsqueeze(dim=1)    # batch_size, 1
        negative = x[:, 1:]     # batch_size, num_candidates - 1
        diff = positive - negative      # batch_size, num_candidates - 1
        inv_loss = F.logsigmoid(diff)       # batch_size, num_candidates - 1
        inv_loss = torch.sum(inv_loss)      # scalar
        loss = -inv_loss    # scalar
        return loss
