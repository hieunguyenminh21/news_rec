from torch import Tensor
import torch
import torch.nn as nn


class MaskedSoftmax1d(nn.Module):
    """
    Masked softmax on 1d data
    """
    def forward(
            self, x: Tensor, attn_mask: Tensor
    ) -> Tensor:
        """
        Compute masked softmax
        :param x: batch_size, num_features
        :param attn_mask: batch_size, num_features
        :return: batch_size, num_features
        """
        max_x, _ = x.max(dim=1, keepdim=True)    # batch_size, 1
        exp_x = torch.exp(
            x - max_x
        ) * attn_mask    # batch_size, num_features
        softmax = exp_x / (
            exp_x.sum(dim=1, keepdims=True) + 1e-9
        )    # batch_size, num_features
        return softmax
