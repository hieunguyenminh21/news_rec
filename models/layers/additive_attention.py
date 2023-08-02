import torch
import torch.nn as nn
from torch import Tensor


class AdditiveAttention(nn.Module):
    def __init__(self, in_features: int, query_vector_dim: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Sequential(nn.Linear(in_features=in_features, out_features=query_vector_dim),
                                     nn.Tanh())
        self.att_fc2 = nn.Linear(in_features=query_vector_dim, out_features=1)

    def forward(self, x: Tensor, attn_mask: Tensor = None) -> Tensor:
        '''
        Compute context vector and attn weight for elements
        :param x: batch_size, sequence_len, in_features
        :param attn_mask: batch_size, sequence_len
        :return: batch_size, in_features
        '''
        e = self.att_fc1(x)    # batch_size, sequence_len, query_dim
        e = self.att_fc2(e)     # batch_size, sequence_len, 1
        e = e.squeeze(dim=2)     # batch_size, sequence_len

        alpha = torch.exp(e)    # batch_size, sequence_len
        if attn_mask is not None:
            alpha = alpha * attn_mask   # batch_size, sequence_len
        sum_alpha = alpha.sum(dim=1, keepdim=True)    # batch_size, 1
        alpha = alpha / (sum_alpha + 1e-8)  # batch_size, sequence_len

        context = torch.bmm(alpha.unsqueeze(dim=1), x)     # batch_size, 1, in_features
        context = context.squeeze(dim=1)    # batch_size, in_features

        return context
