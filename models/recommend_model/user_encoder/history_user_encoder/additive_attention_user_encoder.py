from torch import Tensor
from .base_history_user_encoder import BaseHistoryUserEncoder
from models.layers import AdditiveAttention
import torch.nn as nn

"""
class AdditiveAttentionUserEncoder(BaseHistoryUserEncoder):
    def __init__(self, out_features: int):
        super(AdditiveAttentionUserEncoder, self).__init__(out_features=out_features)
        self.additive_attention = AdditiveAttention(in_features=out_features)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        '''

        :param x: batch_size, sequence_len, out_features
        :param attention_mask: batch_size, sequence_len
        :return: batch_size, out_features
        '''
        context = self.additive_attention(x=x, attn_mask=attention_mask)     # batch_size, in_features

        return context
"""


class AdditiveAttentionUserEncoder(BaseHistoryUserEncoder):
    def __init__(self, out_features: int, query_vector_dim: int):
        super(AdditiveAttentionUserEncoder, self).__init__(out_features=out_features)
        self.additive_attention = AdditiveAttention(in_features=out_features, query_vector_dim=query_vector_dim)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        '''

        :param x: batch_size, sequence_len, out_features
        :param attention_mask: batch_size, sequence_len
        :return: batch_size, out_features
        '''
        context = self.additive_attention(x=x, attn_mask=attention_mask)     # batch_size, in_features

        return context