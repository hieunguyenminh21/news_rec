from torch import Tensor
from .base_history_user_encoder import BaseHistoryUserEncoder
from models.layers import Fastformer, AdditiveAttention
import torch.nn as nn

"""
class FastformerUserEncoder(BaseHistoryUserEncoder):
    def __init__(self, out_features: int, num_heads: int):
        super(FastformerUserEncoder, self).__init__(out_features=out_features)
        self.fastformer = Fastformer(in_features=out_features, out_features=out_features, num_heads=num_heads)
        self.additive_attention = AdditiveAttention(in_features=out_features)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        '''

        :param x: batch_size, sequence_len, out_features
        :param attention_mask: batch_size, sequence_len
        :return: batch_size, out_features
        '''

        matrix = self.fastformer(x=x, attn_mask=attention_mask)   # batch_size, sequence_len, in_features

        context = self.additive_attention(x=matrix, attn_mask=attention_mask)     # batch_size, in_features

        return context
"""
    
    
class FastformerUserEncoder(BaseHistoryUserEncoder):
    def __init__(self, out_features: int, num_heads: int, query_vector_dim: int):
        super(FastformerUserEncoder, self).__init__(out_features=out_features)
        self.fastformer = Fastformer(in_features=out_features, out_features=out_features, num_heads=num_heads)
        self.additive_attention = AdditiveAttention(in_features=out_features, query_vector_dim=query_vector_dim)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        '''

        :param x: batch_size, sequence_len, out_features
        :param attention_mask: batch_size, sequence_len
        :return: batch_size, out_features
        '''

        matrix = self.fastformer(x=x, attn_mask=attention_mask)   # batch_size, sequence_len, in_features

        context = self.additive_attention(x=matrix, attn_mask=attention_mask)     # batch_size, in_features

        return context