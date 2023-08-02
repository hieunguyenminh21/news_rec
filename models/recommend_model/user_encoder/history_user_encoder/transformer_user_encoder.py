from torch import Tensor
from .base_history_user_encoder import BaseHistoryUserEncoder
from models.layers import Transformer, AdditiveAttention
import torch.nn as nn

"""
class TransformerUserEncoder(BaseHistoryUserEncoder):
    def __init__(self, out_features: int, num_heads: int):
        super(TransformerUserEncoder, self).__init__(out_features=out_features)
        self.transformer = Transformer(in_features=out_features, out_features=out_features, num_heads=num_heads)
        self.additive_attention = AdditiveAttention(in_features=out_features)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        '''

        :param x: batch_size, sequence_len, out_features
        :param attention_mask: batch_size, sequence_len
        :return: batch_size, out_features
        '''
        batch_size, sequence_len = x.shape[:2]
        if attention_mask is not None:
            attn_mask_expand = attention_mask.unsqueeze(dim=1)  # batch_size, 1, sequence_len
            attn_mask_expand = attn_mask_expand.repeat(1, sequence_len, 1)  # batch_size, sequence_len, sequence_len
        else:
            attn_mask_expand = None

        matrix = self.transformer(x=x, attn_mask=attn_mask_expand)   # batch_size, sequence_len, in_features

        context = self.additive_attention(x=matrix, attn_mask=attention_mask)     # batch_size, in_features

        return context
"""


class TransformerUserEncoder(BaseHistoryUserEncoder):
    def __init__(self, out_features: int, num_heads: int, query_vector_dim: int):
        super(TransformerUserEncoder, self).__init__(out_features=out_features)
        self.transformer = Transformer(in_features=out_features, out_features=out_features, num_heads=num_heads)
        self.additive_attention = AdditiveAttention(in_features=out_features, query_vector_dim=query_vector_dim)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        '''

        :param x: batch_size, sequence_len, out_features
        :param attention_mask: batch_size, sequence_len
        :return: batch_size, out_features
        '''
        batch_size, sequence_len = x.shape[:2]
        if attention_mask is not None:
            attn_mask_expand = attention_mask.unsqueeze(dim=1)  # batch_size, 1, sequence_len
            attn_mask_expand = attn_mask_expand.repeat(1, sequence_len, 1)  # batch_size, sequence_len, sequence_len
        else:
            attn_mask_expand = None

        matrix = self.transformer(x=x, attn_mask=attn_mask_expand)   # batch_size, sequence_len, in_features

        context = self.additive_attention(x=matrix, attn_mask=attention_mask)     # batch_size, in_features

        return context