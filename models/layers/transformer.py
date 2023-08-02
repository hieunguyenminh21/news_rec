import torch.nn as nn
from models.layers import MultiHeadSelfAttention
from torch import Tensor


class Transformer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int, drop_out: float=0.2):
        super(Transformer, self).__init__()

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.num_heads: int = num_heads

        self.self_attention = MultiHeadSelfAttention(in_features=self.in_features, out_features=self.out_features, num_heads=self.num_heads)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.out_features)

        self.feed_forward = nn.Sequential(nn.Linear(in_features=self.out_features, out_features=self.out_features),
                                          nn.ReLU(),
                                          nn.Linear(in_features=self.out_features, out_features=self.out_features))
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=out_features)
        
        self.drop_layer = nn.Dropout(p=drop_out)

    def forward(self, x: Tensor, attn_mask: Tensor = None):
        '''

        :param x: batch_size, sequence_len, in_features
        :param attn_mask: batch_size, sequence_len, sequence_len
        :return:    # batch_size, sequence_len, out_features
        '''
        context, query, key, value = self.self_attention(x=x, attn_mask=attn_mask, return_full=True)   # batch_size, sequence_len, out_features
        context = self.drop_layer(context)
        add_1 = context + query     # batch_size, sequence_len, out_features
        norm_1 = self.layer_norm_1(add_1)      # batch_size, sequence_len, out_features

        ff = self.feed_forward(norm_1)      # batch_size, sequence_len, out_features
        ff = self.drop_layer(ff)
        add_2 = norm_1 + ff     # batch_size, sequence_len, out_features
        norm_2 = self.layer_norm_2(add_2)       # batch_size, sequence_len, out_features

        return norm_2
