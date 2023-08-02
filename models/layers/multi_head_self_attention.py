import torch
import torch.nn as nn
from torch import Tensor
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int):
        assert out_features % num_heads == 0, f"Invalid (out_features, num_heads) pair: got ({out_features}, {num_heads})"

        super(MultiHeadSelfAttention, self).__init__()

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.num_heads: int = num_heads
        self.head_features: int = self.out_features // self.num_heads
        self.scale_factor: float = math.sqrt(self.head_features)

        self.query_transform = nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=False)
        self.key_transform = nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=False)
        self.value_transform = nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=False)

    def split_heads_features(self, x: Tensor) -> Tensor:
        '''

        :param x: batch_size, sequence_len, out_features
        :return: batch_size, num_heads, sequence_len, head_features
        '''
        batch_size, sequence_len, out_features = x.shape
        x = x.reshape(batch_size, sequence_len, self.num_heads, self.head_features)     # batch_size, sequence_len, num_heads, head_features
        x = x.permute(dims=(0, 2, 1, 3))    # batch_size, num_heads, sequence_len, head_features
        return x

    def merge_heads_features(self, x: Tensor) -> Tensor:
        '''

        :param x: batch_size, num_heads, sequence_len, head_features
        :return: batch_size, sequence_len, out_features
        '''
        batch_size, num_heads, sequence_len, head_features = x.shape
        x = x.permute(dims=(0, 2, 1, 3))    # batch_size, sequence_len, num_heads, head_features
        x = x.reshape(batch_size, sequence_len, self.out_features)      # batch_size, sequence_len, out_features
        return x

    def forward(self, x: Tensor, attn_mask: Tensor = None, return_full: bool = False):
        '''

        :param x: batch_size, sequence_len, in_features
        :param attn_mask: batch_size, sequence_len, sequence_len
        :param return_full: bool
        :return: batch_size, sequence_len, in_features
        '''

        query = self.query_transform(x)     # batch_size, sequence_len, out_features
        key = self.key_transform(x)         # batch_size, sequence_len, out_features
        value = self.value_transform(x)     # batch_size, sequence_len, out_features

        query_heads = self.split_heads_features(x=query)      # batch_size, num_heads, sequence_len, head_features
        key_heads = self.split_heads_features(x=key)          # batch_size, num_heads, sequence_len, head_features
        value_heads = self.split_heads_features(x=value)      # batch_size, num_heads, sequence_len, head_features

        query_key_heads = torch.matmul(query_heads, key_heads.transpose(dim0=2, dim1=3))    # batch_size, num_heads, sequence_len, sequence_len
        scaled_query_key_heads = query_key_heads / self.scale_factor    # batch_size, num_heads, sequence_len, sequence_len

        exp_query_key_heads = torch.exp(scaled_query_key_heads)     # batch_size, num_heads, sequence_len, sequence_len
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(dim=1)     # batch_size, 1, sequence_len, sequence_len
            exp_query_key_heads = exp_query_key_heads * attn_mask       # batch_size, num_heads, sequence_len, sequence_len
        sum_exp_query_key_heads = exp_query_key_heads.sum(dim=3, keepdim=True)      # batch_size, num_heads, sequence_len, 1
        attention_weight_heads = exp_query_key_heads / (sum_exp_query_key_heads + 1e-8)     # batch_size, num_heads, sequence_len, sequence_len

        context_heads = torch.matmul(attention_weight_heads, value_heads)     # batch_size, num_heads, sequence_len, head_features
        context = self.merge_heads_features(x=context_heads)      # batch_size, sequence_len, out_features

        if return_full:
            return context, query, key, value
        else:
            return context
