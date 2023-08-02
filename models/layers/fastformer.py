import torch
import torch.nn as nn
import math
from torch import Tensor


class Fastformer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int):
        assert out_features % num_heads == 0, f"Invalid (out_features, num_heads) pair: got ({out_features}, {num_heads})"

        super(Fastformer, self).__init__()

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.num_heads: int = num_heads
        self.head_features: int = self.out_features // self.num_heads
        self.scale_factor: float = math.sqrt(self.head_features)

        self.query_transform = nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=False)
        self.query_align = nn.Linear(in_features=self.out_features, out_features=self.num_heads, bias=False)

        self.key_transform = nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=False)
        self.key_align = nn.Linear(in_features=self.out_features, out_features=self.num_heads, bias=False)

        self.output_transform = nn.Linear(in_features=self.head_features, out_features=self.head_features, bias=False)

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

    def forward(self, x: Tensor, attn_mask: Tensor = None):
        '''
        :param x: batch_size, sequence_len, in_features
        :param attn_mask: batch_size, sequence_len
        :return: batch_size, sequence_len, out_features
        '''

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(dim=1)      # batch_size, 1, sequence_len

        query = self.query_transform(x)     # batch_size, sequence_len, out_features
        key = self.key_transform(x)         # batch_size, sequence_len, out_features
        value = query                       # batch_size, sequence_len, out_features

        query_heads = self.split_heads_features(x=query)      # batch_size, num_heads, sequence_len, head_features
        key_heads = self.split_heads_features(x=key)          # batch_size, num_heads, sequence_len, head_features
        value_heads = query_heads                               # batch_size, num_heads, sequence_len, head_features

        query_alignment_score = self.query_align(query)     # batch_size, sequence_len, num_heads
        query_alignment_score_heads = query_alignment_score.transpose(dim0=1, dim1=2)   # batch_size, num_heads, sequence_len
        query_alignment_score_heads = query_alignment_score_heads / self.scale_factor   # batch_size, num_heads, sequence_len

        exp_query_alignment_score_heads = torch.exp(query_alignment_score_heads)    # batch_size, num_heads, sequence_len
        if attn_mask is not None:
            exp_query_alignment_score_heads = exp_query_alignment_score_heads * attn_mask   # batch_size, num_heads, sequence_len
        sum_exp_query_alignment_score_heads = exp_query_alignment_score_heads.sum(dim=2, keepdim=True)      # batch_size, num_heads, 1
        query_attention_weight_heads = exp_query_alignment_score_heads / (sum_exp_query_alignment_score_heads + 1e-8)    # batch_size, num_heads, sequence_len

        weighted_query_heads = query_heads * query_attention_weight_heads.unsqueeze(dim=3)   # batch_size, num_heads, sequence_len, head_features
        query_context_heads = weighted_query_heads.sum(dim=2, keepdim=True)    # batch_size, num_heads, 1, head_features

        query_key_heads = key_heads * query_context_heads       # batch_size, num_heads, sequence_len, head_features
        query_key = self.merge_heads_features(x=query_key_heads)      # batch_size, sequence_len, out_features

        key_alignment_score = self.key_align(query_key)     # batch_size, sequence_len, num_heads
        key_alignment_score_heads = key_alignment_score.transpose(dim0=1, dim1=2)   # batch_size, num_heads, sequence_len
        key_alignment_score_heads = key_alignment_score_heads / self.scale_factor   # batch_size, num_heads, sequence_len

        exp_key_alignment_score_heads = torch.exp(key_alignment_score_heads)   # batch_size, num_heads, sequence_len
        if attn_mask is not None:
            exp_key_alignment_score_heads = exp_key_alignment_score_heads * attn_mask   # batch_size, num_heads, sequence_len
        sum_exp_key_alignment_score_heads = exp_key_alignment_score_heads.sum(dim=2, keepdim=True)  # batch_size, num_heads, 1
        key_attention_weight_heads = exp_key_alignment_score_heads / (sum_exp_key_alignment_score_heads + 1e-8)     # batch_size, num_heads, sequence_len

        weighted_key_heads = query_key_heads * key_attention_weight_heads.unsqueeze(dim=3)      # batch_size, num_heads, sequence_len, head_features
        key_context_heads = weighted_key_heads.sum(dim=2, keepdim=True)     # batch_size, num_heads, 1, head_features

        key_value_heads = value_heads * key_context_heads       # batch_size, num_heads, sequence_len, head_features
        key_value_heads = self.output_transform(key_value_heads)    # batch_size, num_heads, sequence_len, head_features

        key_value = self.merge_heads_features(x=key_value_heads)      # batch_size, sequence_len, out_features

        return value + key_value
