from torch import Tensor
from .base_raw_text_encoder import BaseRawTextEncoder
from models.layers import MultiHeadSelfAttention, AdditiveAttention
import torch.nn as nn


class MultiHeadSelfAttentionTextEncoder(BaseRawTextEncoder):
    def __init__(self, out_features: int, word_embedding: nn.Embedding, num_heads: int, query_vector_dim: float, drop_out: float):
        super(MultiHeadSelfAttentionTextEncoder, self).__init__(out_features=out_features, word_embedding=word_embedding)
        self.drop_out = nn.Dropout(p=drop_out)
        self.self_attention = MultiHeadSelfAttention(in_features=word_embedding.embedding_dim, out_features=out_features, num_heads=num_heads)
        self.additive_attention = AdditiveAttention(in_features=out_features, query_vector_dim=query_vector_dim)

    def forward(self, token_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        '''
        :param token_ids: batch_size, length
        :param attention_mask: batch_size, length
        :return: batch_size, out_features
        '''
        batch_size, sequence_len = token_ids.shape
        if attention_mask is not None:
            attn_mask_expand = attention_mask.unsqueeze(dim=1)       # batch_size, 1, sequence_len
            attn_mask_expand = attn_mask_expand.repeat(1, sequence_len, 1)      # batch_size, sequence_len, sequence_len
        else:
            attn_mask_expand = None

        word_embedding_matrix = self.word_embedding(token_ids)       # batch_size, sequence_len, embedding_dim
        word_embedding_matrix = self.drop_out(word_embedding_matrix)    # batch_size, sequence_len, embedding_dim

        word_context_matrix = self.self_attention(x=word_embedding_matrix, attn_mask=attn_mask_expand)   # batch_size, sequence_len, embedding_dim
        word_context_matrix = self.drop_out(word_context_matrix)

        context = self.additive_attention(x=word_context_matrix, attn_mask=attention_mask)   # batch_size, embedding_dim

        return context