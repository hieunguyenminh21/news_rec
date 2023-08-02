from torch import Tensor
from models.layers import AdditiveAttention, Fastformer
from .base_raw_text_encoder import BaseRawTextEncoder
import torch.nn as nn


class FastformerTextEncoder(BaseRawTextEncoder):
    def __init__(self, out_features: int, word_embedding: nn.Embedding, num_heads: int, query_vector_dim: int, drop_out: float):
        super(FastformerTextEncoder, self).__init__(out_features=out_features, word_embedding=word_embedding)
        self.drop_out = nn.Dropout(p=drop_out)
        self.fastformer = Fastformer(in_features=word_embedding.embedding_dim, out_features=out_features, num_heads=num_heads)
        self.additive_attention = AdditiveAttention(in_features=out_features, query_vector_dim=query_vector_dim)

    def forward(self, token_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        '''
        :param token_ids: batch_size, length
        :param attention_mask: batch_size, length
        :return: batch_size, out_features
        '''
        word_embedding_matrix = self.word_embedding(token_ids)       # batch_size, sequence_len, embedding_dim
        word_embedding_matrix = self.drop_out(word_embedding_matrix)     # batch_size, sequence_len, embedding_dim

        word_context_matrix = self.fastformer(x=word_embedding_matrix, attn_mask=attention_mask)   # batch_size, sequence_len, embedding_dim
        word_context_matrix = self.drop_out(word_context_matrix)

        context = self.additive_attention(x=word_context_matrix, attn_mask=attention_mask)   # batch_size, embedding_dim
        return context
