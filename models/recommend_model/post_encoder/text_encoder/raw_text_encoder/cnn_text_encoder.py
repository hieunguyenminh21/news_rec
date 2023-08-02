from torch import Tensor
from models.layers import AdditiveAttention
from .base_raw_text_encoder import BaseRawTextEncoder
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F


class CNNTextEncoder(BaseRawTextEncoder):
    def __init__(self, out_features: int, word_embedding: nn.Embedding, kernel_size: int, query_vector_dim: int, drop_out: float):
        super(CNNTextEncoder, self).__init__(out_features=out_features, word_embedding=word_embedding)

        padding: Tuple[int, int] = (int((kernel_size - 1) / 2), 0)
        word_embedding_dim: int = word_embedding.embedding_dim
        self.conv = nn.Conv2d(in_channels=1, out_channels=out_features, kernel_size=(kernel_size, word_embedding_dim), padding=padding)

        self.additive_attention = AdditiveAttention(in_features=out_features, query_vector_dim=query_vector_dim)
        self.drop_layer = nn.Dropout(p=drop_out)
    
    def forward(self, token_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        '''
        :param token_ids: batch_size, length
        :param attention_mask: batch_size, length
        :return: batch_size, out_features
        '''
        emb = self.word_embedding( )     # batch_size, length, embedding_dim
        emb = self.drop_layer(emb)      # batch_size, length, embedding_dim
        emb = emb.unsqueeze(dim=1)      # batch_size, 1, length, embedding_dim

        conv = self.conv(emb)    # batch_size, out_features, length, 1
        conv = conv.squeeze(dim=3)  # batch_size, out_features, length
        conv = F.relu(conv)     # batch_size, out_features, length
        conv = self.drop_layer(conv)    # batch_size, out_features, length
        conv = conv.transpose(dim0=1, dim1=2)   # batch_size, length, out_features

        context = self.additive_attention(x=conv, attn_mask=attention_mask)     # batch_size, out_features
        
        return context
